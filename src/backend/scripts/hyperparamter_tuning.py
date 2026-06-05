import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import optuna
import warnings
warnings.filterwarnings("ignore")

from xgboost import XGBRegressor
from engine.model import load_training_data, FEATURE_COLS

# ── Config ────────────────────────────────────────────────────────────────────
N_TRIALS          = 1000
MIN_TRAIN_SEASONS = 5
EDGE_THRESHOLD    = 2.0
MIN_BETS          = 20
OPTUNA_SEED       = 42


# ── Objective — full walk-forward, ATS margin target ─────────────────────────
def make_objective(df: pd.DataFrame):
    seasons = sorted(df["season"].unique())

    # Pre-build filtered test sets once
    season_cache = {}
    for i, test_season in enumerate(seasons):
        if i < MIN_TRAIN_SEASONS:
            continue
        train = df[df["season"].isin(seasons[:i])]
        test  = df[df["season"] == test_season].copy()
        test  = test[~test["week"].isin([17, 18])]   # no late
        test  = test[test["div_game"] == 0]           # no div
        if len(train) < 200 or len(test) == 0:
            continue
        season_cache[test_season] = (train, test)

    print(f"  Walk-forward seasons: {len(season_cache)} "
          f"({seasons[MIN_TRAIN_SEASONS]}–{seasons[-1]})")
    print(f"  Target: ATS margin (result - vegas_spread)")
    print(f"  Each trial trains {len(season_cache)} models")

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 50, 600),
            "max_depth":        trial.suggest_int("max_depth", 2, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
            "gamma":            trial.suggest_float("gamma", 0.0, 2.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.1, 10.0),
        }

        all_ats_wins = []

        try:
            for test_season, (train_df, test_df) in season_cache.items():
                X_train = train_df[FEATURE_COLS].fillna(0)
                # ATS margin target
                y_train = train_df["result"] - train_df["vegas_spread"].fillna(0)
                X_test  = test_df[FEATURE_COLS].fillna(0)

                model = XGBRegressor(**params, random_state=OPTUNA_SEED, verbosity=0)
                model.fit(X_train, y_train)

                # Predict ATS margin, convert to spread
                ats_preds    = model.predict(X_test)
                model_spread = pd.Series(
                    ats_preds + test_df["vegas_spread"].fillna(0).values,
                    index=test_df.index
                )
                vegas_spread = test_df["vegas_spread"].fillna(0)
                spread_gap   = model_spread - vegas_spread

                agrees = (
                    ((model_spread > 0) & (vegas_spread > 0)) |
                    ((model_spread < 0) & (vegas_spread < 0))
                )
                edge_mask = agrees & (spread_gap.abs() >= EDGE_THRESHOLD)
                edge_idx  = test_df[edge_mask].index

                if len(edge_idx) == 0:
                    continue

                bet_home = spread_gap[edge_idx] > 0
                result   = test_df.loc[edge_idx, "result"].values
                vegas    = test_df.loc[edge_idx, "vegas_spread"].values
                ats_wins = np.where(bet_home, result > vegas, result < vegas)
                all_ats_wins.extend(ats_wins.tolist())

            total_bets = len(all_ats_wins)
            if total_bets < MIN_BETS:
                return 0.5

            ats_pct  = sum(all_ats_wins) / total_bets
            weighted = ats_pct * min(total_bets / 50.0, 1.0)
            return -weighted

        except Exception:
            return 0.5

    return objective


# ── Run study ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        import optuna
    except ImportError:
        print("pip install optuna")
        sys.exit(1)

    print("Loading training data...")
    df = load_training_data()
    print(f"  {len(df)} games, {len(FEATURE_COLS)} features")
    print(f"  Trials: {N_TRIALS} (parallel)")
    print(f"  Objective: ATS% walk-forward, threshold=2.0, no-div, no-late\n")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=OPTUNA_SEED),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )
    study.optimize(
        make_objective(df),
        n_trials=N_TRIALS,
        n_jobs=-1,
        show_progress_bar=True,
    )

    best       = study.best_trial
    best_score = -best.value

    print(f"\n{'='*60}")
    print(f"OPTUNA RESULTS — {N_TRIALS} trials")
    print(f"Target: ATS margin  |  Pure XGBoost  |  Vegas as feature  |  threshold=2.0, no-div, no-late")
    print(f"{'='*60}")
    print(f"Best weighted ATS score: {best_score:.4f}")
    print(f"\nBest hyperparameters:")
    for k, v in best.params.items():
        if isinstance(v, float):
            print(f"  {k:<22} {v:.4f}")
        else:
            print(f"  {k:<22} {v}")

    top5 = sorted(study.trials, key=lambda t: t.value)[:5]
    print(f"\nTop 5 trials:")
    for i, t in enumerate(top5, 1):
        print(f"  {i}. score={-t.value:.4f}  trial={t.number}")

    # Full backtest
    print(f"\nRunning full walk-forward with best params...")
    from engine.model import walk_forward_backtest, print_report, calculate_edges, \
                             print_edge_report, print_stacked_filter_report
    import engine.model as model_module

    original_make_base = model_module._make_base

    def patched_make_base():
        return XGBRegressor(**best.params, random_state=OPTUNA_SEED, verbosity=0)

    model_module._make_base = patched_make_base
    results = walk_forward_backtest(df)
    print_report(results)

    print("\nBase edges (asymmetric filter):")
    edges = calculate_edges(results)
    print_edge_report(edges, label="Best params — asymmetric filter (446 bet target)")

    print("\nStacked filter analysis:")
    print_stacked_filter_report(results)

    model_module._make_base = original_make_base

    print(f"\n{'='*60}")
    print(f"Baseline: 66.45% accuracy (ATS margin target, untuned)")
    print(f"{'='*60}")
    print(f"UPDATE _make_base() in model.py:")
    print(f"def _make_base() -> XGBRegressor:")
    print(f"    return XGBRegressor(")
    for k, v in best.params.items():
        if isinstance(v, float):
            print(f"        {k}={v:.4f},")
        else:
            print(f"        {k}={v},")
    print(f"        random_state=42, verbosity=0,")
    print(f"    )")
    print(f"\nThen run: python3.11 engine/model.py")