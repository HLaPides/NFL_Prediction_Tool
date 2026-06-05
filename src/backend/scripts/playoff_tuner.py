import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from data.sql_helpers import get_connection
from engine.model import load_training_data

# ── Config ────────────────────────────────────────────────────────────────────
N_TRIALS          = 2500   # logistic regression is fast, 500 is thorough
MIN_TRAIN_SEASONS = 5
OPTUNA_SEED       = 42

# Full candidate feature pool — Optuna will select the best subset
CANDIDATE_FEATURES = [
    "elo_diff",
    "qb_epa_diff",
    "roll_pass_epa_diff",
    "roll_rush_epa_diff",
    "roll_first_downs_diff",
    "clutch_rate_diff",
    "qb_out_diff",
    "roll_def_sacks_diff",
]


# ── Data helpers ──────────────────────────────────────────────────────────────
def get_playoff_ids() -> set:
    with get_connection() as conn:
        df = pd.read_sql("""
            SELECT game_id FROM schedules
            WHERE game_type IN ('WC','DIV','CON','SB')
        """, conn)
    return set(df["game_id"].tolist())


def get_playoff_meta() -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql("""
            SELECT game_id, game_type FROM schedules
            WHERE game_type IN ('WC','DIV','CON','SB')
        """, conn)


# ── Walk-forward accuracy ─────────────────────────────────────────────────────
def walk_forward_accuracy(df: pd.DataFrame, playoff_ids: set,
                           features: list, C: float,
                           solver: str) -> dict:
    """
    Walk-forward evaluation returning per-round accuracy.
    Weights Conference + Super Bowl more heavily since playoff model
    earns its edge there.
    """
    seasons = sorted(df["season"].unique())
    results = []

    for i, test_season in enumerate(seasons):
        train_seasons = seasons[:i]
        if len(train_seasons) < MIN_TRAIN_SEASONS:
            continue

        train        = df[df["season"].isin(train_seasons)]
        test_playoff = df[
            (df["season"] == test_season) &
            (df["game_id"].isin(playoff_ids))
        ]

        if len(test_playoff) == 0:
            continue

        train_po = train[train["game_id"].isin(playoff_ids)]
        if len(train_po) < 25:
            continue

        X_tr = train_po[features].fillna(0)
        y_tr = (train_po["result"] > 0).astype(int)
        X_te = test_playoff[features].fillna(0)

        try:
            scaler = StandardScaler()
            model  = LogisticRegression(
                C=C, solver=solver, max_iter=2000, random_state=OPTUNA_SEED
            )
            model.fit(scaler.fit_transform(X_tr), y_tr)
            preds  = model.predict(scaler.transform(X_te))
            actual = (test_playoff["result"] > 0).astype(int).values

            for j, (_, game) in enumerate(test_playoff.iterrows()):
                results.append({
                    "season":    test_season,
                    "game_id":   game["game_id"],
                    "correct":   int(preds[j] == actual[j]),
                })
        except Exception:
            continue

    if not results:
        return {"overall": 0.5, "weighted": 0.5, "n": 0}

    res_df   = pd.DataFrame(results)
    meta_df  = get_playoff_meta()
    res_df   = res_df.merge(meta_df, on="game_id", how="left")

    overall  = res_df["correct"].mean()
    n        = len(res_df)

    # Weighted accuracy — Conference and Super Bowl count more
    # because that's where the playoff model has genuine edge
    weights  = res_df["game_type"].map({
        "WC": 1.0, "DIV": 1.0, "CON": 2.0, "SB": 2.0
    }).fillna(1.0)
    weighted = (res_df["correct"] * weights).sum() / weights.sum()

    return {"overall": overall, "weighted": weighted, "n": n}


# ── Optuna objective ──────────────────────────────────────────────────────────
def make_objective(df: pd.DataFrame, playoff_ids: set):
    def objective(trial: optuna.Trial) -> float:
        # C — regularization strength
        C = trial.suggest_float("C", 0.001, 2.0, log=True)

        # Solver
        solver = trial.suggest_categorical("solver", ["lbfgs", "liblinear"])

        # Feature selection — each feature is on/off
        selected = []
        for feat in CANDIDATE_FEATURES:
            if trial.suggest_categorical(f"use_{feat}", [True, False]):
                selected.append(feat)

        # Need at least 3 features
        if len(selected) < 3:
            return 0.5

        try:
            result = walk_forward_accuracy(df, playoff_ids, selected, C, solver)
            if result["n"] < 50:
                return 0.5
            # Minimize negative weighted accuracy
            return -result["weighted"]
        except Exception:
            return 0.5

    return objective


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        import optuna
    except ImportError:
        print("Optuna not installed. Run:")
        print("  pip install optuna --break-system-packages")
        sys.exit(1)

    print("Loading training data...")
    df = load_training_data()
    print(f"  {len(df)} games")

    playoff_ids = get_playoff_ids()
    playoff_df  = df[df["game_id"].isin(playoff_ids)]
    print(f"  {len(playoff_df)} playoff games across {playoff_df['season'].nunique()} seasons")
    print(f"  Candidate features: {len(CANDIDATE_FEATURES)}")
    print(f"  Trials: {N_TRIALS}\n")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=OPTUNA_SEED),
    )
    study.optimize(
        make_objective(df, playoff_ids),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    best       = study.best_trial
    best_score = -best.value

    # Extract best config
    best_C      = best.params["C"]
    best_solver = best.params["solver"]
    best_feats  = [f for f in CANDIDATE_FEATURES
                   if best.params.get(f"use_{f}", False)]

    print(f"\n{'='*60}")
    print(f"OPTUNA RESULTS — {N_TRIALS} trials")
    print(f"Optimizing: walk-forward accuracy (CON/SB weighted 2x)")
    print(f"{'='*60}")
    print(f"Best weighted accuracy: {best_score:.4f} ({best_score*100:.2f}%)")
    print(f"\nBest C:      {best_C:.4f}")
    print(f"Best solver: {best_solver}")
    print(f"Best features ({len(best_feats)}):")
    for f in best_feats:
        print(f"  + {f}")
    removed = [f for f in CANDIDATE_FEATURES if f not in best_feats]
    for f in removed:
        print(f"  - {f}  (removed)")

    # Top 5 trials
    print(f"\nTop 5 trials:")
    top5 = sorted(study.trials, key=lambda t: t.value)[:5]
    for i, t in enumerate(top5, 1):
        feats  = [f for f in CANDIDATE_FEATURES if t.params.get(f"use_{f}", False)]
        print(f"  {i}. score={-t.value:.4f}  C={t.params['C']:.4f}  "
              f"solver={t.params['solver']}  features={len(feats)}")

    # Full walk-forward with best params
    print(f"\nRunning full walk-forward with best params...")
    seasons = sorted(df["season"].unique())
    results = []

    for i, test_season in enumerate(seasons):
        train_seasons = seasons[:i]
        if len(train_seasons) < MIN_TRAIN_SEASONS:
            continue
        train        = df[df["season"].isin(train_seasons)]
        test_playoff = df[
            (df["season"] == test_season) &
            (df["game_id"].isin(playoff_ids))
        ]
        if len(test_playoff) == 0:
            continue
        train_po = train[train["game_id"].isin(playoff_ids)]
        if len(train_po) < 25:
            continue

        X_tr = train_po[best_feats].fillna(0)
        y_tr = (train_po["result"] > 0).astype(int)
        X_te = test_playoff[best_feats].fillna(0)

        scaler = StandardScaler()
        model  = LogisticRegression(
            C=best_C, solver=best_solver, max_iter=2000, random_state=OPTUNA_SEED
        )
        model.fit(scaler.fit_transform(X_tr), y_tr)
        preds  = model.predict(scaler.transform(X_te))
        actual = (test_playoff["result"] > 0).astype(int).values

        for j, (_, game) in enumerate(test_playoff.iterrows()):
            results.append({
                "season":      test_season,
                "game_id":     game["game_id"],
                "home_team":   game["home_team"],
                "away_team":   game["away_team"],
                "result":      game["result"],
                "vegas_spread": game.get("vegas_spread"),
                "correct":     int(preds[j] == actual[j]),
                "vegas_correct": int(
                    ((game.get("vegas_spread") or 0) > 0) == (game["result"] > 0)
                ) if game.get("vegas_spread") is not None else None,
            })

    res_df   = pd.DataFrame(results)
    meta_df  = get_playoff_meta()
    res_df   = res_df.merge(meta_df, on="game_id", how="left")

    total     = len(res_df)
    acc       = res_df["correct"].mean() * 100
    vegas_res = res_df[res_df["vegas_correct"].notna()]
    vegas_acc = vegas_res["vegas_correct"].mean() * 100

    print(f"\n{'='*60}")
    print(f"WALK-FORWARD RESULTS — BEST PARAMS")
    print(f"{'='*60}")
    print(f"  Games:    {total}")
    print(f"  Accuracy: {acc:.1f}%")
    print(f"  Vegas:    {vegas_acc:.1f}%")
    print(f"  Edge:     {acc-vegas_acc:+.1f}%")

    print(f"\n  By round:")
    for gtype, label in [("WC","Wild Card"),("DIV","Divisional"),
                          ("CON","Conference"),("SB","Super Bowl")]:
        s = res_df[res_df["game_type"] == gtype]
        if len(s) > 0:
            sv    = s[s["vegas_correct"].notna()]
            s_acc = s["correct"].mean() * 100
            v_acc = sv["vegas_correct"].mean() * 100 if len(sv) > 0 else 0
            print(f"  {label:<14} {len(s):>5} games  "
                  f"{s_acc:.1f}%  (Vegas: {v_acc:.1f}%  edge: {s_acc-v_acc:+.1f}%)")

    # Compare to current model
    print(f"\n  Current model baseline:")
    print(f"  Overall 65.0%  Conference 77.3%  Super Bowl 59.1%")
    print(f"\n  New model:")
    con = res_df[res_df["game_type"] == "CON"]
    sb  = res_df[res_df["game_type"] == "SB"]
    con_acc = con["correct"].mean() * 100 if len(con) > 0 else 0
    sb_acc  = sb["correct"].mean()  * 100 if len(sb)  > 0 else 0
    print(f"  Overall {acc:.1f}%  Conference {con_acc:.1f}%  Super Bowl {sb_acc:.1f}%")

    improved = acc > 65.0
    print(f"\n  {'IMPROVEMENT — update playoff_model.py' if improved else 'No improvement — keep current params'}")

    print(f"\n{'='*60}")
    print(f"UPDATE playoff_model.py WITH:")
    print(f"{'='*60}")
    print(f"PLAYOFF_C        = {best_C:.4f}")
    print(f"PLAYOFF_SOLVER   = '{best_solver}'")
    print(f"PLAYOFF_FEATURES = [")
    for f in best_feats:
        print(f"    '{f}',")
    print(f"]")
    print(f"\nThen run: python3.11 engine/playoff_model.py")
    print(f"\nDone. No DB writes. playoff_model.py unchanged.")