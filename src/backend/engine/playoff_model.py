import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import warnings
warnings.filterwarnings("ignore")

import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from data.sql_helpers import get_connection
from engine.model import load_training_data, FEATURE_COLS

# ── Config ────────────────────────────────────────────────────────────────────
PLAYOFF_C        = 0.2105
PLAYOFF_SOLVER   = 'lbfgs'
PLAYOFF_MAX_ITER = 1000
MIN_TRAIN_SEASONS = 5

PLAYOFF_FEATURES = [
    'elo_diff', 'qb_epa_diff', 'roll_pass_epa_diff',
    'roll_rush_epa_diff', 'roll_first_downs_diff',
    'clutch_rate_diff', 'qb_out_diff', 'roll_def_sacks_diff',
]

MODEL_DIR   = Path(__file__).resolve().parent
MODEL_PATH  = MODEL_DIR / "trained_playoff_model.pkl"
SCALER_PATH = MODEL_DIR / "trained_playoff_scaler.pkl"


# ── Data helpers ──────────────────────────────────────────────────────────────
def get_playoff_ids() -> set:
    with get_connection() as conn:
        df = pd.read_sql("""
            SELECT game_id FROM schedules
            WHERE game_type IN ('WC','DIV','CON','SB')
        """, conn)
    return set(df["game_id"].tolist())


def get_playoff_metadata() -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql("""
            SELECT game_id, game_type, season, week, home_team, away_team
            FROM schedules
            WHERE game_type IN ('WC','DIV','CON','SB')
            ORDER BY season, week
        """, conn)


def load_playoff_training_data() -> tuple:
    df          = load_training_data()
    playoff_ids = get_playoff_ids()
    playoff_df  = df[df["game_id"].isin(playoff_ids)].copy()

    if playoff_df.empty:
        raise ValueError("No playoff games found in training data")

    X = playoff_df[PLAYOFF_FEATURES].fillna(0)
    y = (playoff_df["result"] > 0).astype(int).values
    return X, y, playoff_df, df


# ── Hyperparameter tuning ─────────────────────────────────────────────────────
def tune_playoff_model(X: pd.DataFrame, y: np.ndarray) -> dict:
    print(f"\n{'='*55}")
    print(f"STEP 1 — HYPERPARAMETER TUNING")
    print(f"{'='*55}")
    print(f"  Dataset: {len(X)} playoff games, {len(PLAYOFF_FEATURES)} features")

    C_values = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.5, 1.0]
    configs  = [
        ("lbfgs",     "l2"),
        ("liblinear", "l1"),
        ("liblinear", "l2"),
    ]

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_score  = 0.0
    best_params = {"C": PLAYOFF_C, "solver": PLAYOFF_SOLVER, "penalty": "l2"}
    all_results = []

    print(f"\n  {'C':>8} {'Solver':<12} {'Penalty':<8} {'CV Acc':>8} {'Std':>6}")
    print(f"  {'-'*48}")

    for solver, penalty in configs:
        for C in C_values:
            try:
                model  = LogisticRegression(
                    C=C, solver=solver, penalty=penalty,
                    max_iter=2000, random_state=42,
                )
                scores = cross_val_score(model, X_scaled, y, cv=5, scoring="accuracy")
                mean_s = scores.mean()
                std_s  = scores.std()
                all_results.append({
                    "C": C, "solver": solver, "penalty": penalty,
                    "score": mean_s, "std": std_s,
                })
                marker = "  ← best" if mean_s > best_score else ""
                print(f"  {C:>8.3f} {solver:<12} {penalty:<8} "
                      f"{mean_s*100:>7.2f}%  {std_s*100:>5.2f}%{marker}")
                if mean_s > best_score:
                    best_score  = mean_s
                    best_params = {"C": C, "solver": solver, "penalty": penalty}
            except Exception:
                continue

    print(f"\n  Best: C={best_params['C']}, solver={best_params['solver']}, "
          f"penalty={best_params['penalty']}")
    print(f"  Best CV accuracy: {best_score*100:.2f}%")

    top5 = sorted(all_results, key=lambda x: x["score"], reverse=True)[:5]
    print(f"\n  Top 5 configurations:")
    for r in top5:
        print(f"    C={r['C']:<6}  {r['solver']:<12}  {r['penalty']:<6}  "
              f"{r['score']*100:.2f}%  ±{r['std']*100:.2f}%")

    return best_params


# ── Training ──────────────────────────────────────────────────────────────────
def train_playoff_model(C: float, solver: str) -> tuple:
    print(f"\n{'='*55}")
    print(f"STEP 2 — TRAINING")
    print(f"{'='*55}")
    print(f"  C={C}, solver={solver}")

    X, y, _, _ = load_playoff_training_data()

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(
        C=C, solver=solver, max_iter=PLAYOFF_MAX_ITER, random_state=42,
    )
    model.fit(X_scaled, y)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    print(f"  Training accuracy: {model.score(X_scaled, y)*100:.1f}%")
    print(f"  Saved to {MODEL_PATH}")

    print(f"\n  Feature coefficients:")
    coefs = sorted(zip(PLAYOFF_FEATURES, model.coef_[0]),
                   key=lambda x: abs(x[1]), reverse=True)
    for feat, coef in coefs:
        bar = "█" * int(abs(coef) * 12)
        d   = "↑ home" if coef > 0 else "↓ away"
        print(f"  {feat:<38} {coef:>+.3f}  {bar} {d}")

    return model, scaler


# ── Walk-forward validation ───────────────────────────────────────────────────
def validate_playoff_model(C: float, df: pd.DataFrame) -> None:
    print(f"\n{'='*55}")
    print(f"STEP 3 — WALK-FORWARD VALIDATION")
    print(f"{'='*55}")

    playoff_ids  = get_playoff_ids()
    playoff_meta = get_playoff_metadata()
    seasons      = sorted(df["season"].unique())
    results      = []

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
        if len(train_po) < 30:
            continue

        X_tr = train_po[PLAYOFF_FEATURES].fillna(0)
        y_tr = (train_po["result"] > 0).astype(int)
        X_te = test_playoff[PLAYOFF_FEATURES].fillna(0)

        scaler = StandardScaler()
        model  = LogisticRegression(
            C=C, solver=PLAYOFF_SOLVER,
            max_iter=PLAYOFF_MAX_ITER, random_state=42,
        )
        model.fit(scaler.fit_transform(X_tr), y_tr)
        probs  = model.predict_proba(scaler.transform(X_te))[:, 1]
        preds  = (probs >= 0.5).astype(int)
        actual = (test_playoff["result"] > 0).astype(int).values

        for j, (_, game) in enumerate(test_playoff.iterrows()):
            meta  = playoff_meta[playoff_meta["game_id"] == game["game_id"]]
            gtype = meta.iloc[0]["game_type"] if not meta.empty else "?"
            results.append({
                "season":          test_season,
                "game_type":       gtype,
                "home_team":       game["home_team"],
                "away_team":       game["away_team"],
                "result":          game["result"],
                "vegas_spread":    game.get("vegas_spread"),
                "playoff_prob":    round(float(probs[j]), 3),
                "playoff_winner":  game["home_team"] if preds[j] == 1 else game["away_team"],
                "actual_winner":   game["home_team"] if game["result"] > 0 else game["away_team"],
                "vegas_winner":    game["home_team"] if (game.get("vegas_spread") or 0) > 0 else game["away_team"],
                "correct":         int(preds[j] == actual[j]),
                "vegas_correct":   int(
                    ((game.get("vegas_spread") or 0) > 0) == (game["result"] > 0)
                ) if game.get("vegas_spread") is not None else 0,
            })

    if not results:
        print("  No validation results")
        return

    res       = pd.DataFrame(results)
    total     = len(res)
    acc       = res["correct"].mean() * 100
    vegas_acc = res["vegas_correct"].mean() * 100
    agree     = res["playoff_winner"] == res["vegas_winner"]
    agree_acc = res[agree]["correct"].mean() * 100
    hc        = res[(res["playoff_prob"] >= 0.65) | (res["playoff_prob"] <= 0.35)]
    hc_acc    = hc["correct"].mean() * 100 if len(hc) > 0 else 0

    print(f"  Prob range: {res['playoff_prob'].min():.2f} — {res['playoff_prob'].max():.2f}")
    print(f"\n  Games: {total}  |  Accuracy: {acc:.1f}%  |  Vegas: {vegas_acc:.1f}%")
    print(f"  Edge over Vegas: {acc-vegas_acc:+.1f}%")
    print(f"  Both agree: {agree.sum()} games → {agree_acc:.1f}% accuracy")
    print(f"  High confidence (>0.65 or <0.35): {len(hc)} games → {hc_acc:.1f}%")

    print(f"\n  By round:")
    for gtype, label in [("WC","Wild Card"),("DIV","Divisional"),
                          ("CON","Conference"),("SB","Super Bowl")]:
        s = res[res["game_type"] == gtype]
        if len(s) > 0:
            print(f"  {label:<14} {len(s):>5} games  "
                  f"{s['correct'].mean()*100:.1f}%  "
                  f"(Vegas: {s['vegas_correct'].mean()*100:.1f}%  "
                  f"edge: {s['correct'].mean()*100 - s['vegas_correct'].mean()*100:+.1f}%)")

    print(f"\n  Super Bowl predictions:")
    print(f"  {'Season':<8} {'Home':>5} {'Away':>5} {'Prob':>6} "
          f"{'Predicted':>10} {'Actual':>10} {'':>3}")
    print(f"  {'-'*60}")
    for _, r in res[res["game_type"] == "SB"].iterrows():
        mark = "✓" if r["correct"] else "✗"
        print(f"  {int(r['season']):<8} {r['home_team']:>5} {r['away_team']:>5} "
              f"{r['playoff_prob']:>5.2f}  {r['playoff_winner']:>10}  "
              f"{r['actual_winner']:>10}  {mark}")


# ── Load / predict ────────────────────────────────────────────────────────────
def load_playoff_model() -> tuple:
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        raise FileNotFoundError(
            "Playoff model not found. Run engine/playoff_model.py to train."
        )
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


def predict_playoff_game(game_features: dict,
                          model=None, scaler=None) -> dict:
    """
    Predict win probability for a single playoff game.
    game_features: dict with keys matching PLAYOFF_FEATURES.
    Returns home_win_prob, away_win_prob, playoff_pick, confidence.
    """
    if model is None or scaler is None:
        model, scaler = load_playoff_model()

    X    = pd.DataFrame([{f: game_features.get(f, 0) for f in PLAYOFF_FEATURES}])
    prob = float(model.predict_proba(scaler.transform(X.fillna(0)))[0][1])

    return {
        "home_win_prob": round(prob, 3),
        "away_win_prob": round(1 - prob, 3),
        "playoff_pick":  "home" if prob >= 0.5 else "away",
        "confidence":    round(abs(prob - 0.5) * 2, 3),
    }


# ── Main — tune → train → validate ───────────────────────────────────────────
if __name__ == "__main__":
    print("Playoff model — tune, train, validate")
    print("=" * 55)
    print(f"  Regularization C:  {PLAYOFF_C}")
    print(f"  Features:          {len(PLAYOFF_FEATURES)}")

    print("\nLoading training data...")
    X, y, _, df_full = load_playoff_training_data()
    print(f"  {len(X)} playoff games loaded")

    best_params = tune_playoff_model(X, y)
    best_C      = best_params["C"]
    best_solver = best_params["solver"]

    train_playoff_model(best_C, best_solver)
    validate_playoff_model(best_C, df_full)

    print(f"\n{'='*55}")
    print(f"Done. Model saved to {MODEL_PATH}")
    print(f"Update PLAYOFF_C = {best_C} in playoff_model.py if desired.")