"""
playoff_model.py — Logistic regression model for playoff game predictions.

Replaces the main XGBoost spread model silently during playoff weeks (19-22).
Trained on playoff games only, using features that matter more in the
postseason (ELO, QB quality, clutch rate, defensive pressure).

Architecture:
    LogisticRegression with L2 regularization, StandardScaler preprocessing.
    Walk-forward validated on 254 playoff games across 20 seasons.

Params (Optuna-tuned, finalized):
    C = 0.2105, solver = 'lbfgs'

Features (8):
    elo_diff, qb_epa_diff, roll_pass_epa_diff, roll_rush_epa_diff,
    roll_first_downs_diff, clutch_rate_diff, qb_out_diff, roll_def_sacks_diff

Walk-forward results:
    Overall: 68.9% accuracy  (+5.5% vs Vegas)
    Wild Card: 62.0%  |  Divisional: 72.7%
    Conference: 77.3%  |  Super Bowl: 68.2%

Usage:
    Prediction only (normal operation):
        model, scaler = load_playoff_model()
        pred = predict_playoff_game(features_dict, model, scaler)

    Retraining (only needed if DB has changed significantly):
        python3.11 engine/playoff_model.py

Writes to disk:
    engine/trained_playoff_model.pkl
    engine/trained_playoff_scaler.pkl
"""
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
PLAYOFF_C        = 0.2105   # Optuna-tuned, confirmed — do not change without retuning
PLAYOFF_SOLVER   = 'lbfgs'
PLAYOFF_MAX_ITER = 1000
MIN_TRAIN_SEASONS = 5

PLAYOFF_FEATURES = [
    'elo_diff',            # Team quality differential — strongest single feature
    'qb_epa_diff',         # Starter EPA per attempt differential
    'roll_pass_epa_diff',  # Recent passing EPA trend
    'roll_rush_epa_diff',  # Recent rushing EPA trend
    'roll_first_downs_diff',
    'clutch_rate_diff',    # Close-game win rate — matters more in playoffs
    'qb_out_diff',         # Binary: opponent QB confirmed out
    'roll_def_sacks_diff', # Defensive pressure differential
]

MODEL_DIR   = Path(__file__).resolve().parent
MODEL_PATH  = MODEL_DIR / "trained_playoff_model.pkl"
SCALER_PATH = MODEL_DIR / "trained_playoff_scaler.pkl"


# ── Data helpers ──────────────────────────────────────────────────────────────
def get_playoff_ids() -> set:
    """Return set of game_ids for all playoff games in the DB."""
    with get_connection() as conn:
        df = pd.read_sql("""
            SELECT game_id FROM schedules
            WHERE game_type IN ('WC','DIV','CON','SB')
        """, conn)
    return set(df["game_id"].tolist())


def get_playoff_metadata() -> pd.DataFrame:
    """Return game_id, game_type, season, week, home_team, away_team for all playoff games."""
    with get_connection() as conn:
        return pd.read_sql("""
            SELECT game_id, game_type, season, week, home_team, away_team
            FROM schedules
            WHERE game_type IN ('WC','DIV','CON','SB')
            ORDER BY season, week
        """, conn)


def load_playoff_training_data() -> tuple:
    """
    Load training data filtered to playoff games only.

    Returns:
        tuple: (X, y, playoff_df, full_df)
            X: feature matrix (PLAYOFF_FEATURES columns)
            y: binary labels (1 = home team won)
            playoff_df: full DataFrame of playoff rows
            full_df: complete training data (needed for walk-forward)
    """
    df          = load_training_data()
    playoff_ids = get_playoff_ids()
    playoff_df  = df[df["game_id"].isin(playoff_ids)].copy()

    if playoff_df.empty:
        raise ValueError("No playoff games found in training data")

    X = playoff_df[PLAYOFF_FEATURES].fillna(0)
    y = (playoff_df["result"] > 0).astype(int).values
    return X, y, playoff_df, df


# ── Training ──────────────────────────────────────────────────────────────────
def train_playoff_model(C: float = PLAYOFF_C, solver: str = PLAYOFF_SOLVER) -> tuple:
    """
    Train logistic regression on all historical playoff games and save to disk.

    Args:
        C: Regularization strength (lower = more regularized).
        solver: Optimization solver ('lbfgs' or 'liblinear').

    Returns:
        tuple: (fitted LogisticRegression, fitted StandardScaler)
    """
    print(f"\n{'='*55}")
    print(f"TRAINING")
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
    """
    Run walk-forward validation across all playoff seasons.

    For each test season N: train on playoff games from seasons 1..N-1,
    predict playoff games in season N. Reports accuracy vs Vegas by round.

    Args:
        C: Regularization strength to use for validation.
        df: Full training DataFrame (includes both regular season and playoff games).
    """
    print(f"\n{'='*55}")
    print(f"WALK-FORWARD VALIDATION")
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

    print(f"  Games: {total}  |  Accuracy: {acc:.1f}%  |  Vegas: {vegas_acc:.1f}%")
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
    """
    Load trained model and scaler from disk.

    Returns:
        tuple: (LogisticRegression, StandardScaler)

    Raises:
        FileNotFoundError: If pkl files don't exist — run playoff_model.py to train.
    """
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

    Args:
        game_features: Dict with keys matching PLAYOFF_FEATURES. Missing keys → 0.
        model: Pre-loaded LogisticRegression. If None, loads from disk.
        scaler: Pre-loaded StandardScaler. If None, loads from disk.

    Returns:
        dict: home_win_prob, away_win_prob, playoff_pick ('home'|'away'), confidence.
              confidence = |prob - 0.5| × 2, so 0.0 = coin flip, 1.0 = certain.
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


# ── Main — train → validate ───────────────────────────────────────────────────
if __name__ == "__main__":
    print("Playoff model — train and validate")
    print("=" * 55)
    print(f"  C={PLAYOFF_C}, solver={PLAYOFF_SOLVER}")
    print(f"  Features: {len(PLAYOFF_FEATURES)}")

    print("\nLoading training data...")
    X, y, _, df_full = load_playoff_training_data()
    print(f"  {len(X)} playoff games loaded")

    train_playoff_model(PLAYOFF_C, PLAYOFF_SOLVER)
    validate_playoff_model(PLAYOFF_C, df_full)

    print(f"\n{'='*55}")
    print(f"Done. Model saved to {MODEL_PATH}")