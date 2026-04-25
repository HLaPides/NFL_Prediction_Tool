import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import pickle
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from data.sql_helpers import get_connection, save

# ── Constants ─────────────────────────────────────────────────────────────────
MIN_SEASON        = 1999
MIN_ATTEMPTS      = 10      # minimum attempts to count as a starting QB game
ROLL_WINDOW       = 4       # rolling average window in games
MIN_TRAIN_SEASONS = 4
STAT_TARGETS      = ["passing_yards", "passing_tds", "attempts", "completions",
                     "passing_interceptions"]

# Hyperparameter search space
PARAM_GRID = [
    {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
    {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.02, "subsample": 0.8, "colsample_bytree": 0.8},
    {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.01, "subsample": 0.8, "colsample_bytree": 0.8},
    {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.02, "subsample": 0.8, "colsample_bytree": 0.8},
    {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.02, "subsample": 0.7, "colsample_bytree": 0.7},
    {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.02, "subsample": 0.9, "colsample_bytree": 1.0},
    {"n_estimators": 400, "max_depth": 2, "learning_rate": 0.01, "subsample": 0.8, "colsample_bytree": 0.8},
]


# ── Data loading ──────────────────────────────────────────────────────────────
def load_qb_data() -> pd.DataFrame:
    with get_connection() as conn:
        # QB game logs
        qb = pd.read_sql("""
            SELECT
                p.player_id, p.player_name, p.season, p.week,
                p.team, p.opponent_team, p.game_id,
                p.passing_yards, p.passing_tds, p.attempts,
                p.completions, p.passing_interceptions,
                p.passing_epa, p.passing_air_yards,
                p.passing_cpoe, p.sacks_suffered,
                p.rushing_yards AS qb_rush_yards
            FROM player_stats p
            WHERE p.position = 'QB'
              AND p.season_type = 'REG'
              AND p.attempts >= ?
              AND p.season >= ?
            ORDER BY p.season, p.week
        """, conn, params=(MIN_ATTEMPTS, MIN_SEASON))

        # Game context
        sched = pd.read_sql("""
            SELECT s.game_id, s.season, s.week,
                   s.home_team, s.away_team,
                   COALESCE(s.home_rest, 7) AS home_rest,
                   COALESCE(s.away_rest, 7) AS away_rest,
                   CASE WHEN c.roof IN ('dome','closed') THEN 1 ELSE 0 END AS is_dome,
                   CASE WHEN CAST(COALESCE(c.wind,0) AS FLOAT) > 15 THEN 1 ELSE 0 END AS high_wind,
                   CAST(COALESCE(c.temp, 60) AS FLOAT) AS temp,
                   o.total_line,
                   o.spread_line
            FROM schedules s
            LEFT JOIN game_conditions c ON c.game_id = s.game_id
            LEFT JOIN game_odds o       ON o.game_id = s.game_id
            WHERE s.game_type = 'REG'
        """, conn)

        # Opponent defensive stats — passing yards allowed per game
        def_pass = pd.read_sql("""
            SELECT opponent_team AS def_team, season, week,
                   passing_yards AS opp_pass_yards_allowed,
                   passing_tds   AS opp_pass_tds_allowed,
                   attempts      AS opp_att_allowed
            FROM player_stats
            WHERE position = 'QB'
              AND season_type = 'REG'
              AND attempts >= ?
        """, conn, params=(MIN_ATTEMPTS,))

    # ── Merge game context onto QB logs ──────────────────────────────────────
    qb = qb.merge(sched, on=["game_id", "season", "week"], how="left")

    # Is QB playing at home?
    qb["is_home"] = (qb["team"] == qb["home_team"]).astype(int)
    qb["rest"]    = np.where(qb["is_home"] == 1, qb["home_rest"], qb["away_rest"])

    # ── Rolling QB stats (last N games, shifted to avoid leakage) ────────────
    qb = qb.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    for stat in ["passing_yards", "passing_tds", "attempts", "completions",
                 "passing_interceptions", "passing_epa", "qb_rush_yards"]:
        qb[f"roll_{stat}"] = (
            qb.groupby("player_id")[stat]
              .transform(lambda x: x.shift(1).rolling(ROLL_WINDOW, min_periods=2).mean())
        )

    # Completion rate rolling
    qb["roll_comp_rate"] = (qb["roll_completions"] / qb["roll_attempts"].replace(0, np.nan)).fillna(0.6)

    # Air yards per attempt — proxy for deep vs short passing scheme
    qb["air_yards_per_att"] = qb["passing_air_yards"] / qb["attempts"].replace(0, np.nan)
    qb["roll_air_yards_per_att"] = (
        qb.groupby("player_id")["air_yards_per_att"]
          .transform(lambda x: x.shift(1).rolling(ROLL_WINDOW, min_periods=2).mean())
    ).fillna(7.5)  # league average ~7.5

    # CPOE rolling — completion % over expected (skill vs scheme)
    qb["roll_cpoe"] = (
        qb.groupby("player_id")["passing_cpoe"]
          .transform(lambda x: x.shift(1).rolling(ROLL_WINDOW, min_periods=2).mean())
    ).fillna(0.0)

    # ── Rolling opponent defense (how many passing yards do they allow?) ──────
    def_pass = def_pass.sort_values(["def_team", "season", "week"]).reset_index(drop=True)
    def_pass["roll_pass_allowed"] = (
        def_pass.groupby("def_team")["opp_pass_yards_allowed"]
                .transform(lambda x: x.shift(1).rolling(ROLL_WINDOW, min_periods=2).mean())
    )
    def_pass["roll_tds_allowed"] = (
        def_pass.groupby("def_team")["opp_pass_tds_allowed"]
                .transform(lambda x: x.shift(1).rolling(ROLL_WINDOW, min_periods=2).mean())
    )

    # Keep one row per def_team/season/week
    def_agg = def_pass.groupby(["def_team", "season", "week"])[
        ["roll_pass_allowed", "roll_tds_allowed"]
    ].mean().reset_index()

    qb = qb.merge(
        def_agg.rename(columns={"def_team": "opponent_team"}),
        on=["opponent_team", "season", "week"], how="left"
    )

    # League average for missing defense data
    qb["roll_pass_allowed"] = qb["roll_pass_allowed"].fillna(qb["roll_pass_allowed"].mean())
    qb["roll_tds_allowed"]  = qb["roll_tds_allowed"].fillna(qb["roll_tds_allowed"].mean())
    qb["total_line"]        = qb["total_line"].fillna(qb["total_line"].median())
    qb["spread_line"]       = qb["spread_line"].fillna(0)

    # Implied team total from moneyline + total
    # Home implied = (total/2) - (spread/2), Away implied = (total/2) + (spread/2)
    qb["implied_team_total"] = np.where(
        qb["is_home"] == 1,
        (qb["total_line"] / 2) - (qb["spread_line"] / 2),
        (qb["total_line"] / 2) + (qb["spread_line"] / 2)
    )

    # Opponent pass defense rank within season (1=best, 32=worst)
    # Higher rank = easier matchup for QB
    season_def = (
        qb.groupby(["season", "opponent_team"])["passing_yards"]
          .mean()
          .reset_index()
          .rename(columns={"passing_yards": "avg_pass_allowed", "opponent_team": "def_team"})
    )
    season_def["def_rank"] = season_def.groupby("season")["avg_pass_allowed"].rank(ascending=True)
    qb = qb.merge(
        season_def.rename(columns={"def_team": "opponent_team"})[["season", "opponent_team", "def_rank"]],
        on=["season", "opponent_team"], how="left"
    )
    qb["def_rank"] = qb["def_rank"].fillna(16)
    qb["is_dome"]           = qb["is_dome"].fillna(0)
    qb["high_wind"]         = qb["high_wind"].fillna(0)
    qb["temp"]              = qb["temp"].fillna(60)
    qb["rest"]              = qb["rest"].fillna(7)

    return qb


# ── Feature columns ───────────────────────────────────────────────────────────
FEATURE_COLS = [
    "roll_passing_yards",       # QB's recent form
    "roll_passing_tds",
    "roll_attempts",
    "roll_comp_rate",
    "roll_cpoe",
    "roll_air_yards_per_att",
    "roll_passing_epa",
    "roll_qb_rush_yards",
    "roll_pass_allowed",        # opponent defense
    "roll_tds_allowed",
    "def_rank",                 # opponent pass defense rank this season
    "total_line",               # game environment
    "spread_line",
    "implied_team_total",       # team-specific implied scoring
    "is_home",
]


# ── Hyperparameter tuning ─────────────────────────────────────────────────────
def tune_hyperparameters(df: pd.DataFrame, target: str, n_folds: int = 5) -> dict:
    """
    Grid search over PARAM_GRID using k-fold cross-validation.
    Returns the best parameter set for the given target stat.
    """
    clean = df.dropna(subset=FEATURE_COLS + [target])
    X     = clean[FEATURE_COLS].values
    y     = clean[target].values
    kf    = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    best_params = None
    best_mae    = float("inf")

    print(f"  Tuning {target} ({len(PARAM_GRID)} configs × {n_folds} folds)...")
    for params in PARAM_GRID:
        fold_maes = []
        for train_idx, val_idx in kf.split(X):
            m = XGBRegressor(**params, random_state=42, verbosity=0)
            m.fit(X[train_idx], y[train_idx])
            preds = m.predict(X[val_idx])
            fold_maes.append(mean_absolute_error(y[val_idx], preds))
        avg_mae = np.mean(fold_maes)
        if avg_mae < best_mae:
            best_mae    = avg_mae
            best_params = params

    print(f"    Best MAE: {best_mae:.2f}  →  {best_params}")
    return best_params


# ── Walk-forward backtest ─────────────────────────────────────────────────────
def walk_forward_backtest(df: pd.DataFrame,
                          target: str = "passing_yards",
                          params: dict = None) -> pd.DataFrame:
    if params is None:
        params = {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.02,
                  "subsample": 0.8, "colsample_bytree": 0.8}
    seasons  = sorted(df["season"].unique())
    results  = []

    for i, test_season in enumerate(seasons):
        train_seasons = seasons[:i]
        if len(train_seasons) < MIN_TRAIN_SEASONS:
            continue

        train = df[df["season"].isin(train_seasons)].copy()
        test  = df[df["season"] == test_season].copy()

        # Drop rows with missing rolling features
        train = train.dropna(subset=FEATURE_COLS + [target])
        test  = test.dropna(subset=FEATURE_COLS + [target])

        if len(train) < 50 or len(test) < 5:
            continue

        X_train = train[FEATURE_COLS]
        y_train = train[target]
        X_test  = test[FEATURE_COLS]

        model = XGBRegressor(**params, random_state=42, verbosity=0)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        s = test[["player_id", "player_name", "season", "week",
                   "team", "opponent_team", target]].copy()
        s["predicted"]  = preds.round(1)
        s["error"]      = (s["predicted"] - s[target]).abs().round(1)
        results.append(s)

    return pd.concat(results, ignore_index=True)


# ── Train & save final model ──────────────────────────────────────────────────
def train_final_model(df: pd.DataFrame,
                      target: str = "passing_yards",
                      params: dict = None) -> XGBRegressor:
    if params is None:
        params = {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.02,
                  "subsample": 0.8, "colsample_bytree": 0.8}
    clean = df.dropna(subset=FEATURE_COLS + [target])
    X = clean[FEATURE_COLS]
    y = clean[target]

    model = XGBRegressor(**params, random_state=42, verbosity=0)
    model.fit(X, y)

    model_dir = Path(__file__).resolve().parent
    path = model_dir / f"props_qb_{target}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved: {path}")
    return model


def load_props_model(target: str = "passing_yards") -> XGBRegressor:
    model_dir = Path(__file__).resolve().parent
    with open(model_dir / f"props_qb_{target}.pkl", "rb") as f:
        return pickle.load(f)


# ── Predict current week ──────────────────────────────────────────────────────
def predict_current_week(models: dict) -> pd.DataFrame:
    """
    Predict prop lines for starting QBs in upcoming games.
    Uses player_stats_current for rolling stats and schedules_current for matchups.
    """
    with get_connection() as conn:
        # Current season QB stats
        qb_cur = pd.read_sql("""
            SELECT player_id, player_name, season, week, team,
                   opponent_team, game_id, attempts,
                   passing_yards, passing_tds, completions,
                   passing_interceptions, passing_epa,
                   rushing_yards AS qb_rush_yards
            FROM player_stats_current
            WHERE position = 'QB'
              AND season_type = 'REG'
              AND attempts >= ?
            ORDER BY week
        """, conn, params=(MIN_ATTEMPTS,))

        # Upcoming games
        upcoming = pd.read_sql("""
            SELECT s.game_id, s.season, s.week,
                   s.home_team, s.away_team,
                   COALESCE(s.home_rest, 7) AS home_rest,
                   COALESCE(s.away_rest, 7) AS away_rest,
                   CASE WHEN c.roof IN ('dome','closed') THEN 1 ELSE 0 END AS is_dome,
                   CASE WHEN CAST(COALESCE(c.wind,0) AS FLOAT) > 15 THEN 1 ELSE 0 END AS high_wind,
                   CAST(COALESCE(c.temp,60) AS FLOAT) AS temp,
                   o.total_line, o.spread_line
            FROM schedules_current s
            LEFT JOIN game_conditions c ON c.game_id = s.game_id
            LEFT JOIN game_odds o       ON o.game_id = s.game_id
            WHERE s.home_score IS NULL
        """, conn)

        # Current season defense rolling stats
        def_cur = pd.read_sql("""
            SELECT opponent_team AS def_team, week,
                   passing_yards AS opp_pass_yards_allowed,
                   passing_tds   AS opp_pass_tds_allowed
            FROM player_stats_current
            WHERE position = 'QB'
              AND season_type = 'REG'
              AND attempts >= ?
        """, conn, params=(MIN_ATTEMPTS,))

    if qb_cur.empty or upcoming.empty:
        print("No current week data available.")
        return pd.DataFrame()

    next_week = upcoming["week"].min()
    upcoming  = upcoming[upcoming["week"] == next_week]

    # Rolling stats for each QB
    qb_cur = qb_cur.sort_values(["player_id", "week"])
    rolling_stats = {}
    for pid, grp in qb_cur.groupby("player_id"):
        last_n = grp.tail(ROLL_WINDOW)
        rolling_stats[pid] = {
            "player_name":       grp["player_name"].iloc[-1],
            "team":              grp["team"].iloc[-1],
            "roll_passing_yards": last_n["passing_yards"].mean(),
            "roll_passing_tds":   last_n["passing_tds"].mean(),
            "roll_attempts":      last_n["attempts"].mean(),
            "roll_comp_rate":     last_n["completions"].sum() / max(last_n["attempts"].sum(), 1),
            "roll_passing_epa":   last_n["passing_epa"].mean(),
            "roll_qb_rush_yards": last_n["qb_rush_yards"].mean(),
        }

    # Rolling defense
    def_rolling = {}
    for team, grp in def_cur.groupby("def_team"):
        last_n = grp.tail(ROLL_WINDOW)
        def_rolling[team] = {
            "roll_pass_allowed": last_n["opp_pass_yards_allowed"].mean(),
            "roll_tds_allowed":  last_n["opp_pass_tds_allowed"].mean(),
        }

    # Build prediction rows
    rows = []
    for _, game in upcoming.iterrows():
        for side, team, opp in [("home", game["home_team"], game["away_team"]),
                                 ("away", game["away_team"], game["home_team"])]:
            # Find starting QB for this team
            team_qbs = {pid: s for pid, s in rolling_stats.items() if s["team"] == team}
            if not team_qbs:
                continue

            # Use QB with most recent attempts as starter
            starter_id = max(team_qbs.keys(),
                             key=lambda p: qb_cur[qb_cur["player_id"]==p]["week"].max()
                             if not qb_cur[qb_cur["player_id"]==p].empty else 0)
            stats = team_qbs[starter_id]
            def_stats = def_rolling.get(opp, {"roll_pass_allowed": 230, "roll_tds_allowed": 1.5})

            row = {
                "game_id":      game["game_id"],
                "week":         next_week,
                "player_id":    starter_id,
                "player_name":  stats["player_name"],
                "team":         team,
                "opponent":     opp,
                "is_home":      1 if side == "home" else 0,
                "rest":         game["home_rest"] if side == "home" else game["away_rest"],
                "is_dome":      game["is_dome"],
                "high_wind":    game["high_wind"],
                "temp":         game["temp"],
                "total_line":   game["total_line"] or 45.0,
                "spread_line":  game["spread_line"] or 0.0,
                **{k: v for k, v in stats.items() if k not in ["player_name", "team"]},
                **def_stats,
            }
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    pred_df = pd.DataFrame(rows)
    pred_df["total_line"]  = pred_df["total_line"].fillna(45.0)
    pred_df["spread_line"] = pred_df["spread_line"].fillna(0.0)

    for stat, model in models.items():
        feat = pred_df[FEATURE_COLS].fillna(0)
        pred_df[f"pred_{stat}"] = model.predict(feat).round(1)

    return pred_df


# ── Reporting ─────────────────────────────────────────────────────────────────
def print_accuracy_report(results: pd.DataFrame, target: str) -> None:
    seasons = sorted(results["season"].unique())

    # Use appropriate thresholds based on target
    is_counting_stat = target in ["passing_tds", "rushing_tds", "receiving_tds",
                                   "passing_interceptions"]
    if is_counting_stat:
        t1, t2, l1, l2 = 0.5, 1.0, "Exact±0.5", "Within 1"
    else:
        t1, t2, l1, l2 = 25, 50, "Within 25", "Within 50"

    print(f"\n{'='*60}")
    print(f"QB {target} prediction accuracy")
    print(f"{'Season':<8} {'Games':>6} {'MAE':>8} {l1:>12} {l2:>12}")
    print(f"{'='*60}")
    total_games = total_mae = w1 = w2 = 0
    for season in seasons:
        s    = results[results["season"] == season]
        mae  = s["error"].mean()
        p1   = (s["error"] <= t1).mean() * 100
        p2   = (s["error"] <= t2).mean() * 100
        total_games += len(s)
        total_mae   += s["error"].sum()
        w1          += (s["error"] <= t1).sum()
        w2          += (s["error"] <= t2).sum()
        print(f"{season:<8} {len(s):>6} {mae:>8.2f} {p1:>11.1f}% {p2:>11.1f}%")
    print(f"{'='*60}")
    print(f"{'All-time':<8} {total_games:>6} {total_mae/total_games:>8.2f} "
          f"{w1/total_games*100:>11.1f}% {w2/total_games*100:>11.1f}%")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading QB data...")
    df = load_qb_data()
    print(f"Loaded {len(df)} QB game logs ({df['season'].min()}–{df['season'].max()})")

    print("\nNaN counts per feature:")
    print(df[FEATURE_COLS].isna().sum())

    print("\nTuning hyperparameters and running backtests...")
    best_params = {}
    models = {}
    for target in ["passing_yards", "passing_tds", "attempts"]:
        bp = tune_hyperparameters(df, target=target)
        best_params[target] = bp
        results = walk_forward_backtest(df, target=target, params=bp)
        print_accuracy_report(results, target)
        print(f"  Training final {target} model...")
        models[target] = train_final_model(df, target=target, params=bp)

    print("\nBest params summary:")
    for t, p in best_params.items():
        print(f"  {t}: {p}")

    print("\nPredicting current week...")
    preds = predict_current_week(models)
    if not preds.empty:
        print(preds[["player_name", "team", "opponent",
                      "pred_passing_yards", "pred_passing_tds", "pred_attempts"]].to_string(index=False))
    else:
        print("No upcoming games found — run weekly_update.py first.")

    print("\nSaving QB prop predictions to DB...")
    if not preds.empty:
        save(preds, "qb_prop_predictions")
    print("Done.")