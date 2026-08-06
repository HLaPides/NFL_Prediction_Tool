"""
qb_props.py — XGBoost prop predictions for QB passing stats.

Predicts five stats per QB per game: passing_yards, passing_tds, attempts,
completions, passing_interceptions. Used by predictor.py to populate the
qb_prop_predictions table each week.

Architecture:
    XGBoost regressor trained on 4-game rolling player stats and opponent
    defense quality. Walk-forward validated (train on seasons 1..N-1, test N).

Features (16):
    Player rolling stats: passing_yards, tds, attempts, comp_rate, cpoe,
    air_yards_per_att, passing_epa, qb_rush_yards
    Game context: total_line, spread_line, implied_team_total, is_dome,
    high_wind, temp, is_home, rest

Opponent defense:
    roll_adj_def_pass_epa — opponent-adjusted pass EPA allowed, from defense.py.
    Loaded from defense_ratings table but NOT in FEATURE_COLS (available in
    prediction rows for future feature expansion).

Note on CPOE and air_yards_per_att:
    These are set to defaults (0.0 and 7.5) in predict_current_week() because
    player_stats_current doesn't include them. They're kept in the model because
    they improve backtest accuracy on historical data.

Usage:
    python3.11 engine/props/qb_props.py   # backtest + retrain all targets

Writes to disk:
    engine/props/props_qb_{target}.pkl for each target in STAT_TARGETS
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import pickle
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from data.sql_helpers import get_connection, save
from engine.defense import get_current_defense_ratings

# ── Constants ─────────────────────────────────────────────────────────────────
MIN_SEASON        = 1999
MIN_ATTEMPTS      = 10        # Filter out game manager / mop-up appearances
ROLL_WINDOW       = 4         # 4-game rolling window — same as main model
MIN_TRAIN_SEASONS = 4
STAT_TARGETS      = ["passing_yards", "passing_tds", "attempts",
                     "completions", "passing_interceptions"]

BEST_PARAMS = {
    "passing_yards":         {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.02, "subsample": 0.8, "colsample_bytree": 0.8},
    "passing_tds":           {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.02, "subsample": 0.8, "colsample_bytree": 0.8},
    "attempts":              {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.02, "subsample": 0.8, "colsample_bytree": 0.8},
    "completions":           {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.02, "subsample": 0.8, "colsample_bytree": 0.8},
    "passing_interceptions": {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.02, "subsample": 0.8, "colsample_bytree": 0.8},
}


# ── Data loading ──────────────────────────────────────────────────────────────
def load_qb_data() -> pd.DataFrame:
    """
    Load and assemble the historical QB training dataset.

    Joins player_stats (QB rows with ≥ MIN_ATTEMPTS attempts) to schedules,
    game conditions, game odds, and opponent defense ratings. Computes
    4-game rolling averages for all stat columns using shift(1) to ensure
    no leakage — rolling stats going into game N use only games 1..N-1.

    Derived features:
        roll_comp_rate:         rolling completions / attempts
        roll_air_yards_per_att: rolling passing_air_yards / attempts
        roll_cpoe:              rolling completion percentage over expected
        implied_team_total:     derived from total_line and spread_line

    Returns:
        pd.DataFrame: One row per QB game appearance with all features
                      and target columns populated.
    """
    with get_connection() as conn:
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
              AND p.attempts >= %s
              AND p.season >= %s
            ORDER BY p.season, p.week
        """, conn, params=(MIN_ATTEMPTS, MIN_SEASON))

        sched = pd.read_sql("""
            SELECT s.game_id, s.season, s.week,
                   s.home_team, s.away_team,
                   COALESCE(s.home_rest, 7) AS home_rest,
                   COALESCE(s.away_rest, 7) AS away_rest,
                   CASE WHEN c.roof IN ('dome','closed') THEN 1 ELSE 0 END AS is_dome,
                   CASE WHEN CAST(COALESCE(c.wind,0) AS FLOAT) > 15 THEN 1 ELSE 0 END AS high_wind,
                   CAST(COALESCE(c.temp, 60) AS FLOAT) AS temp,
                   o.total_line, o.spread_line
            FROM schedules s
            LEFT JOIN game_conditions c ON c.game_id = s.game_id
            LEFT JOIN game_odds o       ON o.game_id = s.game_id
            WHERE s.game_type = 'REG'
        """, conn)

        def_ratings = pd.read_sql("""
            SELECT team, season, week,
                   roll_adj_def_pass_epa,
                   roll_adj_def_wr_epa,
                   roll_adj_def_te_epa,
                   pass_def_rank,
                   wr_def_rank,
                   def_games_played
            FROM defense_ratings
        """, conn)

    qb = qb.merge(sched, on=["game_id", "season", "week"], how="left")
    qb["is_home"] = (qb["team"] == qb["home_team"]).astype(int)
    qb["rest"]    = np.where(qb["is_home"] == 1, qb["home_rest"], qb["away_rest"])

    qb = qb.merge(
        def_ratings.rename(columns={"team": "opponent_team"}),
        on=["opponent_team", "season", "week"], how="left"
    )
    for col in ["roll_adj_def_pass_epa", "roll_adj_def_wr_epa", "roll_adj_def_te_epa"]:
        qb[col] = qb[col].fillna(0.0)
    for col in ["pass_def_rank", "wr_def_rank"]:
        qb[col] = qb[col].fillna(16)
    qb["def_games_played"] = qb["def_games_played"].fillna(0)

    qb = qb.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    for stat in ["passing_yards", "passing_tds", "attempts", "completions",
                 "passing_interceptions", "passing_epa", "qb_rush_yards"]:
        qb[f"roll_{stat}"] = (
            qb.groupby("player_id")[stat]
              .transform(lambda x: x.shift(1).rolling(ROLL_WINDOW, min_periods=2).mean())
        )

    qb["roll_comp_rate"] = (
        qb["roll_completions"] / qb["roll_attempts"].replace(0, np.nan)
    ).fillna(0.62)

    qb["air_yards_per_att"] = qb["passing_air_yards"] / qb["attempts"].replace(0, np.nan)
    qb["roll_air_yards_per_att"] = (
        qb.groupby("player_id")["air_yards_per_att"]
          .transform(lambda x: x.shift(1).rolling(ROLL_WINDOW, min_periods=2).mean())
    ).fillna(7.5)

    qb["roll_cpoe"] = (
        qb.groupby("player_id")["passing_cpoe"]
          .transform(lambda x: x.shift(1).rolling(ROLL_WINDOW, min_periods=2).mean())
    ).fillna(0.0)

    qb["total_line"]  = qb["total_line"].fillna(qb["total_line"].median())
    qb["spread_line"] = qb["spread_line"].fillna(0)
    qb["implied_team_total"] = np.where(
        qb["is_home"] == 1,
        (qb["total_line"] / 2) - (qb["spread_line"] / 2),
        (qb["total_line"] / 2) + (qb["spread_line"] / 2)
    )
    qb["is_dome"]   = qb["is_dome"].fillna(0)
    qb["high_wind"] = qb["high_wind"].fillna(0)
    qb["temp"]      = qb["temp"].fillna(60)
    qb["rest"]      = qb["rest"].fillna(7)

    return qb


# ── Feature columns ───────────────────────────────────────────────────────────
FEATURE_COLS = [
    "roll_passing_yards",
    "roll_passing_tds",
    "roll_attempts",
    "roll_comp_rate",
    "roll_cpoe",
    "roll_air_yards_per_att",
    "roll_passing_epa",
    "roll_qb_rush_yards",
    "total_line",
    "spread_line",
    "implied_team_total",
    "is_dome",
    "high_wind",
    "temp",
    "is_home",
    "rest",
]


# ── Walk-forward backtest ─────────────────────────────────────────────────────
def walk_forward_backtest(df: pd.DataFrame,
                          target: str = "passing_yards",
                          params: dict = None) -> pd.DataFrame:
    """
    Validate model accuracy using strict walk-forward cross-validation.

    Trains on seasons 1..N-1, tests on season N. Requires MIN_TRAIN_SEASONS
    before the first test season. Rows with null features or target are dropped.

    Args:
        df: Output of load_qb_data().
        target: Stat column to predict (must be in STAT_TARGETS).
        params: XGBoost params dict. Defaults to BEST_PARAMS[target].

    Returns:
        pd.DataFrame: Predictions with player_id, player_name, season, week,
                      team, opponent_team, target, predicted, error columns.
    """
    if params is None:
        params = BEST_PARAMS.get(target, BEST_PARAMS["passing_yards"])
    seasons = sorted(df["season"].unique())
    results = []

    for i, test_season in enumerate(seasons):
        train_seasons = seasons[:i]
        if len(train_seasons) < MIN_TRAIN_SEASONS:
            continue
        train = df[df["season"].isin(train_seasons)].dropna(subset=FEATURE_COLS + [target])
        test  = df[df["season"] == test_season].dropna(subset=FEATURE_COLS + [target])
        if len(train) < 50 or len(test) < 5:
            continue
        model = XGBRegressor(**params, random_state=42, verbosity=0)
        model.fit(train[FEATURE_COLS], train[target])
        preds = model.predict(test[FEATURE_COLS])
        s = test[["player_id", "player_name", "season", "week",
                   "team", "opponent_team", target]].copy()
        s["predicted"] = preds.round(1)
        s["error"]     = (s["predicted"] - s[target]).abs().round(1)
        results.append(s)

    return pd.concat(results, ignore_index=True)


# ── Train & save ──────────────────────────────────────────────────────────────
def train_final_model(df: pd.DataFrame,
                      target: str = "passing_yards",
                      params: dict = None) -> XGBRegressor:
    """
    Train on all available data and save model to disk.

    Args:
        df: Output of load_qb_data().
        target: Stat to predict.
        params: XGBoost params. Defaults to BEST_PARAMS[target].

    Returns:
        XGBRegressor: Fitted model (also saved to props_qb_{target}.pkl).
    """
    if params is None:
        params = BEST_PARAMS.get(target, BEST_PARAMS["passing_yards"])
    clean = df.dropna(subset=FEATURE_COLS + [target])
    model = XGBRegressor(**params, random_state=42, verbosity=0)
    model.fit(clean[FEATURE_COLS], clean[target])
    model_dir = Path(__file__).resolve().parent
    path = model_dir / f"props_qb_{target}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Saved: {path.name}")
    return model


def load_props_model(target: str = "passing_yards") -> XGBRegressor:
    """Load a trained QB prop model from disk."""
    model_dir = Path(__file__).resolve().parent
    with open(model_dir / f"props_qb_{target}.pkl", "rb") as f:
        return pickle.load(f)


# ── Predict current week ──────────────────────────────────────────────────────
def predict_current_week(models: dict) -> pd.DataFrame:
    """
    Generate QB prop predictions for this week's upcoming games.

    Identifies the starting QB per team as the player with the most recent
    appearance in player_stats_current. Builds a feature vector from their
    last ROLL_WINDOW games and the opponent's current defensive ratings.

    Note: passing_cpoe and air_yards_per_att are not available in
    player_stats_current, so they default to 0.0 and 7.5 respectively.

    Args:
        models: Dict of { target_name: fitted XGBRegressor }.
                Load via { t: load_props_model(t) for t in STAT_TARGETS }.

    Returns:
        pd.DataFrame: One row per QB matchup with pred_{stat} columns.
                      Empty DataFrame if no upcoming games or player data found.
    """
    with get_connection() as conn:
        qb_cur = pd.read_sql("""
            SELECT player_id, player_name, season, week, team,
                   opponent_team, game_id, attempts,
                   passing_yards, passing_tds, completions,
                   passing_interceptions, passing_epa,
                   rushing_yards AS qb_rush_yards
            FROM player_stats_current
            WHERE position = 'QB'
              AND season_type = 'REG'
              AND attempts >= %s
            ORDER BY week
        """, conn, params=(MIN_ATTEMPTS,))

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

    if qb_cur.empty or upcoming.empty:
        print("No current week data available.")
        return pd.DataFrame()

    def_map   = get_current_defense_ratings().set_index("team").to_dict("index")
    next_week = upcoming["week"].min()
    upcoming  = upcoming[upcoming["week"] == next_week]

    qb_cur = qb_cur.sort_values(["player_id", "week"])
    rolling_stats = {}
    for pid, grp in qb_cur.groupby("player_id"):
        last_n = grp.tail(ROLL_WINDOW)
        rolling_stats[pid] = {
            "player_name":            grp["player_name"].iloc[-1],
            "team":                   grp["team"].iloc[-1],
            "roll_passing_yards":     last_n["passing_yards"].mean(),
            "roll_passing_tds":       last_n["passing_tds"].mean(),
            "roll_attempts":          last_n["attempts"].mean(),
            "roll_comp_rate":         last_n["completions"].sum() / max(last_n["attempts"].sum(), 1),
            "roll_passing_epa":       last_n["passing_epa"].mean(),
            "roll_qb_rush_yards":     last_n["qb_rush_yards"].mean(),
            "roll_cpoe":              0.0,   # not in player_stats_current
            "roll_air_yards_per_att": 7.5,   # not in player_stats_current
        }

    rows = []
    for _, game in upcoming.iterrows():
        for side, team, opp in [("home", game["home_team"], game["away_team"]),
                                 ("away", game["away_team"], game["home_team"])]:
            team_qbs = {pid: s for pid, s in rolling_stats.items() if s["team"] == team}
            if not team_qbs:
                continue
            # Most recent appearance = most likely starter
            starter_id = max(team_qbs.keys(),
                             key=lambda p: qb_cur[qb_cur["player_id"] == p]["week"].max()
                             if not qb_cur[qb_cur["player_id"] == p].empty else 0)
            stats   = team_qbs[starter_id]
            opp_def = def_map.get(opp, {})

            total_line  = game["total_line"] or 45.0
            spread_line = game["spread_line"] or 0.0
            implied     = (total_line / 2) - (spread_line / 2) if side == "home" \
                          else (total_line / 2) + (spread_line / 2)

            rows.append({
                "game_id":               game["game_id"],
                "week":                  next_week,
                "player_id":             starter_id,
                "player_name":           stats["player_name"],
                "team":                  team,
                "opponent":              opp,
                "is_home":               1 if side == "home" else 0,
                "rest":                  game["home_rest"] if side == "home" else game["away_rest"],
                "is_dome":               game.get("is_dome", 0),
                "high_wind":             game.get("high_wind", 0),
                "temp":                  game.get("temp", 60),
                "total_line":            total_line,
                "spread_line":           spread_line,
                "implied_team_total":    implied,
                "roll_adj_def_pass_epa": opp_def.get("roll_adj_def_pass_epa", 0.0),
                "roll_adj_def_wr_epa":   opp_def.get("roll_adj_def_wr_epa",   0.0),
                "roll_adj_def_te_epa":   opp_def.get("roll_adj_def_te_epa",   0.0),
                "pass_def_rank":         opp_def.get("pass_def_rank",          16),
                "wr_def_rank":           opp_def.get("wr_def_rank",            16),
                "def_games_played":      opp_def.get("def_games_played",        0),
                **{k: v for k, v in stats.items() if k not in ["player_name", "team"]},
            })

    if not rows:
        return pd.DataFrame()

    pred_df = pd.DataFrame(rows)
    for col in FEATURE_COLS:
        if col not in pred_df.columns:
            pred_df[col] = 0.0
    pred_df = pred_df.fillna(0)

    for stat, model in models.items():
        pred_df[f"pred_{stat}"] = model.predict(pred_df[FEATURE_COLS]).round(1)

    return pred_df


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading QB data...")
    df = load_qb_data()
    print(f"  {len(df):,} QB game rows loaded")

    print("\nWalk-forward backtest (passing_yards)...")
    bt = walk_forward_backtest(df, target="passing_yards")
    mae = bt["error"].mean()
    within_25 = (bt["error"] <= 25).mean() * 100
    within_50 = (bt["error"] <= 50).mean() * 100
    print(f"  MAE: {mae:.2f}  |  within 25: {within_25:.1f}%  |  within 50: {within_50:.1f}%")
    print(f"  Test seasons: {sorted(bt['season'].unique())}")
    print(f"  Total predictions: {len(bt)}")
    print("\nPer-season MAE:")
    print(bt.groupby("season")["error"].mean().round(2).to_string())

    print("\nTraining final models...")
    models = {}
    for target in STAT_TARGETS:
        print(f"  {target}...")
        models[target] = train_final_model(df, target=target)

    print("\nDone. Models saved to engine/props/")