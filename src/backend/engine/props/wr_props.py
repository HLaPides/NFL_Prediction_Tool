"""
wr_props.py — XGBoost prop predictions for WR receiving stats.

Predicts four stats per WR per game: receiving_yards, receiving_tds,
receptions, targets. The top 3 WRs per team (by recent target volume)
are predicted — unlike RBs where only the lead back is used. Used by
predictor.py to populate the wr_prop_predictions table each week.

Architecture:
    XGBoost regressor, one model per stat target. Walk-forward validated.
    Uses WR-specific defense rating (roll_adj_def_wr_epa) which measures
    EPA allowed specifically on WR targets — more precise than overall
    pass defense, which is diluted by TE and RB routes.

Features (18):
    Rolling receiving: receiving_yards, receiving_tds, receptions, targets,
    target_share, air_yards_share, ypr, catch_rate, receiving_epa, racr, wopr
    Opponent defense: wr_def_rank, def_games_played
    Game context: total_line, spread_line, implied_team_total, is_home, rest

Note on RACR and WOPR:
    RACR (Receiver Air Conversion Ratio) and WOPR (Weighted Opportunity Rating)
    are advanced target quality metrics. They're in FEATURE_COLS because they
    improve backtest accuracy. They're available in player_stats_current so
    no default substitution needed.

Usage:
    python3.11 engine/props/wr_props.py   # backtest + retrain all targets

Writes to disk:
    engine/props/props_wr_{target}.pkl for each target in STAT_TARGETS
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
MIN_TARGETS       = 3         # Filter out gadget plays / single-target appearances
ROLL_WINDOW       = 4
MIN_TRAIN_SEASONS = 4
STAT_TARGETS      = ["receiving_yards", "receiving_tds", "receptions", "targets"]

BEST_PARAMS = {
    "receiving_yards": {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
    "receiving_tds":   {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
    "receptions":      {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
    "targets":         {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
}


# ── Data loading ──────────────────────────────────────────────────────────────
def load_wr_data() -> pd.DataFrame:
    """
    Load and assemble the historical WR training dataset.

    Joins player_stats (WR rows with ≥ MIN_TARGETS targets) to schedules,
    game odds, and opponent WR-specific defense ratings. Computes 4-game
    rolling averages using shift(1) to prevent leakage.

    Derived features:
        roll_ypr:        rolling receiving_yards / receptions (yards per reception)
        roll_catch_rate: rolling receptions / targets
        implied_team_total: derived from total_line and spread_line

    Opponent defense features:
        roll_adj_def_wr_epa — EPA allowed on WR-targeted pass plays,
        opponent-adjusted. More precise than overall pass defense rank
        because it isolates coverage quality against boundary receivers.

    Returns:
        pd.DataFrame: One row per WR game appearance with all features
                      and target columns populated.
    """
    with get_connection() as conn:
        wr = pd.read_sql("""
            SELECT
                p.player_id, p.player_name, p.season, p.week,
                p.team, p.opponent_team, p.game_id,
                p.receiving_yards, p.receiving_tds, p.receptions,
                p.targets, p.target_share, p.air_yards_share,
                p.receiving_air_yards, p.receiving_yards_after_catch,
                p.receiving_epa, p.racr, p.wopr
            FROM player_stats p
            WHERE p.position = 'WR'
              AND p.season_type = 'REG'
              AND p.targets >= ?
              AND p.season >= ?
            ORDER BY p.season, p.week
        """, conn, params=(MIN_TARGETS, MIN_SEASON))

        sched = pd.read_sql("""
            SELECT s.game_id, s.season, s.week,
                   s.home_team, s.away_team,
                   COALESCE(s.home_rest, 7) AS home_rest,
                   COALESCE(s.away_rest, 7) AS away_rest,
                   o.total_line, o.spread_line
            FROM schedules s
            LEFT JOIN game_odds o ON o.game_id = s.game_id
            WHERE s.game_type = 'REG'
        """, conn)

        def_ratings = pd.read_sql("""
            SELECT team, season, week,
                   roll_adj_def_pass_epa,
                   roll_adj_def_wr_epa,
                   pass_def_rank,
                   wr_def_rank,
                   def_games_played
            FROM defense_ratings
        """, conn)

    wr = wr.merge(sched, on=["game_id", "season", "week"], how="left")
    wr["is_home"] = (wr["team"] == wr["home_team"]).astype(int)
    wr["rest"]    = np.where(wr["is_home"] == 1, wr["home_rest"], wr["away_rest"])

    wr = wr.merge(
        def_ratings.rename(columns={"team": "opponent_team"}),
        on=["opponent_team", "season", "week"], how="left"
    )
    for col in ["roll_adj_def_pass_epa", "roll_adj_def_wr_epa"]:
        wr[col] = wr[col].fillna(0.0)
    for col in ["pass_def_rank", "wr_def_rank"]:
        wr[col] = wr[col].fillna(16)
    wr["def_games_played"] = wr["def_games_played"].fillna(0)

    wr["total_line"]  = wr["total_line"].fillna(wr["total_line"].median())
    wr["spread_line"] = wr["spread_line"].fillna(0)
    wr["implied_team_total"] = np.where(
        wr["is_home"] == 1,
        (wr["total_line"] / 2) - (wr["spread_line"] / 2),
        (wr["total_line"] / 2) + (wr["spread_line"] / 2)
    )

    wr = wr.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    for stat in ["receiving_yards", "receiving_tds", "receptions", "targets",
                 "target_share", "air_yards_share", "receiving_air_yards",
                 "receiving_yards_after_catch", "receiving_epa", "racr", "wopr"]:
        if stat in wr.columns:
            wr[f"roll_{stat}"] = (
                wr.groupby("player_id")[stat]
                  .transform(lambda x: x.shift(1).rolling(ROLL_WINDOW, min_periods=2).mean())
            )

    wr["ypr"] = wr["receiving_yards"] / wr["receptions"].replace(0, np.nan)
    wr["roll_ypr"] = (
        wr.groupby("player_id")["ypr"]
          .transform(lambda x: x.shift(1).rolling(ROLL_WINDOW, min_periods=2).mean())
    ).fillna(12.0)

    wr["catch_rate"] = wr["receptions"] / wr["targets"].replace(0, np.nan)
    wr["roll_catch_rate"] = (
        wr.groupby("player_id")["catch_rate"]
          .transform(lambda x: x.shift(1).rolling(ROLL_WINDOW, min_periods=2).mean())
    ).fillna(0.65)

    for col in ["roll_target_share", "roll_air_yards_share", "roll_racr", "roll_wopr"]:
        if col in wr.columns:
            wr[col] = wr[col].fillna(0)

    wr["rest"] = wr["rest"].fillna(7)
    return wr


# ── Feature columns ───────────────────────────────────────────────────────────
FEATURE_COLS = [
    "roll_receiving_yards",
    "roll_receiving_tds",
    "roll_receptions",
    "roll_targets",
    "roll_target_share",
    "roll_air_yards_share",
    "roll_ypr",
    "roll_catch_rate",
    "roll_receiving_epa",
    "roll_racr",
    "roll_wopr",
    "wr_def_rank",
    "def_games_played",
    "total_line",
    "spread_line",
    "implied_team_total",
    "is_home",
    "rest",
]


# ── Walk-forward backtest ─────────────────────────────────────────────────────
def walk_forward_backtest(df: pd.DataFrame,
                          target: str = "receiving_yards",
                          params: dict = None) -> pd.DataFrame:
    """
    Validate model accuracy using strict walk-forward cross-validation.

    Trains on seasons 1..N-1, tests on season N. Requires MIN_TRAIN_SEASONS
    before the first test season. Rows with null features or target are dropped.

    Args:
        df: Output of load_wr_data().
        target: Stat column to predict (must be in STAT_TARGETS).
        params: XGBoost params dict. Defaults to BEST_PARAMS[target].

    Returns:
        pd.DataFrame: Predictions with player_id, player_name, season, week,
                      team, opponent_team, target, predicted, error columns.
    """
    if params is None:
        params = BEST_PARAMS.get(target, BEST_PARAMS["receiving_yards"])
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

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


# ── Train & save ──────────────────────────────────────────────────────────────
def train_final_model(df: pd.DataFrame,
                      target: str = "receiving_yards",
                      params: dict = None) -> XGBRegressor:
    """
    Train on all available data and save model to disk.

    Args:
        df: Output of load_wr_data().
        target: Stat to predict.
        params: XGBoost params. Defaults to BEST_PARAMS[target].

    Returns:
        XGBRegressor: Fitted model (also saved to props_wr_{target}.pkl).
    """
    if params is None:
        params = BEST_PARAMS.get(target, BEST_PARAMS["receiving_yards"])
    clean = df.dropna(subset=FEATURE_COLS + [target])
    model = XGBRegressor(**params, random_state=42, verbosity=0)
    model.fit(clean[FEATURE_COLS], clean[target])
    model_dir = Path(__file__).resolve().parent
    path = model_dir / f"props_wr_{target}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Saved: {path.name}")
    return model


def load_props_model(target: str = "receiving_yards") -> XGBRegressor:
    """Load a trained WR prop model from disk."""
    model_dir = Path(__file__).resolve().parent
    with open(model_dir / f"props_wr_{target}.pkl", "rb") as f:
        return pickle.load(f)


# ── Predict current week ──────────────────────────────────────────────────────
def predict_current_week(models: dict) -> pd.DataFrame:
    """
    Generate WR prop predictions for this week's upcoming games.

    Selects the top 3 WRs per team by recent target volume (last ROLL_WINDOW
    games in player_stats_current). Unlike RBs where only the lead back is
    predicted, WR passing games are distributed enough that predicting 3
    receivers per team is meaningful for prop betting.

    Args:
        models: Dict of { target_name: fitted XGBRegressor }.
                Load via { t: load_props_model(t) for t in STAT_TARGETS }.

    Returns:
        pd.DataFrame: Up to 3 rows per team matchup with pred_{stat} columns.
                      Empty DataFrame if no upcoming games or player data found.
    """
    with get_connection() as conn:
        wr_cur = pd.read_sql("""
            SELECT player_id, player_name, season, week, team,
                   opponent_team, game_id, targets,
                   receiving_yards, receiving_tds, receptions,
                   target_share, air_yards_share, receiving_air_yards,
                   receiving_yards_after_catch, receiving_epa, racr, wopr
            FROM player_stats_current
            WHERE position = 'WR'
              AND season_type = 'REG'
              AND targets >= ?
            ORDER BY week
        """, conn, params=(MIN_TARGETS,))

        upcoming = pd.read_sql("""
            SELECT s.game_id, s.season, s.week,
                   s.home_team, s.away_team,
                   COALESCE(s.home_rest, 7) AS home_rest,
                   COALESCE(s.away_rest, 7) AS away_rest,
                   o.total_line, o.spread_line
            FROM schedules_current s
            LEFT JOIN game_odds o ON o.game_id = s.game_id
            WHERE s.home_score IS NULL
        """, conn)

    if wr_cur.empty or upcoming.empty:
        print("No current week data available.")
        return pd.DataFrame()

    def_map   = get_current_defense_ratings().set_index("team").to_dict("index")
    next_week = upcoming["week"].min()
    upcoming  = upcoming[upcoming["week"] == next_week]

    wr_cur = wr_cur.sort_values(["player_id", "week"])
    team_wrs = {}
    for pid, grp in wr_cur.groupby("player_id"):
        last_n        = grp.tail(ROLL_WINDOW)
        team          = grp["team"].iloc[-1]
        total_targets = last_n["targets"].sum()
        if team not in team_wrs:
            team_wrs[team] = []
        team_wrs[team].append({
            "player_id":                 pid,
            "player_name":               grp["player_name"].iloc[-1],
            "team":                      team,
            "total_targets":             total_targets,
            "roll_receiving_yards":      last_n["receiving_yards"].mean(),
            "roll_receiving_tds":        last_n["receiving_tds"].mean(),
            "roll_receptions":           last_n["receptions"].mean(),
            "roll_targets":              last_n["targets"].mean(),
            "roll_target_share":         last_n["target_share"].mean() if "target_share" in last_n else 0,
            "roll_air_yards_share":      last_n["air_yards_share"].mean() if "air_yards_share" in last_n else 0,
            "roll_ypr":                  last_n["receiving_yards"].sum() / max(last_n["receptions"].sum(), 1),
            "roll_catch_rate":           last_n["receptions"].sum() / max(last_n["targets"].sum(), 1),
            "roll_receiving_epa":        last_n["receiving_epa"].mean(),
            "roll_racr":                 last_n["racr"].mean() if "racr" in last_n else 0,
            "roll_wopr":                 last_n["wopr"].mean() if "wopr" in last_n else 0,
        })

    # Top 3 per team by target volume
    for team in team_wrs:
        team_wrs[team] = sorted(team_wrs[team], key=lambda x: x["total_targets"], reverse=True)[:3]

    rows = []
    for _, game in upcoming.iterrows():
        for side, team, opp in [("home", game["home_team"], game["away_team"]),
                                 ("away", game["away_team"], game["home_team"])]:
            if team not in team_wrs:
                continue
            opp_def     = def_map.get(opp, {})
            total_line  = game["total_line"] or 45.0
            spread_line = game["spread_line"] or 0.0
            implied     = (total_line / 2) - (spread_line / 2) if side == "home" \
                          else (total_line / 2) + (spread_line / 2)

            for wr in team_wrs[team]:
                rows.append({
                    "game_id":               game["game_id"],
                    "week":                  next_week,
                    "player_id":             wr["player_id"],
                    "player_name":           wr["player_name"],
                    "team":                  team,
                    "opponent":              opp,
                    "is_home":               1 if side == "home" else 0,
                    "rest":                  game["home_rest"] if side == "home" else game["away_rest"],
                    "total_line":            total_line,
                    "spread_line":           spread_line,
                    "implied_team_total":    implied,
                    "roll_adj_def_pass_epa": opp_def.get("roll_adj_def_pass_epa", 0.0),
                    "roll_adj_def_wr_epa":   opp_def.get("roll_adj_def_wr_epa",   0.0),
                    "pass_def_rank":         opp_def.get("pass_def_rank",           16),
                    "wr_def_rank":           opp_def.get("wr_def_rank",             16),
                    "def_games_played":      opp_def.get("def_games_played",         0),
                    **{k: v for k, v in wr.items()
                       if k not in ["player_id", "player_name", "team", "total_targets"]},
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
    print("Loading WR data...")
    df = load_wr_data()
    print(f"  {len(df):,} WR game rows loaded")

    print("\nWalk-forward backtest (receiving_yards)...")
    bt = walk_forward_backtest(df, target="receiving_yards")
    mae = bt["error"].mean()
    within_15 = (bt["error"] <= 15).mean() * 100
    within_30 = (bt["error"] <= 30).mean() * 100
    print(f"  MAE: {mae:.2f}  |  within 15: {within_15:.1f}%  |  within 30: {within_30:.1f}%")

    print("\nPer-season MAE:")
    print(bt.groupby("season")["error"].mean().round(2).to_string())

    print("\nTraining final models...")
    models = {}
    for target in STAT_TARGETS:
        print(f"  {target}...")
        models[target] = train_final_model(df, target=target)

    print("\nDone. Models saved to engine/props/")