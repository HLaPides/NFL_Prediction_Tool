"""
rb_props.py — XGBoost prop predictions for RB rushing and receiving stats.

Predicts five stats per RB per game: rushing_yards, rushing_tds, carries,
receptions, receiving_yards. Only the lead back per team (most carries over
the last ROLL_WINDOW games) is predicted. Used by predictor.py to populate
the rb_prop_predictions table each week.

Architecture:
    XGBoost regressor, one model per stat target. Walk-forward validated.
    Separate defense features for rushing (roll_adj_def_rush_epa) and
    receiving (roll_adj_def_rb_epa) — because a strong run defense doesn't
    necessarily mean strong coverage against RBs out of the backfield.

Features (20):
    Rolling rushing: rushing_yards, rushing_tds, carries, ypc, rushing_epa
    Rolling receiving: receptions, targets, receiving_yards, rec_rate, receiving_epa
    Opponent defense: roll_adj_def_rush_epa, roll_adj_def_rb_epa, rush_def_rank,
                      rb_def_rank, def_games_played
    Game context: total_line, spread_line, implied_team_total, is_home, rest

Usage:
    python3.11 engine/props/rb_props.py   # backtest + retrain all targets

Writes to disk:
    engine/props/props_rb_{target}.pkl for each target in STAT_TARGETS
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
MIN_CARRIES       = 5         # Filter out change-of-pace / emergency appearances
ROLL_WINDOW       = 4
MIN_TRAIN_SEASONS = 4
STAT_TARGETS      = ["rushing_yards", "rushing_tds", "carries",
                     "receptions", "receiving_yards"]

BEST_PARAMS = {
    "rushing_yards":   {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
    "rushing_tds":     {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
    "carries":         {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
    "receptions":      {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.02, "subsample": 0.7, "colsample_bytree": 0.7},
    "receiving_yards": {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.02, "subsample": 0.8, "colsample_bytree": 0.8},
}


# ── Data loading ──────────────────────────────────────────────────────────────
def load_rb_data() -> pd.DataFrame:
    """
    Load and assemble the historical RB training dataset.

    Joins player_stats (RB rows with ≥ MIN_CARRIES carries) to schedules,
    game odds, and opponent defense ratings. Computes 4-game rolling averages
    using shift(1) to prevent leakage.

    Derived features:
        roll_ypc:      rolling yards per carry
        roll_rec_rate: rolling receptions / targets (catch rate)
        implied_team_total: derived from total_line and spread_line

    Opponent defense features:
        roll_adj_def_rush_epa — for rushing stat targets
        roll_adj_def_rb_epa   — for receiving stat targets (RB targets allowed)

    Returns:
        pd.DataFrame: One row per RB game appearance with all features
                      and target columns populated.
    """
    with get_connection() as conn:
        rb = pd.read_sql("""
            SELECT
                p.player_id, p.player_name, p.season, p.week,
                p.team, p.opponent_team, p.game_id,
                p.rushing_yards, p.rushing_tds, p.carries,
                p.receptions, p.targets, p.receiving_yards,
                p.receiving_tds, p.rushing_epa, p.receiving_epa,
                p.rushing_fumbles
            FROM player_stats p
            WHERE p.position = 'RB'
              AND p.season_type = 'REG'
              AND p.carries >= ?
              AND p.season >= ?
            ORDER BY p.season, p.week
        """, conn, params=(MIN_CARRIES, MIN_SEASON))

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
                   roll_adj_def_rush_epa,
                   roll_adj_def_rb_epa,
                   rush_def_rank,
                   rb_def_rank,
                   def_games_played
            FROM defense_ratings
        """, conn)

    rb = rb.merge(sched, on=["game_id", "season", "week"], how="left")
    rb["is_home"] = (rb["team"] == rb["home_team"]).astype(int)
    rb["rest"]    = np.where(rb["is_home"] == 1, rb["home_rest"], rb["away_rest"])

    rb = rb.merge(
        def_ratings.rename(columns={"team": "opponent_team"}),
        on=["opponent_team", "season", "week"], how="left"
    )
    for col in ["roll_adj_def_rush_epa", "roll_adj_def_rb_epa"]:
        rb[col] = rb[col].fillna(0.0)
    for col in ["rush_def_rank", "rb_def_rank"]:
        rb[col] = rb[col].fillna(16)
    rb["def_games_played"] = rb["def_games_played"].fillna(0)

    rb["total_line"]  = rb["total_line"].fillna(rb["total_line"].median())
    rb["spread_line"] = rb["spread_line"].fillna(0)
    rb["implied_team_total"] = np.where(
        rb["is_home"] == 1,
        (rb["total_line"] / 2) - (rb["spread_line"] / 2),
        (rb["total_line"] / 2) + (rb["spread_line"] / 2)
    )

    rb = rb.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    for stat in ["rushing_yards", "rushing_tds", "carries",
                 "receptions", "targets", "receiving_yards",
                 "receiving_tds", "rushing_epa", "receiving_epa"]:
        rb[f"roll_{stat}"] = (
            rb.groupby("player_id")[stat]
              .transform(lambda x: x.shift(1).rolling(ROLL_WINDOW, min_periods=2).mean())
        )

    rb["ypc"] = rb["rushing_yards"] / rb["carries"].replace(0, np.nan)
    rb["roll_ypc"] = (
        rb.groupby("player_id")["ypc"]
          .transform(lambda x: x.shift(1).rolling(ROLL_WINDOW, min_periods=2).mean())
    ).fillna(4.0)

    rb["rec_rate"] = rb["receptions"] / rb["targets"].replace(0, np.nan)
    rb["roll_rec_rate"] = (
        rb.groupby("player_id")["rec_rate"]
          .transform(lambda x: x.shift(1).rolling(ROLL_WINDOW, min_periods=2).mean())
    ).fillna(0.75)

    rb["rest"] = rb["rest"].fillna(7)
    return rb


# ── Feature columns ───────────────────────────────────────────────────────────
FEATURE_COLS = [
    "roll_rushing_yards",
    "roll_rushing_tds",
    "roll_carries",
    "roll_ypc",
    "roll_rushing_epa",
    "roll_receptions",
    "roll_targets",
    "roll_receiving_yards",
    "roll_rec_rate",
    "roll_receiving_epa",
    "roll_adj_def_rush_epa",
    "roll_adj_def_rb_epa",
    "rush_def_rank",
    "rb_def_rank",
    "def_games_played",
    "total_line",
    "spread_line",
    "implied_team_total",
    "is_home",
    "rest",
]


# ── Walk-forward backtest ─────────────────────────────────────────────────────
def walk_forward_backtest(df: pd.DataFrame,
                          target: str = "rushing_yards",
                          params: dict = None) -> pd.DataFrame:
    """
    Validate model accuracy using strict walk-forward cross-validation.

    Trains on seasons 1..N-1, tests on season N. Requires MIN_TRAIN_SEASONS
    before the first test season. Rows with null features or target are dropped.

    Args:
        df: Output of load_rb_data().
        target: Stat column to predict (must be in STAT_TARGETS).
        params: XGBoost params dict. Defaults to BEST_PARAMS[target].

    Returns:
        pd.DataFrame: Predictions with player_id, player_name, season, week,
                      team, opponent_team, target, predicted, error columns.
    """
    if params is None:
        params = BEST_PARAMS.get(target, BEST_PARAMS["rushing_yards"])
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
                      target: str = "rushing_yards",
                      params: dict = None) -> XGBRegressor:
    """
    Train on all available data and save model to disk.

    Args:
        df: Output of load_rb_data().
        target: Stat to predict.
        params: XGBoost params. Defaults to BEST_PARAMS[target].

    Returns:
        XGBRegressor: Fitted model (also saved to props_rb_{target}.pkl).
    """
    if params is None:
        params = BEST_PARAMS.get(target, BEST_PARAMS["rushing_yards"])
    clean = df.dropna(subset=FEATURE_COLS + [target])
    model = XGBRegressor(**params, random_state=42, verbosity=0)
    model.fit(clean[FEATURE_COLS], clean[target])
    model_dir = Path(__file__).resolve().parent
    path = model_dir / f"props_rb_{target}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Saved: {path.name}")
    return model


def load_props_model(target: str = "rushing_yards") -> XGBRegressor:
    """Load a trained RB prop model from disk."""
    model_dir = Path(__file__).resolve().parent
    with open(model_dir / f"props_rb_{target}.pkl", "rb") as f:
        return pickle.load(f)


# ── Predict current week ──────────────────────────────────────────────────────
def predict_current_week(models: dict) -> pd.DataFrame:
    """
    Generate RB prop predictions for this week's upcoming games.

    Identifies the lead back per team as the RB with the most carries over
    the last ROLL_WINDOW games in player_stats_current. Only one RB per team
    is predicted — the clear workhorse. If usage is split, the model will
    still pick one; prop betting on committee backfields should be avoided.

    Args:
        models: Dict of { target_name: fitted XGBRegressor }.
                Load via { t: load_props_model(t) for t in STAT_TARGETS }.

    Returns:
        pd.DataFrame: One row per RB/team matchup with pred_{stat} columns.
                      Empty DataFrame if no upcoming games or player data found.
    """
    with get_connection() as conn:
        rb_cur = pd.read_sql("""
            SELECT player_id, player_name, season, week, team,
                   opponent_team, game_id, carries,
                   rushing_yards, rushing_tds, receptions, targets,
                   receiving_yards, receiving_tds,
                   rushing_epa, receiving_epa
            FROM player_stats_current
            WHERE position = 'RB'
              AND season_type = 'REG'
              AND carries >= ?
            ORDER BY week
        """, conn, params=(MIN_CARRIES,))

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

    if rb_cur.empty or upcoming.empty:
        print("No current week data available.")
        return pd.DataFrame()

    def_map   = get_current_defense_ratings().set_index("team").to_dict("index")
    next_week = upcoming["week"].min()
    upcoming  = upcoming[upcoming["week"] == next_week]

    rb_cur = rb_cur.sort_values(["player_id", "week"])
    team_leaders = {}
    for pid, grp in rb_cur.groupby("player_id"):
        last_n = grp.tail(ROLL_WINDOW)
        team   = grp["team"].iloc[-1]
        total_carries = last_n["carries"].sum()
        if team not in team_leaders or total_carries > team_leaders[team]["total_carries"]:
            team_leaders[team] = {
                "player_id":            pid,
                "player_name":          grp["player_name"].iloc[-1],
                "team":                 team,
                "total_carries":        total_carries,
                "roll_rushing_yards":   last_n["rushing_yards"].mean(),
                "roll_rushing_tds":     last_n["rushing_tds"].mean(),
                "roll_carries":         last_n["carries"].mean(),
                "roll_ypc":             last_n["rushing_yards"].sum() / max(last_n["carries"].sum(), 1),
                "roll_rushing_epa":     last_n["rushing_epa"].mean(),
                "roll_receptions":      last_n["receptions"].mean(),
                "roll_targets":         last_n["targets"].mean(),
                "roll_receiving_yards": last_n["receiving_yards"].mean(),
                "roll_rec_rate":        last_n["receptions"].sum() / max(last_n["targets"].sum(), 1),
                "roll_receiving_epa":   last_n["receiving_epa"].mean(),
            }

    rows = []
    for _, game in upcoming.iterrows():
        for side, team, opp in [("home", game["home_team"], game["away_team"]),
                                 ("away", game["away_team"], game["home_team"])]:
            if team not in team_leaders:
                continue
            stats   = team_leaders[team]
            opp_def = def_map.get(opp, {})

            total_line  = game["total_line"] or 45.0
            spread_line = game["spread_line"] or 0.0
            implied     = (total_line / 2) - (spread_line / 2) if side == "home" \
                          else (total_line / 2) + (spread_line / 2)

            rows.append({
                "game_id":               game["game_id"],
                "week":                  next_week,
                "player_id":             stats["player_id"],
                "player_name":           stats["player_name"],
                "team":                  team,
                "opponent":              opp,
                "is_home":               1 if side == "home" else 0,
                "rest":                  game["home_rest"] if side == "home" else game["away_rest"],
                "total_line":            total_line,
                "spread_line":           spread_line,
                "implied_team_total":    implied,
                "roll_adj_def_rush_epa": opp_def.get("roll_adj_def_rush_epa", 0.0),
                "roll_adj_def_rb_epa":   opp_def.get("roll_adj_def_rb_epa",   0.0),
                "rush_def_rank":         opp_def.get("rush_def_rank",           16),
                "rb_def_rank":           opp_def.get("rb_def_rank",             16),
                "def_games_played":      opp_def.get("def_games_played",         0),
                **{k: v for k, v in stats.items()
                   if k not in ["player_id", "player_name", "team", "total_carries"]},
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
    print("Loading RB data...")
    df = load_rb_data()
    print(f"  {len(df):,} RB game rows loaded")

    print("\nWalk-forward backtest (rushing_yards)...")
    bt = walk_forward_backtest(df, target="rushing_yards")
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