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
MIN_CARRIES       = 5       # minimum carries to count as a featured RB game
ROLL_WINDOW       = 4       # rolling average window in games
MIN_TRAIN_SEASONS = 4
STAT_TARGETS      = ["rushing_yards", "rushing_tds", "carries",
                     "receptions", "receiving_yards"]

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
def load_rb_data() -> pd.DataFrame:
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

        # Opponent rushing defense — total rush yards allowed to RBs per game
        def_rush = pd.read_sql("""
            SELECT opponent_team AS def_team, season, week,
                   SUM(rushing_yards) AS opp_rush_yards_allowed,
                   SUM(rushing_tds)   AS opp_rush_tds_allowed,
                   SUM(carries)       AS opp_carries_allowed
            FROM player_stats
            WHERE position = 'RB'
              AND season_type = 'REG'
              AND carries >= ?
            GROUP BY opponent_team, season, week
        """, conn, params=(MIN_CARRIES,))

    # ── Merge game context ────────────────────────────────────────────────────
    rb = rb.merge(sched, on=["game_id", "season", "week"], how="left")
    rb["is_home"] = (rb["team"] == rb["home_team"]).astype(int)
    rb["rest"]    = np.where(rb["is_home"] == 1, rb["home_rest"], rb["away_rest"])
    rb["total_line"]  = rb["total_line"].fillna(rb["total_line"].median())
    rb["spread_line"] = rb["spread_line"].fillna(0)

    # Implied team total
    rb["implied_team_total"] = np.where(
        rb["is_home"] == 1,
        (rb["total_line"] / 2) - (rb["spread_line"] / 2),
        (rb["total_line"] / 2) + (rb["spread_line"] / 2)
    )

    # ── Rolling RB stats ──────────────────────────────────────────────────────
    rb = rb.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    for stat in ["rushing_yards", "rushing_tds", "carries",
                 "receptions", "targets", "receiving_yards",
                 "receiving_tds", "rushing_epa", "receiving_epa"]:
        rb[f"roll_{stat}"] = (
            rb.groupby("player_id")[stat]
              .transform(lambda x: x.shift(1).rolling(ROLL_WINDOW, min_periods=2).mean())
        )

    # Yards per carry rolling — efficiency proxy
    rb["ypc"] = rb["rushing_yards"] / rb["carries"].replace(0, np.nan)
    rb["roll_ypc"] = (
        rb.groupby("player_id")["ypc"]
          .transform(lambda x: x.shift(1).rolling(ROLL_WINDOW, min_periods=2).mean())
    ).fillna(4.0)  # league average ~4.0

    # Reception rate — how often does this RB catch passes
    rb["rec_rate"] = rb["receptions"] / rb["targets"].replace(0, np.nan)
    rb["roll_rec_rate"] = (
        rb.groupby("player_id")["rec_rate"]
          .transform(lambda x: x.shift(1).rolling(ROLL_WINDOW, min_periods=2).mean())
    ).fillna(0.75)

    # ── Rolling opponent rush defense ─────────────────────────────────────────
    def_rush = def_rush.sort_values(["def_team", "season", "week"]).reset_index(drop=True)
    def_rush["roll_rush_allowed"] = (
        def_rush.groupby("def_team")["opp_rush_yards_allowed"]
                .transform(lambda x: x.shift(1).rolling(ROLL_WINDOW, min_periods=2).mean())
    )
    def_rush["roll_rush_tds_allowed"] = (
        def_rush.groupby("def_team")["opp_rush_tds_allowed"]
                .transform(lambda x: x.shift(1).rolling(ROLL_WINDOW, min_periods=2).mean())
    )
    def_rush["roll_carries_allowed"] = (
        def_rush.groupby("def_team")["opp_carries_allowed"]
                .transform(lambda x: x.shift(1).rolling(ROLL_WINDOW, min_periods=2).mean())
    )

    def_agg = def_rush.groupby(["def_team", "season", "week"])[
        ["roll_rush_allowed", "roll_rush_tds_allowed", "roll_carries_allowed"]
    ].mean().reset_index()

    rb = rb.merge(
        def_agg.rename(columns={"def_team": "opponent_team"}),
        on=["opponent_team", "season", "week"], how="left"
    )

    rb["roll_rush_allowed"]      = rb["roll_rush_allowed"].fillna(rb["roll_rush_allowed"].mean())
    rb["roll_rush_tds_allowed"]  = rb["roll_rush_tds_allowed"].fillna(rb["roll_rush_tds_allowed"].mean())
    rb["roll_carries_allowed"]   = rb["roll_carries_allowed"].fillna(rb["roll_carries_allowed"].mean())

    # Season rush defense rank (1=best, 32=worst allowed)
    season_def = (
        rb.groupby(["season", "opponent_team"])["rushing_yards"]
          .mean().reset_index()
          .rename(columns={"rushing_yards": "avg_rush_allowed", "opponent_team": "def_team"})
    )
    season_def["def_rank"] = season_def.groupby("season")["avg_rush_allowed"].rank(ascending=True)
    rb = rb.merge(
        season_def.rename(columns={"def_team": "opponent_team"})[["season", "opponent_team", "def_rank"]],
        on=["season", "opponent_team"], how="left"
    )
    rb["def_rank"] = rb["def_rank"].fillna(16)
    rb["rest"]     = rb["rest"].fillna(7)

    return rb


# ── Feature columns ───────────────────────────────────────────────────────────
FEATURE_COLS = [
    "roll_rushing_yards",       # RB's recent rushing form
    "roll_rushing_tds",
    "roll_carries",
    "roll_ypc",                 # yards per carry efficiency
    "roll_rushing_epa",
    "roll_receptions",          # receiving role
    "roll_targets",
    "roll_receiving_yards",
    "roll_rec_rate",
    "roll_receiving_epa",
    "roll_rush_allowed",        # opponent rush defense
    "roll_rush_tds_allowed",
    "roll_carries_allowed",
    "def_rank",                 # opponent rush defense rank this season
    "total_line",               # game environment
    "spread_line",
    "implied_team_total",
    "is_home",
    "rest",
]


# ── Hyperparameter tuning ─────────────────────────────────────────────────────
def tune_hyperparameters(df: pd.DataFrame, target: str, n_folds: int = 5) -> dict:
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
                          target: str = "rushing_yards",
                          params: dict = None) -> pd.DataFrame:
    if params is None:
        params = {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.02,
                  "subsample": 0.8, "colsample_bytree": 0.8}
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


# ── Train & save final model ──────────────────────────────────────────────────
def train_final_model(df: pd.DataFrame,
                      target: str = "rushing_yards",
                      params: dict = None) -> XGBRegressor:
    if params is None:
        params = {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.02,
                  "subsample": 0.8, "colsample_bytree": 0.8}
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
    model_dir = Path(__file__).resolve().parent
    with open(model_dir / f"props_rb_{target}.pkl", "rb") as f:
        return pickle.load(f)


# ── Predict current week ──────────────────────────────────────────────────────
def predict_current_week(models: dict) -> pd.DataFrame:
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

        def_cur = pd.read_sql("""
            SELECT opponent_team AS def_team, week,
                   SUM(rushing_yards) AS opp_rush_yards_allowed,
                   SUM(rushing_tds)   AS opp_rush_tds_allowed,
                   SUM(carries)       AS opp_carries_allowed
            FROM player_stats_current
            WHERE position = 'RB'
              AND season_type = 'REG'
              AND carries >= ?
            GROUP BY opponent_team, week
        """, conn, params=(MIN_CARRIES,))

    if rb_cur.empty or upcoming.empty:
        print("No current week data available.")
        return pd.DataFrame()

    next_week = upcoming["week"].min()
    upcoming  = upcoming[upcoming["week"] == next_week]

    # Rolling stats per RB — keep top RB per team by carries
    rb_cur = rb_cur.sort_values(["player_id", "week"])
    team_leaders = {}
    for pid, grp in rb_cur.groupby("player_id"):
        last_n = grp.tail(ROLL_WINDOW)
        team   = grp["team"].iloc[-1]
        total_carries = last_n["carries"].sum()
        if team not in team_leaders or total_carries > team_leaders[team]["total_carries"]:
            team_leaders[team] = {
                "player_id":           pid,
                "player_name":         grp["player_name"].iloc[-1],
                "team":                team,
                "total_carries":       total_carries,
                "roll_rushing_yards":  last_n["rushing_yards"].mean(),
                "roll_rushing_tds":    last_n["rushing_tds"].mean(),
                "roll_carries":        last_n["carries"].mean(),
                "roll_ypc":            last_n["rushing_yards"].sum() / max(last_n["carries"].sum(), 1),
                "roll_rushing_epa":    last_n["rushing_epa"].mean(),
                "roll_receptions":     last_n["receptions"].mean(),
                "roll_targets":        last_n["targets"].mean(),
                "roll_receiving_yards":last_n["receiving_yards"].mean(),
                "roll_rec_rate":       last_n["receptions"].sum() / max(last_n["targets"].sum(), 1),
                "roll_receiving_epa":  last_n["receiving_epa"].mean(),
            }

    # Rolling defense
    def_rolling = {}
    for team, grp in def_cur.groupby("def_team"):
        last_n = grp.tail(ROLL_WINDOW)
        def_rolling[team] = {
            "roll_rush_allowed":     last_n["opp_rush_yards_allowed"].mean(),
            "roll_rush_tds_allowed": last_n["opp_rush_tds_allowed"].mean(),
            "roll_carries_allowed":  last_n["opp_carries_allowed"].mean(),
        }

    rows = []
    for _, game in upcoming.iterrows():
        for side, team, opp in [("home", game["home_team"], game["away_team"]),
                                 ("away", game["away_team"], game["home_team"])]:
            if team not in team_leaders:
                continue

            stats     = team_leaders[team]
            def_stats = def_rolling.get(opp, {
                "roll_rush_allowed": 110, "roll_rush_tds_allowed": 0.9,
                "roll_carries_allowed": 25,
            })

            total_line  = game["total_line"] or 45.0
            spread_line = game["spread_line"] or 0.0
            implied     = (total_line / 2) - (spread_line / 2) if side == "home" \
                          else (total_line / 2) + (spread_line / 2)

            row = {
                "game_id":      game["game_id"],
                "week":         next_week,
                "player_id":    stats["player_id"],
                "player_name":  stats["player_name"],
                "team":         team,
                "opponent":     opp,
                "is_home":      1 if side == "home" else 0,
                "rest":         game["home_rest"] if side == "home" else game["away_rest"],
                "total_line":   total_line,
                "spread_line":  spread_line,
                "implied_team_total": implied,
                "def_rank":     16,
                **{k: v for k, v in stats.items()
                   if k not in ["player_id", "player_name", "team", "total_carries"]},
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

    is_counting = target in ["rushing_tds", "receiving_tds"]
    if is_counting:
        t1, t2, l1, l2 = 0.5, 1.0, "Exact±0.5", "Within 1"
    else:
        t1, t2, l1, l2 = 15, 30, "Within 15", "Within 30"

    print(f"\n{'='*60}")
    print(f"RB {target} prediction accuracy")
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
    print("Loading RB data...")
    df = load_rb_data()
    print(f"Loaded {len(df)} RB game logs ({df['season'].min()}–{df['season'].max()})")

    print("\nNaN counts per feature:")
    print(df[FEATURE_COLS].isna().sum())

    print("\nTuning hyperparameters and running backtests...")
    best_params = {}
    models = {}
    for target in STAT_TARGETS:
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
                      "pred_rushing_yards", "pred_rushing_tds",
                      "pred_carries", "pred_receiving_yards"]].to_string(index=False))
    else:
        print("No upcoming games found — run weekly_update.py first.")

    print("\nSaving RB prop predictions to DB...")
    if not preds.empty:
        save(preds, "rb_prop_predictions")
    print("Done.")