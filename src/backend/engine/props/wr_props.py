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
MIN_TARGETS       = 3       # minimum targets to count as a featured WR game
ROLL_WINDOW       = 4       # rolling average window in games
MIN_TRAIN_SEASONS = 4
STAT_TARGETS      = ["receiving_yards", "receiving_tds", "receptions", "targets"]

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
def load_wr_data() -> pd.DataFrame:
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

        # Opponent passing defense — receiving yards allowed to WRs per game
        def_pass = pd.read_sql("""
            SELECT opponent_team AS def_team, season, week,
                   SUM(receiving_yards) AS opp_rec_yards_allowed,
                   SUM(receiving_tds)   AS opp_rec_tds_allowed,
                   SUM(targets)         AS opp_targets_allowed
            FROM player_stats
            WHERE position = 'WR'
              AND season_type = 'REG'
              AND targets >= ?
            GROUP BY opponent_team, season, week
        """, conn, params=(MIN_TARGETS,))

    # ── Merge game context ────────────────────────────────────────────────────
    wr = wr.merge(sched, on=["game_id", "season", "week"], how="left")
    wr["is_home"] = (wr["team"] == wr["home_team"]).astype(int)
    wr["rest"]    = np.where(wr["is_home"] == 1, wr["home_rest"], wr["away_rest"])
    wr["total_line"]  = wr["total_line"].fillna(wr["total_line"].median())
    wr["spread_line"] = wr["spread_line"].fillna(0)

    # Implied team total
    wr["implied_team_total"] = np.where(
        wr["is_home"] == 1,
        (wr["total_line"] / 2) - (wr["spread_line"] / 2),
        (wr["total_line"] / 2) + (wr["spread_line"] / 2)
    )

    # ── Rolling WR stats ──────────────────────────────────────────────────────
    wr = wr.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    for stat in ["receiving_yards", "receiving_tds", "receptions", "targets",
                 "target_share", "air_yards_share", "receiving_air_yards",
                 "receiving_yards_after_catch", "receiving_epa", "racr", "wopr"]:
        if stat in wr.columns:
            wr[f"roll_{stat}"] = (
                wr.groupby("player_id")[stat]
                  .transform(lambda x: x.shift(1).rolling(ROLL_WINDOW, min_periods=2).mean())
            )

    # Yards per reception rolling — efficiency proxy
    wr["ypr"] = wr["receiving_yards"] / wr["receptions"].replace(0, np.nan)
    wr["roll_ypr"] = (
        wr.groupby("player_id")["ypr"]
          .transform(lambda x: x.shift(1).rolling(ROLL_WINDOW, min_periods=2).mean())
    ).fillna(12.0)  # league average ~12 yds/rec for WRs

    # Catch rate rolling
    wr["catch_rate"] = wr["receptions"] / wr["targets"].replace(0, np.nan)
    wr["roll_catch_rate"] = (
        wr.groupby("player_id")["catch_rate"]
          .transform(lambda x: x.shift(1).rolling(ROLL_WINDOW, min_periods=2).mean())
    ).fillna(0.65)

    # Fill NaNs for advanced metrics that may not exist in early seasons
    for col in ["roll_target_share", "roll_air_yards_share", "roll_racr", "roll_wopr"]:
        if col in wr.columns:
            wr[col] = wr[col].fillna(0)

    # ── Rolling opponent pass defense vs WRs ─────────────────────────────────
    def_pass = def_pass.sort_values(["def_team", "season", "week"]).reset_index(drop=True)
    def_pass["roll_rec_allowed"] = (
        def_pass.groupby("def_team")["opp_rec_yards_allowed"]
                .transform(lambda x: x.shift(1).rolling(ROLL_WINDOW, min_periods=2).mean())
    )
    def_pass["roll_rec_tds_allowed"] = (
        def_pass.groupby("def_team")["opp_rec_tds_allowed"]
                .transform(lambda x: x.shift(1).rolling(ROLL_WINDOW, min_periods=2).mean())
    )
    def_pass["roll_targets_allowed"] = (
        def_pass.groupby("def_team")["opp_targets_allowed"]
                .transform(lambda x: x.shift(1).rolling(ROLL_WINDOW, min_periods=2).mean())
    )

    def_agg = def_pass.groupby(["def_team", "season", "week"])[
        ["roll_rec_allowed", "roll_rec_tds_allowed", "roll_targets_allowed"]
    ].mean().reset_index()

    wr = wr.merge(
        def_agg.rename(columns={"def_team": "opponent_team"}),
        on=["opponent_team", "season", "week"], how="left"
    )

    wr["roll_rec_allowed"]      = wr["roll_rec_allowed"].fillna(wr["roll_rec_allowed"].mean())
    wr["roll_rec_tds_allowed"]  = wr["roll_rec_tds_allowed"].fillna(wr["roll_rec_tds_allowed"].mean())
    wr["roll_targets_allowed"]  = wr["roll_targets_allowed"].fillna(wr["roll_targets_allowed"].mean())

    # Season WR defense rank (1=best, 32=worst allowed)
    season_def = (
        wr.groupby(["season", "opponent_team"])["receiving_yards"]
          .mean().reset_index()
          .rename(columns={"receiving_yards": "avg_rec_allowed", "opponent_team": "def_team"})
    )
    season_def["def_rank"] = season_def.groupby("season")["avg_rec_allowed"].rank(ascending=True)
    wr = wr.merge(
        season_def.rename(columns={"def_team": "opponent_team"})[["season", "opponent_team", "def_rank"]],
        on=["season", "opponent_team"], how="left"
    )
    wr["def_rank"] = wr["def_rank"].fillna(16)
    wr["rest"]     = wr["rest"].fillna(7)

    return wr


# ── Feature columns ───────────────────────────────────────────────────────────
FEATURE_COLS = [
    "roll_receiving_yards",     # WR's recent receiving form
    "roll_receiving_tds",
    "roll_receptions",
    "roll_targets",
    "roll_target_share",        # share of team targets
    "roll_air_yards_share",     # share of team air yards (route depth proxy)
    "roll_ypr",                 # yards per reception efficiency
    "roll_catch_rate",
    "roll_receiving_epa",
    "roll_racr",                # receiver air conversion ratio
    "roll_wopr",                # weighted opportunity rating
    "roll_rec_allowed",         # opponent WR defense
    "roll_rec_tds_allowed",
    "roll_targets_allowed",
    "def_rank",                 # opponent WR defense rank this season
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
                          target: str = "receiving_yards",
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
                      target: str = "receiving_yards",
                      params: dict = None) -> XGBRegressor:
    if params is None:
        params = {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.02,
                  "subsample": 0.8, "colsample_bytree": 0.8}
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
    model_dir = Path(__file__).resolve().parent
    with open(model_dir / f"props_wr_{target}.pkl", "rb") as f:
        return pickle.load(f)


# ── Predict current week ──────────────────────────────────────────────────────
def predict_current_week(models: dict) -> pd.DataFrame:
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

        def_cur = pd.read_sql("""
            SELECT opponent_team AS def_team, week,
                   SUM(receiving_yards) AS opp_rec_yards_allowed,
                   SUM(receiving_tds)   AS opp_rec_tds_allowed,
                   SUM(targets)         AS opp_targets_allowed
            FROM player_stats_current
            WHERE position = 'WR'
              AND season_type = 'REG'
              AND targets >= ?
            GROUP BY opponent_team, week
        """, conn, params=(MIN_TARGETS,))

    if wr_cur.empty or upcoming.empty:
        print("No current week data available.")
        return pd.DataFrame()

    next_week = upcoming["week"].min()
    upcoming  = upcoming[upcoming["week"] == next_week]

    # Rolling stats per WR — keep top 3 WRs per team by targets
    wr_cur = wr_cur.sort_values(["player_id", "week"])
    team_wrs = {}
    for pid, grp in wr_cur.groupby("player_id"):
        last_n = grp.tail(ROLL_WINDOW)
        team   = grp["team"].iloc[-1]
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

    # Sort each team's WRs by targets, keep top 3
    for team in team_wrs:
        team_wrs[team] = sorted(team_wrs[team], key=lambda x: x["total_targets"], reverse=True)[:3]

    # Rolling defense
    def_rolling = {}
    for team, grp in def_cur.groupby("def_team"):
        last_n = grp.tail(ROLL_WINDOW)
        def_rolling[team] = {
            "roll_rec_allowed":     last_n["opp_rec_yards_allowed"].mean(),
            "roll_rec_tds_allowed": last_n["opp_rec_tds_allowed"].mean(),
            "roll_targets_allowed": last_n["opp_targets_allowed"].mean(),
        }

    rows = []
    for _, game in upcoming.iterrows():
        for side, team, opp in [("home", game["home_team"], game["away_team"]),
                                 ("away", game["away_team"], game["home_team"])]:
            if team not in team_wrs:
                continue

            total_line  = game["total_line"] or 45.0
            spread_line = game["spread_line"] or 0.0
            implied     = (total_line / 2) - (spread_line / 2) if side == "home" \
                          else (total_line / 2) + (spread_line / 2)

            def_stats = def_rolling.get(opp, {
                "roll_rec_allowed": 180, "roll_rec_tds_allowed": 1.2,
                "roll_targets_allowed": 20,
            })

            for wr in team_wrs[team]:
                row = {
                    "game_id":      game["game_id"],
                    "week":         next_week,
                    "player_id":    wr["player_id"],
                    "player_name":  wr["player_name"],
                    "team":         team,
                    "opponent":     opp,
                    "is_home":      1 if side == "home" else 0,
                    "rest":         game["home_rest"] if side == "home" else game["away_rest"],
                    "total_line":   total_line,
                    "spread_line":  spread_line,
                    "implied_team_total": implied,
                    "def_rank":     16,
                    **{k: v for k, v in wr.items()
                       if k not in ["player_id", "player_name", "team", "total_targets"]},
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

    is_counting = target in ["receiving_tds"]
    if is_counting:
        t1, t2, l1, l2 = 0.5, 1.0, "Exact±0.5", "Within 1"
    else:
        t1, t2, l1, l2 = 15, 30, "Within 15", "Within 30"

    print(f"\n{'='*60}")
    print(f"WR {target} prediction accuracy")
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
    print("Loading WR data...")
    df = load_wr_data()
    print(f"Loaded {len(df)} WR game logs ({df['season'].min()}–{df['season'].max()})")

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
                      "pred_receiving_yards", "pred_receiving_tds",
                      "pred_receptions", "pred_targets"]].to_string(index=False))
    else:
        print("No upcoming games found — run weekly_update.py first.")

    print("\nSaving WR prop predictions to DB...")
    if not preds.empty:
        save(preds, "wr_prop_predictions")
    print("Done.")