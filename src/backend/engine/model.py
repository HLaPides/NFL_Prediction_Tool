import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from data.sql_helpers import get_connection, save
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_absolute_error

MIN_SEASON = 1999


# ── Data loading ──────────────────────────────────────────────────────────────

def load_training_data() -> pd.DataFrame:
    """
    Joins all sources into one flat DataFrame. One row per regular season game.
    Target: result (home team point differential).
    """
    with get_connection() as conn:
        df = pd.read_sql("""
            SELECT
                s.game_id, s.season, s.week,
                s.home_team, s.away_team,
                s.result,
                COALESCE(s.home_rest, 7) - COALESCE(s.away_rest, 7) AS rest_diff,
                COALESCE(s.div_game, 0)                              AS div_game,
                h.elo_before  AS home_elo,
                a.elo_before  AS away_elo,
                (h.elo_before - a.elo_before)                        AS elo_diff,
                o.spread_line AS vegas_spread,
                CASE WHEN c.roof IN ('dome', 'closed') THEN 1 ELSE 0 END AS is_dome,
                CASE WHEN CAST(COALESCE(c.temp, 60) AS FLOAT) < 40 THEN 1 ELSE 0 END AS is_cold,
                CASE WHEN CAST(COALESCE(c.wind, 0)  AS FLOAT) > 15 THEN 1 ELSE 0 END AS high_wind,
                CASE WHEN CAST(COALESCE(c.wind, 0)  AS FLOAT) > 25 THEN 1 ELSE 0 END AS severe_wind,
                CASE WHEN c.surface NOT IN ('grass', 'Grass') THEN 1 ELSE 0 END       AS is_turf,

                 pf.home_qb_epa,          pf.away_qb_epa,
                pf.home_qb_games,        pf.away_qb_games,
                pf.home_wr_yards,        pf.away_wr_yards,
                pf.home_wr_target_share, pf.away_wr_target_share,
                pf.home_te_yards,        pf.away_te_yards,
                pf.home_te_target_share, pf.away_te_target_share,
                pf.home_rb_rush_yards,   pf.away_rb_rush_yards,
                pf.home_rb_rec_yards,    pf.away_rb_rec_yards,
                pf.home_rb_carry_share,  pf.away_rb_carry_share,

                hr.roll_passing_epa    AS home_roll_pass_epa,
                hr.roll_rushing_epa    AS home_roll_rush_epa,
                hr.roll_passing_yards  AS home_roll_pass_yards,
                hr.roll_rushing_yards  AS home_roll_rush_yards,
                hr.roll_def_sacks      AS home_roll_def_sacks,
                hr.roll_def_ints       AS home_roll_def_ints,
                hr.roll_def_tfl        AS home_roll_def_tfl,

                ar.roll_passing_epa    AS away_roll_pass_epa,
                ar.roll_rushing_epa    AS away_roll_rush_epa,
                ar.roll_passing_yards  AS away_roll_pass_yards,
                ar.roll_rushing_yards  AS away_roll_rush_yards,
                ar.roll_def_sacks      AS away_roll_def_sacks,
                ar.roll_def_ints       AS away_roll_def_ints,
                ar.roll_def_tfl        AS away_roll_def_tfl,
                         
                hr.roll_avg_margin     AS home_roll_avg_margin,
                hr.roll_margin_trend   AS home_roll_margin_trend,
                hr.roll_blowout_wins   AS home_roll_blowout_wins,
                hr.roll_blowout_losses AS home_roll_blowout_losses,
                         
                ar.roll_avg_margin     AS away_roll_avg_margin,
                ar.roll_margin_trend   AS away_roll_margin_trend,
                ar.roll_blowout_wins   AS away_roll_blowout_wins,
                ar.roll_blowout_losses AS away_roll_blowout_losses,
                         
                hr.roll_blowout_losses AS home_roll_blowout_losses,
                hr.roll_turnovers      AS home_roll_turnovers,
                hr.roll_first_downs    AS home_roll_first_downs,
                hr.roll_penalty_yards  AS home_roll_penalty_yards,

                ar.roll_blowout_losses AS away_roll_blowout_losses,
                ar.roll_turnovers      AS away_roll_turnovers,
                ar.roll_first_downs    AS away_roll_first_downs,
                ar.roll_penalty_yards  AS away_roll_penalty_yards
                         
            FROM schedules s
            JOIN elo_history h  ON h.game_id = s.game_id AND h.team = s.home_team
            JOIN elo_history a  ON a.game_id = s.game_id AND a.team = s.away_team
            JOIN player_features pf ON pf.game_id = s.game_id
            LEFT JOIN game_odds o        ON o.game_id = s.game_id
            LEFT JOIN game_conditions c  ON c.game_id = s.game_id
            LEFT JOIN team_rolling_stats hr ON hr.game_id = s.game_id AND hr.team = s.home_team
            LEFT JOIN team_rolling_stats ar ON ar.game_id = s.game_id AND ar.team = s.away_team
            WHERE s.game_type IN ('REG', 'WC', 'DIV', 'CON', 'SB')
            AND s.result IS NOT NULL
            AND s.season >= ?
            ORDER BY s.season, s.week
        """, conn, params=(MIN_SEASON,))

    # Rolling deltas
    df["roll_pass_epa_diff"]   = df["home_roll_pass_epa"]   - df["away_roll_pass_epa"]
    df["roll_rush_epa_diff"]   = df["home_roll_rush_epa"]   - df["away_roll_rush_epa"]
    df["roll_pass_yards_diff"] = df["home_roll_pass_yards"] - df["away_roll_pass_yards"]
    df["roll_rush_yards_diff"] = df["home_roll_rush_yards"] - df["away_roll_rush_yards"]
    df["roll_def_sacks_diff"]  = df["home_roll_def_sacks"]  - df["away_roll_def_sacks"]
    df["roll_def_ints_diff"]   = df["home_roll_def_ints"]   - df["away_roll_def_ints"]
    df["roll_def_tfl_diff"]    = df["home_roll_def_tfl"]    - df["away_roll_def_tfl"]
    df["roll_avg_margin_diff"]   = df["home_roll_avg_margin"]   - df["away_roll_avg_margin"]
    df["roll_margin_trend_diff"] = df["home_roll_margin_trend"] - df["away_roll_margin_trend"]
    df["roll_turnovers_diff"]  = df["home_roll_turnovers"]     - df["away_roll_turnovers"]
    df["roll_first_downs_diff"]= df["home_roll_first_downs"]   - df["away_roll_first_downs"]
    df["roll_penalty_diff"]    = df["home_roll_penalty_yards"] - df["away_roll_penalty_yards"]

    # Derived delta features — model learns from matchup differences
    df["qb_epa_diff"]           = df["home_qb_epa"]          - df["away_qb_epa"]
    df["wr_yards_diff"]         = df["home_wr_yards"]         - df["away_wr_yards"]
    df["te_yards_diff"]         = df["home_te_yards"]         - df["away_te_yards"]
    df["rb_rush_diff"]          = df["home_rb_rush_yards"]    - df["away_rb_rush_yards"]
    df["rb_rec_diff"]           = df["home_rb_rec_yards"]     - df["away_rb_rec_yards"]
    

    return df


# ── Feature columns ───────────────────────────────────────────────────────────

FEATURE_COLS = [
    "elo_diff", "home_elo", "away_elo",
    "vegas_spread",
    "rest_diff", "div_game",
    "is_dome", "is_cold", "high_wind", "severe_wind", "is_turf",
    "qb_epa_diff", "home_qb_epa", "away_qb_epa",
    "wr_yards_diff",  "home_wr_yards",  "away_wr_yards",
    "te_yards_diff",  "home_te_yards",  "away_te_yards",
    "rb_rush_diff",   "home_rb_rush_yards", "away_rb_rush_yards",
    "rb_rec_diff",
    "roll_pass_epa_diff", "roll_rush_epa_diff",
    "roll_pass_yards_diff", "roll_rush_yards_diff",
    "roll_def_sacks_diff", "roll_def_ints_diff", "roll_def_tfl_diff",
    "home_roll_pass_epa", "away_roll_pass_epa",
    "home_roll_pass_yards", "away_roll_pass_yards",
    "home_roll_rush_yards", "away_roll_rush_yards",
    "home_roll_turnovers",  "away_roll_turnovers",  
    "home_roll_first_downs","away_roll_first_downs",
    "home_roll_penalty_yards", "away_roll_penalty_yards",
    "roll_turnovers_diff", "roll_first_downs_diff", "roll_penalty_diff"
]


def tune_hyperparameters(df: pd.DataFrame) -> dict:
    """
    Runs grid search on a single train/test split (2020-2023 train, 2024 test)
    to find best XGBoost params before running full walk-forward backtest.
    """
    train = df[df["season"] < 2025]
    test  = df[df["season"] == 2025]

    X_train = train[FEATURE_COLS].fillna(0)
    y_train = train["result"]
    X_test  = test[FEATURE_COLS].fillna(0)
    y_test  = test["result"]

    param_grid = {
        "n_estimators":    [100, 200, 400],
        "max_depth":       [3, 4, 6],
        "learning_rate":   [0.01, 0.05, 0.1],
        "subsample":       [0.7, 0.8, 1.0],
        "colsample_bytree":[0.7, 0.8, 1.0],
    }

    model = XGBRegressor(random_state=42, verbosity=0)
    scorer = make_scorer(mean_absolute_error, greater_is_better=False)

    search = GridSearchCV(model, param_grid, scoring=scorer,
                          cv=3, n_jobs=-1, verbose=2)
    search.fit(X_train, y_train)

    print(f"\nBest params: {search.best_params_}")
    print(f"Best CV MAE: {-search.best_score_:.2f}")

    best = search.best_estimator_
    preds = best.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    winner_pct = (
        (np.where(preds > 0, test["home_team"], test["away_team"]) ==
         np.where(test["result"] > 0, test["home_team"], test["away_team"]))
    ).mean() * 100
    print(f"2025 MAE: {mae:.2f}")
    print(f"2025 Winner%: {winner_pct:.1f}%")

    return search.best_params_

# ── Walk-forward validation ───────────────────────────────────────────────────

def walk_forward_backtest(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each season S, trains on all seasons < S and tests on season S.
    Returns a DataFrame with predictions and value bet flags for every test game.
    """
    seasons = sorted(df["season"].unique())
    results = []

    for i, test_season in enumerate(seasons):
        train_seasons = seasons[:i]
        if not train_seasons:
            continue  # need at least one season of training data

        train = df[df["season"].isin(train_seasons)]
        test  = df[df["season"] == test_season]

        X_train = train[FEATURE_COLS].fillna(0)
        y_train = train["result"]
        X_test  = test[FEATURE_COLS].fillna(0)

        model = XGBRegressor(
                n_estimators=200, 
                max_depth=3, 
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=1.0,
                verbosity=0,
                random_state = 10211949
            )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        season_df = test[["game_id", "season", "week", "home_team", "away_team",
                           "result", "vegas_spread"]].copy()
        season_df["model_spread"] = preds.round(1)
        season_df["spread_gap"]   = (season_df["model_spread"] - season_df["vegas_spread"]).round(1)

        season_df["model_winner"] = np.where(season_df["model_spread"] > 0,
                                              season_df["home_team"],
                                              season_df["away_team"])
        season_df["vegas_winner"] = np.where(season_df["vegas_spread"] > 0,
                                              season_df["home_team"],
                                              season_df["away_team"])
        season_df["actual_winner"] = np.where(season_df["result"] > 0,
                                               season_df["home_team"],
                                               season_df["away_team"])

        results.append(season_df)

    return pd.concat(results, ignore_index=True)


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_report(df: pd.DataFrame) -> None:
    seasons = sorted(df["season"].unique())

    print(f"\n{'='*55}")
    print(f"{'Season':<8} {'Games':>6} {'MAE':>6} {'Winner%':>8} {'vs Vegas%':>10}")
    print(f"{'='*55}")

    total_correct = total_vegas = total_games = 0

    for season in seasons:
        s = df[df["season"] == season]

        mae        = mean_absolute_error(s["result"], s["model_spread"])
        winner_pct = (s["model_winner"] == s["actual_winner"]).mean() * 100
        vegas_pct  = (s["vegas_winner"] == s["actual_winner"]).mean() * 100
        edge       = winner_pct - vegas_pct

        total_correct += (s["model_winner"] == s["actual_winner"]).sum()
        total_vegas   += (s["vegas_winner"] == s["actual_winner"]).sum()
        total_games   += len(s)

        print(f"{season:<8} {len(s):>6} {mae:>6.1f} {winner_pct:>7.1f}%  {edge:>+8.1f}%")

    print(f"{'='*55}")
    print(f"{'All-time':<8} {total_games:>6}        "
          f"{total_correct/total_games*100:>7.1f}%  "
          f"{(total_correct-total_vegas)/total_games*100:>+8.1f}%")


if __name__ == "__main__":
    print("Loading training data...")
    df = load_training_data()
    print(f"Loaded {len(df)} games ({df['season'].min()}–{df['season'].max()})")
    print(f"Features: {len(FEATURE_COLS)}")

    print("\nRunning walk-forward backtest...")
    results = walk_forward_backtest(df)

    print_report(results)

    print("\nSaving predictions to model_predictions table...")
    save(results, "model_predictions")
    print("Done.")

    """
    print("Loading training data...")
    df = load_training_data()

    print("Tuning hyperparameters...")
    best_params = tune_hyperparameters(df)
    """