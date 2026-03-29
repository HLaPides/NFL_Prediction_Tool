import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from data.sql_helpers import get_connection, save

# ── Constants ─────────────────────────────────────────────────────────────────
MIN_SEASON = 2015   # earliest season with player features


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
                pf.home_pts_allowed,     pf.away_pts_allowed,
                pf.home_pass_yards_allowed, pf.away_pass_yards_allowed,
                pf.home_rush_yards_allowed, pf.away_rush_yards_allowed,
                pf.home_sacks,           pf.away_sacks,
                pf.home_turnovers_forced, pf.away_turnovers_forced,
                         
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
                ar.roll_def_tfl        AS away_roll_def_tfl

            FROM schedules s
            JOIN elo_history h  ON h.game_id = s.game_id AND h.team = s.home_team
            JOIN elo_history a  ON a.game_id = s.game_id AND a.team = s.away_team
            JOIN player_features pf ON pf.game_id = s.game_id
            LEFT JOIN game_odds o        ON o.game_id = s.game_id
            LEFT JOIN game_conditions c  ON c.game_id = s.game_id
            LEFT JOIN team_rolling_stats hr ON hr.game_id = s.game_id AND hr.team = s.home_team
            LEFT JOIN team_rolling_stats ar ON ar.game_id = s.game_id AND ar.team = s.away_team
            WHERE s.game_type = 'REG'
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

    # Derived delta features — model learns from matchup differences
    df["qb_epa_diff"]           = df["home_qb_epa"]          - df["away_qb_epa"]
    df["wr_yards_diff"]         = df["home_wr_yards"]         - df["away_wr_yards"]
    df["te_yards_diff"]         = df["home_te_yards"]         - df["away_te_yards"]
    df["rb_rush_diff"]          = df["home_rb_rush_yards"]    - df["away_rb_rush_yards"]
    df["rb_rec_diff"]           = df["home_rb_rec_yards"]     - df["away_rb_rec_yards"]
    df["pts_allowed_diff"]      = df["home_pts_allowed"]      - df["away_pts_allowed"]
    df["pass_def_diff"]         = df["home_pass_yards_allowed"] - df["away_pass_yards_allowed"]
    df["rush_def_diff"]         = df["home_rush_yards_allowed"] - df["away_rush_yards_allowed"]
    df["sacks_diff"]            = df["home_sacks"]            - df["away_sacks"]
    df["turnovers_forced_diff"] = df["home_turnovers_forced"] - df["away_turnovers_forced"]

    return df


# ── Feature columns ───────────────────────────────────────────────────────────

FEATURE_COLS = [
    # ELO
    "elo_diff", "home_elo", "away_elo",
    # Vegas — single strongest predictor
    "vegas_spread",
    # Game context
    "rest_diff", "div_game",
    "is_dome", "is_cold", "high_wind", "severe_wind", "is_turf",
    # QB — most predictive player signal
    "qb_epa_diff", "home_qb_epa", "away_qb_epa",
    # Skill positions
    "wr_yards_diff",  "home_wr_yards",  "away_wr_yards",
    "te_yards_diff",  "home_te_yards",  "away_te_yards",
    "rb_rush_diff",   "home_rb_rush_yards", "away_rb_rush_yards",
    "rb_rec_diff",
    # Turnovers and pass rush
    "turnovers_forced_diff", "sacks_diff",
    # Rolling team form
    "roll_pass_epa_diff", "roll_rush_epa_diff",
    "roll_pass_yards_diff", "roll_rush_yards_diff",
    "roll_def_sacks_diff", "roll_def_ints_diff", "roll_def_tfl_diff",
    "home_roll_pass_epa", "away_roll_pass_epa",
    "home_roll_pass_yards", "away_roll_pass_yards",
    "home_roll_rush_yards", "away_roll_rush_yards",
]


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
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
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