"""
001_indexes_and_wal.py — one-time migration for jaginator.db.

Adds indexes on every column the API/pipeline actually filters or joins on,
and switches the database from the default 'delete' journal mode to WAL so
reads aren't blocked by weekly_update.py's writes.

Safe to re-run: every statement uses IF NOT EXISTS, and setting journal_mode
to WAL when it's already WAL is a no-op.

Usage:
    python3.11 data/migrations/001_indexes_and_wal.py
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from data.sql_helpers import get_connection

INDEXES = [
    # schedules — game_id is the join key nearly every other table hangs off of
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_schedules_game_id ON schedules(game_id)",
    "CREATE INDEX IF NOT EXISTS idx_schedules_season_week ON schedules(season, week)",

    # player_stats — filtered by player_id (prop lookups) and joined by game_id
    "CREATE INDEX IF NOT EXISTS idx_player_stats_game ON player_stats(game_id)",
    "CREATE INDEX IF NOT EXISTS idx_player_stats_player_season "
        "ON player_stats(player_id, season, week)",
    "CREATE INDEX IF NOT EXISTS idx_player_stats_current_game ON player_stats_current(game_id)",

    # elo_history — 'latest rating per team' subquery and (team, season, week) scans
    "CREATE INDEX IF NOT EXISTS idx_elo_history_team_season_week "
        "ON elo_history(team, season, week)",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_elo_history_game_team "
        "ON elo_history(game_id, team)",

    # team_stats / rolling / defense — all joined or filtered by game_id + team
    "CREATE INDEX IF NOT EXISTS idx_team_rolling_stats_game_team "
        "ON team_rolling_stats(game_id, team)",
    "CREATE INDEX IF NOT EXISTS idx_defense_ratings_game_team "
        "ON defense_ratings(game_id, team)",
    "CREATE INDEX IF NOT EXISTS idx_team_stats_weekly_season_week "
        "ON team_stats_weekly(season, week, team)",

    # player_features / model_predictions / playoff_predictions — joined by game_id
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_player_features_game ON player_features(game_id)",
    "CREATE INDEX IF NOT EXISTS idx_model_predictions_season_week "
        "ON model_predictions(season, week)",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_playoff_predictions_game "
        "ON playoff_predictions(game_id)",

    # prop predictions — filtered by player_name (LIKE), ordered by week
    "CREATE INDEX IF NOT EXISTS idx_qb_prop_predictions_week ON qb_prop_predictions(week)",
    "CREATE INDEX IF NOT EXISTS idx_rb_prop_predictions_week ON rb_prop_predictions(week)",
    "CREATE INDEX IF NOT EXISTS idx_wr_prop_predictions_week ON wr_prop_predictions(week)",

    # injuries — historical joins by (season, week, team); current by team
    "CREATE INDEX IF NOT EXISTS idx_injuries_historical_season_week_team "
        "ON injuries_historical(season, week, team)",
]


def run() -> None:
    with get_connection() as conn:
        cur = conn.cursor()

        mode = cur.execute("PRAGMA journal_mode=WAL").fetchone()[0]
        print(f"journal_mode -> {mode}")

        for stmt in INDEXES:
            cur.execute(stmt)
            print(f"  ok: {stmt.split('ON')[0].strip().split()[-1]}")

        conn.commit()
    print(f"\nDone — {len(INDEXES)} indexes ensured, journal mode set to WAL.")


if __name__ == "__main__":
    run()