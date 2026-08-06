"""
002_sqlite_to_postgres.py — one-time data migration from the legacy
jaginator.db (SQLite) into Postgres.

Prerequisites:
    1. Postgres database created.
    2. data/postgres_schema.sql already run against it (creates empty tables).
    3. DATABASE_URL set in src/backend/.env, pointing at the new Postgres DB.
    4. The old jaginator.db still present at data/jaginator.db (read-only source).

This script does NOT touch jaginator.db — it only reads from it. Safe to
re-run: each table is loaded with if_exists="append" after being cleared
with TRUNCATE, so re-running replaces rather than duplicates rows.

Usage:
    python3.11 data/migrations/002_sqlite_to_postgres.py
"""
import sqlite3
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd

from data.sql_helpers import get_connection

SQLITE_PATH = Path(__file__).resolve().parents[1] / "jaginator.db"

# Order matters where foreign keys exist: schedules first, then anything
# that references schedules(game_id), then everything else.
TABLES_IN_ORDER = [
    "schedules",
    "game_odds", "game_conditions", "game_personnel",
    "team_stats", "player_stats", "injuries_historical",
    "elo_history", "team_rolling_stats", "player_features", "defense_ratings",
    "line_movement",
    "model_predictions", "playoff_predictions", "weekly_predictions",
    "qb_prop_predictions", "rb_prop_predictions", "wr_prop_predictions",
    "schedules_current", "player_stats_current", "team_stats_current",
    "team_stats_weekly", "rosters_current", "depth_charts_current", "injuries_current",
]


def migrate_table(sqlite_conn: sqlite3.Connection, table: str) -> int:
    df = pd.read_sql(f"SELECT * FROM {table}", sqlite_conn)
    if df.empty:
        print(f"  {table}: 0 rows (nothing to migrate)")
        return 0

    with get_connection() as pg_conn:
        pg_conn.exec_driver_sql(f"TRUNCATE TABLE {table} CASCADE")

        # Drop rows that would violate a NOT NULL constraint in the new
        # schema (SQLite never enforced these, so a handful of garbage rows
        # can exist in the source data — e.g. a single team_stats row with
        # a NULL team). Report what's dropped rather than crashing.
        not_null_cols = pg_conn.exec_driver_sql(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema='public' AND table_name = %s AND is_nullable = 'NO'",
            (table,)
        ).fetchall()
        not_null_cols = [c[0] for c in not_null_cols if c[0] in df.columns]

        if not_null_cols:
            before = len(df)
            bad_mask = df[not_null_cols].isna().any(axis=1)
            if bad_mask.any():
                print(f"  {table}: dropping {bad_mask.sum()} row(s) with NULL "
                      f"in required column(s) {not_null_cols}")
                print(f"    sample: {df[bad_mask].head(3).to_dict('records')}")
            df = df[~bad_mask]

        # Tables with a SERIAL id column (id) shouldn't have that column
        # supplied from SQLite — let Postgres assign fresh ids.
        cols = pg_conn.exec_driver_sql(
            "SELECT column_name, column_default FROM information_schema.columns "
            "WHERE table_schema='public' AND table_name = %s",
            (table,)
        ).fetchall()
        serial_cols = {c[0] for c in cols if c[1] and "nextval" in c[1]}
        df = df.drop(columns=[c for c in serial_cols if c in df.columns], errors="ignore")

        df.to_sql(table, pg_conn, if_exists="append", index=False)

    print(f"  {table}: {len(df):,} rows migrated")
    return len(df)


def run() -> None:
    if not SQLITE_PATH.exists():
        raise FileNotFoundError(f"Source SQLite DB not found at {SQLITE_PATH}")

    sqlite_conn = sqlite3.connect(SQLITE_PATH)
    total = 0
    print(f"Migrating {SQLITE_PATH} -> Postgres\n")
    for table in TABLES_IN_ORDER:
        try:
            total += migrate_table(sqlite_conn, table)
        except Exception as e:
            print(f"  {table}: FAILED ({e})")
            print("  Stopping — fix the error above before re-running.")
            sqlite_conn.close()
            raise
    sqlite_conn.close()
    print(f"\nDone — {total:,} total rows migrated across {len(TABLES_IN_ORDER)} tables.")


if __name__ == "__main__":
    run()