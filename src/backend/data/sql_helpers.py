"""
sql_helpers.py — Postgres connection, read, and write helpers for the
Jaginator database.

All DB access goes through get_connection(). Connection details come from
the DATABASE_URL environment variable (see src/backend/.env), e.g.:

    DATABASE_URL=postgresql+psycopg2://user:password@host:5432/jaginator

The context manager (with get_connection() as conn) wraps a single
transaction via SQLAlchemy's engine.begin(): commits on clean exit, rolls
back on exception — matching the behavior this module had under SQLite.

DB tables fall into three categories:

    Historical (append-only, built once, extended weekly):
        schedules, game_odds, game_conditions, game_personnel,
        team_stats, player_stats, injuries_historical, elo_history,
        team_rolling_stats, player_features, defense_ratings

    Current (replaced wholesale each week — see NOTE below):
        schedules_current, player_stats_current, team_stats_current,
        rosters_current, depth_charts_current, injuries_current

    Predictions (written each week by predictor.py):
        weekly_predictions, model_predictions, playoff_predictions,
        qb_prop_predictions, rb_prop_predictions, wr_prop_predictions

NOTE — "current" tables and foreign keys:
    The *_current tables are dropped and recreated wholesale every week
    (save(..., if_exists="replace")). Postgres enforces foreign keys, and a
    DROP TABLE fails if anything still references it. Because of this, the
    Postgres schema (data/postgres_schema.sql) intentionally does NOT put
    foreign keys on any *_current table, and nothing references them. If a
    future migration needs an FK pointing at a *_current table, switch that
    table's refresh to TRUNCATE + INSERT instead of DROP + CREATE.

Setup:
    1. Create the Postgres database and run data/postgres_schema.sql against it.
    2. Set DATABASE_URL in src/backend/.env.
    3. First time or after DB loss:
        python3.11 -c "from data.sql_helpers import seed_database; seed_database()"
    4. Run seed.py to build rolling stats and player features.
    5. Run elo_analyzer.py to build ELO history.
    6. Run defense.py to build defense ratings.
"""
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Connection

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL not set. Add it to src/backend/.env, e.g.\n"
        "  DATABASE_URL=postgresql+psycopg2://user:password@host:5432/jaginator"
    )

# One engine per process; engine.connect()/engine.begin() checks out a
# pooled connection per call, so this is safe to share across threads.
_engine = create_engine(DATABASE_URL, pool_pre_ping=True)

DEAD_FRANCHISES = {"OAK", "SD", "STL"}

# Column subsets for splitting the raw nflverse schedule into normalized tables
SCHEDULE_COLS = [
    "game_id", "season", "game_type", "week", "gameday", "weekday",
    "gametime", "away_team", "home_team", "away_score", "home_score",
    "location", "result", "total", "overtime", "div_game",
    "away_rest", "home_rest",
]
ODDS_COLS = [
    "game_id", "away_moneyline", "home_moneyline", "spread_line",
    "away_spread_odds", "home_spread_odds", "total_line", "under_odds", "over_odds",
]
CONDITIONS_COLS = [
    "game_id", "roof", "surface", "temp", "wind", "stadium_id", "stadium",
]
PERSONNEL_COLS = [
    "game_id", "away_qb_id", "home_qb_id", "away_qb_name", "home_qb_name",
    "away_coach", "home_coach", "referee",
]


# ── Connection ────────────────────────────────────────────────────────────────
def get_connection() -> Connection:
    """
    Return a SQLAlchemy Connection bound to the Postgres database, wrapped
    in engine.begin() so it behaves like the original sqlite3 pattern:

        with get_connection() as conn:
            df = pd.read_sql("SELECT ...", conn)

    Commits on clean exit, rolls back on exception. Every existing
    pd.read_sql/to_sql call site works unchanged. Raw (non-pandas) queries
    elsewhere in the codebase use conn.exec_driver_sql(sql, params) instead
    of conn.execute(sql, params) — SQLAlchemy 2.0's Connection.execute()
    requires a text()-wrapped Executable, not a plain string; exec_driver_sql
    sends the string straight to psycopg2 using %s-style placeholders, same
    as this codebase already uses throughout.
    """
    return _engine.begin()


# ── Write ─────────────────────────────────────────────────────────────────────
def save(df: pd.DataFrame, table: str, if_exists: str = "replace") -> None:
    """
    Write a DataFrame to a DB table, replacing it by default.

    Args:
        df: DataFrame to write.
        table: Target table name.
        if_exists: 'replace' (default) or 'append'.

    NOTE: 'replace' issues DROP TABLE + CREATE TABLE under Postgres, which
    will fail if another table has a foreign key referencing this one. See
    the *_current table note in this module's docstring.
    """
    with get_connection() as conn:
        df.to_sql(table, conn, if_exists=if_exists, index=False)


def merge_season_into_historical(schedules_current: pd.DataFrame) -> int:
    """
    Append newly completed games from the current schedule into historical tables.

    Filters schedules_current to games with both scores present, then checks
    which game_ids don't already exist in the schedules table. New games are
    split into the four normalized historical tables.

    Args:
        schedules_current: Full current season schedule from nflverse
                           (including both completed and upcoming games).

    Returns:
        int: Number of new games added. 0 if all games already exist.
    """
    completed = schedules_current.dropna(subset=["home_score", "away_score"])

    with get_connection() as conn:
        existing = pd.read_sql("SELECT game_id FROM schedules", conn)["game_id"].tolist()
        new      = completed[~completed["game_id"].isin(existing)]
        if new.empty:
            return 0

        def safe_cols(df, cols):
            return df[[c for c in cols if c in df.columns]]

        safe_cols(new, SCHEDULE_COLS).to_sql("schedules",         conn, if_exists="append", index=False)
        safe_cols(new, ODDS_COLS).to_sql("game_odds",             conn, if_exists="append", index=False)
        safe_cols(new, CONDITIONS_COLS).to_sql("game_conditions", conn, if_exists="append", index=False)
        safe_cols(new, PERSONNEL_COLS).to_sql("game_personnel",   conn, if_exists="append", index=False)

    return len(new)


# ── Seed ──────────────────────────────────────────────────────────────────────
def seed_database() -> None:
    """
    Seed the Postgres database from scratch using historical nflverse data.

    Run data/postgres_schema.sql against the target database FIRST — this
    function only inserts rows, it does not create tables.

    Only run this if the DB is lost or corrupted. After seeding, you still
    need to run:
        python3.11 scripts/seed.py          # builds team_rolling_stats, player_features
        python3.11 engine/elo_analyzer.py   # builds elo_history
        python3.11 engine/defense.py        # builds defense_ratings

    This function only seeds the base schedule tables (schedules, game_odds,
    game_conditions, game_personnel, team_stats).
    """
    from data.getter import get_all_historical
    print(f"Seeding Postgres database at {_engine.url.render_as_string(hide_password=True)}")
    data = get_all_historical()
    raw  = data["schedules"]
    save(raw[[c for c in SCHEDULE_COLS   if c in raw.columns]], "schedules", if_exists="append")
    save(raw[[c for c in ODDS_COLS       if c in raw.columns]], "game_odds", if_exists="append")
    save(raw[[c for c in CONDITIONS_COLS if c in raw.columns]], "game_conditions", if_exists="append")
    save(raw[[c for c in PERSONNEL_COLS  if c in raw.columns]], "game_personnel", if_exists="append")
    save(data["team_stats"], "team_stats", if_exists="append")
    print("Database seeded.")


# ── Read ──────────────────────────────────────────────────────────────────────
def load_schedules(min_season: int = 1999) -> pd.DataFrame:
    """Load all historical schedules from min_season onward, ordered by season/week."""
    with get_connection() as conn:
        return pd.read_sql(
            "SELECT * FROM schedules WHERE season >= %s ORDER BY season, week",
            conn, params=(min_season,)
        )


def load_current_elo_ratings() -> dict:
    """
    Return each active team's most recent ELO rating as a dict.

    Excludes dead franchises (OAK, SD, STL — superseded by LV, LAC, LA).

    Returns:
        dict: { team_abbr: elo_rating_float }
    """
    with get_connection() as conn:
        df = pd.read_sql("""
            SELECT h.team, h.elo_after
            FROM elo_history h
            INNER JOIN (
                SELECT team, MAX(season * 100 + week) AS max_key
                FROM elo_history GROUP BY team
            ) latest ON h.team = latest.team
              AND (h.season * 100 + h.week) = latest.max_key
        """, conn)
    return dict(zip(df[~df["team"].isin(DEAD_FRANCHISES)]["team"],
                    df[~df["team"].isin(DEAD_FRANCHISES)]["elo_after"]))


def load_upcoming_games(schedules_current: pd.DataFrame) -> pd.DataFrame:
    """
    Return this week's unplayed games from the current schedule.

    Filters to rows where home_score is null, then returns only the
    earliest week (next week's games, not the full remaining schedule).

    Args:
        schedules_current: Full current season schedule DataFrame.

    Returns:
        pd.DataFrame: Upcoming games for the nearest unplayed week.
                      Empty DataFrame if all games have been played.
    """
    upcoming = schedules_current[schedules_current["home_score"].isna()]
    if upcoming.empty:
        return upcoming
    return upcoming[upcoming["week"] == upcoming["week"].min()].reset_index(drop=True)


# ── Utility ───────────────────────────────────────────────────────────────────
def list_tables() -> list:
    """Return a list of all table names in the public schema."""
    with get_connection() as conn:
        rows = conn.exec_driver_sql(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' ORDER BY table_name"
        ).fetchall()
    return [r[0] for r in rows]


def table_info(table: str) -> None:
    """Print column names and types for a given table."""
    with get_connection() as conn:
        cols = conn.exec_driver_sql(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = %s "
            "ORDER BY ordinal_position",
            (table,)
        ).fetchall()
    print(f"\n{table} — {len(cols)} columns:")
    for col in cols:
        print(f"  {col[0]:30s} {col[1]}")