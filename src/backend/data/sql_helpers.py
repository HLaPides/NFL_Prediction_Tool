"""
sql_helpers.py — SQLite connection, read, and write helpers for jaginator.db.

All DB access goes through get_connection(). The DB lives at data/jaginator.db
relative to this file. The context manager (with get_connection() as conn)
uses sqlite3's built-in transaction handling: commits on clean exit, rolls
back on exception.

DB tables fall into three categories:

    Historical (append-only, built once, extended weekly):
        schedules, game_odds, game_conditions, game_personnel,
        team_stats, player_stats, injuries_historical, elo_history,
        team_rolling_stats, player_features, defense_ratings

    Current (replaced wholesale each week):
        schedules_current, player_stats_current, team_stats_current,
        rosters_current, depth_charts_current, injuries_current

    Predictions (written each week by predictor.py):
        weekly_predictions, model_predictions, playoff_predictions,
        qb_prop_predictions, rb_prop_predictions, wr_prop_predictions

Setup:
    First time or after DB loss:
        python3.11 -c "from data.sql_helpers import seed_database; seed_database()"
    Then run seed.py to build rolling stats and player features.
    Then run elo_analyzer.py to build ELO history.
    Then run defense.py to build defense ratings.
"""
import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "jaginator.db"

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
def get_connection() -> sqlite3.Connection:
    """
    Return a sqlite3 connection to jaginator.db.

    Creates the data directory if it doesn't exist. Use as a context manager:
        with get_connection() as conn:
            df = pd.read_sql("SELECT ...", conn)
    sqlite3 connections commit on clean exit and rollback on exception.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


# ── Write ─────────────────────────────────────────────────────────────────────
def save(df: pd.DataFrame, table: str, if_exists: str = "replace") -> None:
    """
    Write a DataFrame to a DB table, replacing it by default.

    Args:
        df: DataFrame to write.
        table: Target table name.
        if_exists: 'replace' (default) or 'append'.
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
    Seed jaginator.db from scratch using historical nflverse data.

    Only run this if the DB is lost or corrupted. After seeding, you still
    need to run:
        python3.11 scripts/seed.py          # builds team_rolling_stats, player_features
        python3.11 engine/elo_analyzer.py   # builds elo_history
        python3.11 engine/defense.py        # builds defense_ratings

    This function only seeds the base schedule tables (schedules, game_odds,
    game_conditions, game_personnel, team_stats).
    """
    from data.getter import get_all_historical
    print(f"Seeding jaginator.db at {DB_PATH}")
    data = get_all_historical()
    raw  = data["schedules"]
    save(raw[[c for c in SCHEDULE_COLS   if c in raw.columns]], "schedules")
    save(raw[[c for c in ODDS_COLS       if c in raw.columns]], "game_odds")
    save(raw[[c for c in CONDITIONS_COLS if c in raw.columns]], "game_conditions")
    save(raw[[c for c in PERSONNEL_COLS  if c in raw.columns]], "game_personnel")
    save(data["team_stats"], "team_stats")
    print("Database seeded.")


# ── Read ──────────────────────────────────────────────────────────────────────
def load_schedules(min_season: int = 1999) -> pd.DataFrame:
    """Load all historical schedules from min_season onward, ordered by season/week."""
    with get_connection() as conn:
        return pd.read_sql(
            "SELECT * FROM schedules WHERE season >= ? ORDER BY season, week",
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
    """Return a list of all table names in jaginator.db."""
    with get_connection() as conn:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return [r[0] for r in rows]


def table_info(table: str) -> None:
    """Print column names and types for a given table."""
    with get_connection() as conn:
        cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
    print(f"\n{table} — {len(cols)} columns:")
    for col in cols:
        print(f"  {col[1]:30s} {col[2]}")