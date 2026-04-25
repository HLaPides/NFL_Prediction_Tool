import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "jaginator.db"

DEAD_FRANCHISES = {"OAK", "SD", "STL"}

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
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


# ── Write ─────────────────────────────────────────────────────────────────────
def save(df: pd.DataFrame, table: str, if_exists: str = "replace") -> None:
    with get_connection() as conn:
        df.to_sql(table, conn, if_exists=if_exists, index=False)


def merge_season_into_historical(schedules_current: pd.DataFrame) -> int:
    """Appends completed games from the current schedule into historical tables."""
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
    """Seeds the DB from scratch. Only run if the DB is lost or corrupted."""
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
    with get_connection() as conn:
        return pd.read_sql(
            "SELECT * FROM schedules WHERE season >= ? ORDER BY season, week",
            conn, params=(min_season,)
        )

def load_team_stats(min_season: int = 1999) -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql(
            "SELECT * FROM team_stats WHERE season >= ? ORDER BY season",
            conn, params=(min_season,)
        )

def load_game_odds(min_season: int = 1999) -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql("""
            SELECT o.* FROM game_odds o
            JOIN schedules s ON o.game_id = s.game_id
            WHERE s.season >= ?
        """, conn, params=(min_season,))

def load_game_conditions(min_season: int = 1999) -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql("""
            SELECT c.* FROM game_conditions c
            JOIN schedules s ON c.game_id = s.game_id
            WHERE s.season >= ?
        """, conn, params=(min_season,))

def load_game_personnel(min_season: int = 1999) -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql("""
            SELECT p.* FROM game_personnel p
            JOIN schedules s ON p.game_id = s.game_id
            WHERE s.season >= ?
        """, conn, params=(min_season,))

def load_team_games(team: str) -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql(
            "SELECT * FROM schedules WHERE home_team = ? OR away_team = ? ORDER BY season, week",
            conn, params=(team, team)
        )

def load_elo_history() -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql("SELECT * FROM elo_history ORDER BY season, week", conn)

def load_current_elo_ratings() -> dict:
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

def load_depth_charts(starters_only: bool = True) -> pd.DataFrame:
    with get_connection() as conn:
        query = "SELECT * FROM depth_charts_current"
        if starters_only:
            query += " WHERE depth_order = 1"
        return pd.read_sql(query + " ORDER BY team, position, depth_order", conn)

def load_upcoming_games(schedules_current: pd.DataFrame) -> pd.DataFrame:
    upcoming = schedules_current[schedules_current["home_score"].isna()]
    if upcoming.empty:
        return upcoming
    return upcoming[upcoming["week"] == upcoming["week"].min()].reset_index(drop=True)


# ── Utility ───────────────────────────────────────────────────────────────────
def list_tables() -> list:
    with get_connection() as conn:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return [r[0] for r in rows]

def table_info(table: str) -> None:
    with get_connection() as conn:
        cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
    print(f"\n{table} — {len(cols)} columns:")
    for col in cols:
        print(f"  {col[1]:30s} {col[2]}")