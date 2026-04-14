import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import sqlite3
import pandas as pd
import time
from pathlib import Path
from data.getter import get_all_historical, get_current_roster, get_current_weekly_stats

DB_PATH = Path(__file__).resolve().parent / "jaginator.db"

SCHEDULE_COLS = [
    "game_id", "season", "game_type", "week", "gameday", "weekday",
    "gametime", "away_team", "home_team", "away_score", "home_score",
    "location", "result", "total", "overtime", "div_game",
    "away_rest", "home_rest"
]

ODDS_COLS = [
    "game_id", "away_moneyline", "home_moneyline", "spread_line",
    "away_spread_odds", "home_spread_odds", "total_line", "under_odds", "over_odds"
]

CONDITIONS_COLS = [
    "game_id", "roof", "surface", "temp", "wind", "stadium_id", "stadium"
]

PERSONNEL_COLS = [
    "game_id", "away_qb_id", "home_qb_id", "away_qb_name", "home_qb_name",
    "away_coach", "home_coach", "referee"
]

DEAD_FRANCHISES = {"OAK", "SD", "STL"}


#Gets connection to sql database

def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)

#Save functions saves the pd.Dataframe passed in to the header into jaginator.db

def save(df: pd.DataFrame, table: str, if_exists: str = "replace") -> None:
    with get_connection() as conn:
        df.to_sql(table, conn, if_exists=if_exists, index=False)

def merge_season_into_historical(schedules_current: pd.DataFrame) -> int:
    completed = schedules_current.dropna(subset=["home_score", "away_score"])

    with get_connection() as conn:
        existing = pd.read_sql("SELECT game_id FROM schedules", conn)["game_id"].tolist()
        new      = completed[~completed["game_id"].isin(existing)]

        if new.empty:
            return 0

        def safe_slice(df, cols):
            return df[[c for c in cols if c in df.columns]]

        new.pipe(safe_slice, SCHEDULE_COLS).to_sql("schedules",         conn, if_exists="append", index=False)
        new.pipe(safe_slice, ODDS_COLS).to_sql("game_odds",             conn, if_exists="append", index=False)
        new.pipe(safe_slice, CONDITIONS_COLS).to_sql("game_conditions", conn, if_exists="append", index=False)
        new.pipe(safe_slice, PERSONNEL_COLS).to_sql("game_personnel",   conn, if_exists="append", index=False)

    return len(new)

#This seeds the database. Only run if something happens to the db
def seed_database() -> None:
    print(f"Seeding jaginator.db at {DB_PATH}")
    data = get_all_historical()

    raw = data["schedules"]
    save(raw[SCHEDULE_COLS],    "schedules")
    save(raw[ODDS_COLS],        "game_odds")
    save(raw[CONDITIONS_COLS],  "game_conditions")
    save(raw[PERSONNEL_COLS],   "game_personnel")
    save(data["team_stats"],    "team_stats")

    print("Database seeded.")

#Load functions. They all load their respective tables. IE load_team_stats the team_stats table

def load_schedules(min_season: int = 1999) -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql(f""" SELECT * FROM schedules WHERE season >= {min_season} ORDER BY season""", conn)

def load_team_stats(min_season: int = 1999) -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql(f"""SELECT * FROM team_stats WHERE season >= {min_season} ORDER BY season""", conn)
    
def load_game_odds(min_season: int = 1999) -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql(f"""
            SELECT game_odds.* FROM game_odds
            JOIN schedules ON game_odds.game_id = schedules.game_id
            WHERE schedules.season >= {min_season}
        """, conn)

def load_game_conditions(min_season: int = 1999) -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql(f"""
            SELECT game_conditions.* FROM game_conditions
            JOIN schedules ON game_conditions.game_id = schedules.game_id
            WHERE schedules.season >= {min_season}
        """, conn)


def load_game_personnel(min_season: int = 1999) -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql(f"""
            SELECT game_personnel.* FROM game_personnel
            JOIN schedules ON game_personnel.game_id = schedules.game_id
            WHERE schedules.season >= {min_season}
        """, conn)

def load_team_games(team: str) -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql(f"""SELECT * FROM schedules WHERE home_team = ? OR away_team = ? ORDER BY season, week""", conn, params=(team,team))

def load_elo_history() -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql("SELECT * FROM elo_history ORDER BY season, week", conn)

def load_current_elo_ratings() -> dict[str, float]:
    with get_connection() as conn:
        df = pd.read_sql("""
            SELECT h.team, h.elo_after
            FROM elo_history h
            INNER JOIN (
                SELECT team, MAX(season * 100 + week) AS max_key
                FROM elo_history
                GROUP BY team
            ) latest
            ON h.team = latest.team
            AND (h.season * 100 + h.week) = latest.max_key
        """, conn)
    df = df[~df["team"].isin(DEAD_FRANCHISES)]
    return dict(zip(df["team"], df["elo_after"]))
   
def load_depth_charts(starters_only: bool = True) -> pd.DataFrame:
    with get_connection() as conn:
        query = "SELECT * FROM depth_charts_current"
        if starters_only:
            query += " WHERE depth_order = 1"
        query += " ORDER BY team, position, depth_order"
        return pd.read_sql(query, conn)
    
def load_upcoming_games(schedules_current: pd.DataFrame) -> pd.DataFrame:
    upcoming  = schedules_current[schedules_current["home_score"].isna()]
    if upcoming.empty:
        return upcoming
    next_week = upcoming["week"].min()
    return upcoming[upcoming["week"] ==next_week].reset_index(drop=True)

#Utility functions. List_tables() lists the 5 tables the db has, table_info prints all the columns of those tables

def list_tables() -> list[str]:
    with get_connection() as conn:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return [row[0] for row in rows]

def table_info(table: str) -> None:
    with get_connection() as conn:
        current = conn.execute(f"PRAGMA table_info({table})")
        cols = current.fetchall()
        print(f"\n{table} - {len(cols)} columns:")
        for col in cols:
            print(f"{col[1]:30s} {col[2]}")
DB_PATH = Path(__file__).resolve().parent / "jaginator.db"

SCHEDULE_COLS = [
    "game_id", "season", "game_type", "week", "gameday", "weekday",
    "gametime", "away_team", "home_team", "away_score", "home_score",
    "location", "result", "total", "overtime", "div_game",
    "away_rest", "home_rest"
]

ODDS_COLS = [
    "game_id", "away_moneyline", "home_moneyline", "spread_line",
    "away_spread_odds", "home_spread_odds", "total_line", "under_odds", "over_odds"
]

CONDITIONS_COLS = [
    "game_id", "roof", "surface", "temp", "wind", "stadium_id", "stadium"
]

PERSONNEL_COLS = [
    "game_id", "away_qb_id", "home_qb_id", "away_qb_name", "home_qb_name",
    "away_coach", "home_coach", "referee"
]


#Gets connection to sql database

def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)

#Save functions saves the pd.Dataframe passed in to the header into jaginator.db

def save(df: pd.DataFrame, table: str, if_exists: str = "replace") -> None:
    with get_connection() as conn:
        df.to_sql(table, conn, if_exists=if_exists, index=False)

def merge_season_into_historical(schedules_current: pd.DataFrame) -> int:
    completed = schedules_current.dropna(subset=["home_score", "away_score"])

    with get_connection() as conn:
        existing = pd.read_sql("SELECT game_id FROM schedules", conn)["game_id"].tolist()
        new      = completed[~completed["game_id"].isin(existing)]

        if new.empty:
            return 0

        def safe_slice(df, cols):
            return df[[c for c in cols if c in df.columns]]

        new.pipe(safe_slice, SCHEDULE_COLS).to_sql("schedules",         conn, if_exists="append", index=False)
        new.pipe(safe_slice, ODDS_COLS).to_sql("game_odds",             conn, if_exists="append", index=False)
        new.pipe(safe_slice, CONDITIONS_COLS).to_sql("game_conditions", conn, if_exists="append", index=False)
        new.pipe(safe_slice, PERSONNEL_COLS).to_sql("game_personnel",   conn, if_exists="append", index=False)

    return len(new)

#This seeds the database. Only run if something happens to the db
def seed_database() -> None:
    print(f"Seeding jaginator.db at {DB_PATH}")
    data = get_all_historical()

    raw = data["schedules"]
    save(raw[SCHEDULE_COLS],    "schedules")
    save(raw[ODDS_COLS],        "game_odds")
    save(raw[CONDITIONS_COLS],  "game_conditions")
    save(raw[PERSONNEL_COLS],   "game_personnel")
    save(data["team_stats"],    "team_stats")

    print("Database seeded.")

#Updates weekly roster info and weekly stats
def refresh_week(timestamp: pd.Timestamp) -> None:
    with get_connection() as conn:
        save(get_current_roster(timestamp), "rosters", if_exists="replace")
        save(get_current_weekly_stats(timestamp), "weekly_stats", if_exists="replace")

#Load functions. They all load their respective tables. IE load_team_stats the team_stats table

def load_schedules(min_season: int = 1999) -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql(f""" SELECT * FROM schedules WHERE season >= {min_season} ORDER BY season""", conn)

def load_team_stats(min_season: int = 1999) -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql(f"""SELECT * FROM team_stats WHERE season >= {min_season} ORDER BY season""", conn)
    
def load_game_odds(min_season: int = 1999) -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql(f"""
            SELECT game_odds.* FROM game_odds
            JOIN schedules ON game_odds.game_id = schedules.game_id
            WHERE schedules.season >= {min_season}
        """, conn)

def load_game_conditions(min_season: int = 1999) -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql(f"""
            SELECT game_conditions.* FROM game_conditions
            JOIN schedules ON game_conditions.game_id = schedules.game_id
            WHERE schedules.season >= {min_season}
        """, conn)


def load_game_personnel(min_season: int = 1999) -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql(f"""
            SELECT game_personnel.* FROM game_personnel
            JOIN schedules ON game_personnel.game_id = schedules.game_id
            WHERE schedules.season >= {min_season}
        """, conn)

def load_team_games(team: str) -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql(f"""SELECT * FROM schedules WHERE home_team = ? OR away_team = ? ORDER BY season, week""", conn, params={team,team})

def load_elo_history() -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql("SELECT * FROM elo_history ORDER BY season, week", conn)
    
def load_current_elo_ratings() -> dict[str, float]:
   with get_connection() as conn:
        df = pd.read_sql("""
            SELECT h.team, h.elo_after
            FROM elo_history h
            INNER JOIN (
                SELECT team, MAX(season * 100 + week) AS max_key
                FROM elo_history
                GROUP BY team
            ) latest
            ON h.team = latest.team
            AND (h.season * 100 + h.week) = latest.max_key
        """, conn)
        return dict(zip(df["team"], df["elo_after"]))
   
def load_depth_charts(starters_only: bool = True) -> pd.DataFrame:
    with get_connection() as conn:
        query = "SELECT * FROM depth_charts_current"
        if starters_only:
            query += " WHERE depth_order = 1"
        query += " ORDER BY team, position, depth_order"
        return pd.read_sql(query, conn)
    
def load_upcoming_games(schedules_current: pd.DataFrame) -> pd.DataFrame:
    upcoming  = schedules_current[schedules_current["home_score"].isna()]
    if upcoming.empty:
        return upcoming
    next_week = upcoming["week"].min()
    return upcoming[upcoming["week"] ==next_week].reset_index(drop=True)
    
def refresh_current_season() -> None:
    print("Refreshing current season data...")
    save(get_current_weekly_stats(), "weekly_stats", if_exists="replace")
    save(get_current_rosters(),      "rosters",      if_exists="replace")
    print("Current season data refreshed.")



#Utility functions. List_tables() lists the tables the db has, table_info prints all the columns of those tables

def list_tables() -> list[str]:
    with get_connection() as conn:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return [row[0] for row in rows]

def table_info(table: str) -> None:
    with get_connection() as conn:
        current = conn.execute(f"PRAGMA table_info{table}")
        cols = current.fetchall()
        print(f"\n{table} - {len(cols)} columns:")
        for col in cols:
            print(f"{col[1]:30s} {col[2]}")