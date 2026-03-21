import sqlite3
import pandas as pd
import time
from pathlib import Path
from getter import get_all_historical, get_current_roster, get_current_weekly_stats

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


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)

def save(df: pd.DataFrame, table: str, if_exists: str = "replace") -> None:
    with get_connection() as conn:
        df.to_sql(table, conn, if_exists=if_exists, index=False)

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

def refresh_week() -> None:
    with get_connection() as conn:
        save(get_current_roster(time.time), "rosters", if_exists="replace")
        save(get_current_weekly_stats(time.time), "weekly_stats", if_exists="replace")


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

if __name__ == "__main__":
    seed_database()

    print("\nTables in DB:", list_tables())
    print(f"Schedules since 2020:       {len(load_schedules(2020))} games")
    print(f"Team stats since 2020:      {len(load_team_stats(2020))} rows")
    print(f"Game odds since 2020:       {len(load_game_odds(2020))} rows")
    print(f"Game conditions since 2020: {len(load_game_conditions(2020))} rows")
    print(f"Game personnel since 2020:  {len(load_game_personnel(2020))} rows")