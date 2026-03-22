import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import nflreadpy as nfl
import requests
import pandas as pd

#This is as far as nflread supports
HISTORICAL_SEASONS = list(range(1999,2025))

#One row per game. Includes scores, spread, home/away
def get_schedules(Seasons: list[int] = HISTORICAL_SEASONS) -> pd.DataFrame:
    df = nfl.load_schedules(Seasons).to_pandas()
    df = df.dropna(subset=["home_score", "away_score"])
    return df

#gets historical data from 99' to present. 1 row per season per team
def get_team_stats(Seasons : list[int] = HISTORICAL_SEASONS) -> pd.DataFrame:
    return nfl.load_team_stats(seasons=Seasons, summary_level = "reg").to_pandas()

#Week by week player stats for current season
def get_current_weekly_stats(Timestamp) -> pd.DataFrame:
    current_season = Timestamp.year
    return nfl.load_player_stats([current_season]).to_pandas()

#This gets a current roster (IE who is active, injured, benched)
def get_current_roster(Timestamp) -> pd.DataFrame:
    current_season = Timestamp.year
    return nfl.load_rosters([current_season]).to_pandas()


#Prints all the historical data
def get_all_historical() -> dict [str, pd.DataFrame]:
    print(f"Loading {len(HISTORICAL_SEASONS)} seasons of NFL data...")
    print("[1/2] Schedules and Game Results....")
    schedules = get_schedules()
    print(f"{len(schedules)} completed games")
    print(" [2/2] Team stats...")
    team_stats = get_team_stats()
    print(f"{len(team_stats)} team per season rows")

    return {
        "schedules" : schedules,
        "team_stats" : team_stats
    }

