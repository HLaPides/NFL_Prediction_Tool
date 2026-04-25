import nflreadpy as nfl
import pandas as pd
from nflreadpy.utils_date import get_current_season

HISTORICAL_SEASONS = list(range(1999, get_current_season() + 1))


def get_schedules(seasons: list = HISTORICAL_SEASONS) -> pd.DataFrame:
    """One row per completed game — scores, spread, home/away."""
    df = nfl.load_schedules(seasons).to_pandas()
    return df.dropna(subset=["home_score", "away_score"])


def get_team_stats(seasons: list = HISTORICAL_SEASONS) -> pd.DataFrame:
    """Season-level team stats — one row per team per season."""
    return nfl.load_team_stats(seasons=seasons, summary_level="reg").to_pandas()


def get_all_historical() -> dict:
    """Loads all historical data needed to seed the DB."""
    print(f"Loading {len(HISTORICAL_SEASONS)} seasons ({HISTORICAL_SEASONS[0]}–{HISTORICAL_SEASONS[-1]})...")
    print("[1/2] Schedules...")
    schedules = get_schedules()
    print(f"  {len(schedules)} completed games")
    print("[2/2] Team stats...")
    team_stats = get_team_stats()
    print(f"  {len(team_stats)} team-season rows")
    return {"schedules": schedules, "team_stats": team_stats}