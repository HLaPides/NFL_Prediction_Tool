"""
getter.py — Historical data loader for initial DB seeding.

Wraps nflreadpy to load schedules and team stats for all seasons from 1999
to present. Only used by sql_helpers.seed_database() — not part of the weekly
update pipeline (weekly_update.py fetches current season data directly).

Run seed_database() from sql_helpers.py if the DB is lost or corrupted:
    python3.11 -c "from data.sql_helpers import seed_database; seed_database()"
"""
import nflreadpy as nfl
import pandas as pd
from nflreadpy.utils_date import get_current_season

HISTORICAL_SEASONS = list(range(1999, get_current_season() + 1))


def get_schedules(seasons: list = HISTORICAL_SEASONS) -> pd.DataFrame:
    """
    Load completed game schedules for the given seasons.

    Filters to games with both scores present — excludes upcoming games.
    Columns include game_id, season, week, home/away team, scores, spread,
    result, div_game, rest days, and game metadata.

    Returns:
        pd.DataFrame: One row per completed game.
    """
    df = nfl.load_schedules(seasons).to_pandas()
    return df.dropna(subset=["home_score", "away_score"])


def get_team_stats(seasons: list = HISTORICAL_SEASONS) -> pd.DataFrame:
    """
    Load season-level team stats for the given seasons.

    Note: this loads regular season aggregates (summary_level='reg') —
    different from what weekly_update.py loads (summary_level='week').
    These are full-season totals used for seeding team_stats, not the
    week-by-week data used in model training.

    Returns:
        pd.DataFrame: One row per team per season.
    """
    return nfl.load_team_stats(seasons=seasons, summary_level="reg").to_pandas()


def get_all_historical() -> dict:
    """
    Load all historical data needed to seed the DB from scratch.

    Returns:
        dict: {
            "schedules":  DataFrame of all completed games 1999–present,
            "team_stats": DataFrame of season-level team stats 1999–present
        }
    """
    print(f"Loading {len(HISTORICAL_SEASONS)} seasons ({HISTORICAL_SEASONS[0]}–{HISTORICAL_SEASONS[-1]})...")
    print("[1/2] Schedules...")
    schedules = get_schedules()
    print(f"  {len(schedules)} completed games")
    print("[2/2] Team stats...")
    team_stats = get_team_stats()
    print(f"  {len(team_stats)} team-season rows")
    return {"schedules": schedules, "team_stats": team_stats}