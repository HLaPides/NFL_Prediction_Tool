
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import nflreadpy as nfl
from nflreadpy.utils_date import get_current_season
from data.sql_helpers import (
    save, get_connection,
    merge_season_into_historical,
    load_upcoming_games
)
from data.depth_chart_getter import get_all_depth_charts
from engine.elo import build_elo_history

SKILL_POSITIONS = {"QB", "RB", "WR", "TE"}



def fetch_schedules(season: int) -> pd.DataFrame:
    return nfl.load_schedules([season]).to_pandas()

def fetch_player_stats(season: int) -> pd.DataFrame:
    df = nfl.load_player_stats([season]).to_pandas()
    return df[df["position"].isin(SKILL_POSITIONS)]

def fetch_team_stats(season: int) -> pd.DataFrame:
    return nfl.load_team_stats(seasons=[season], summary_level="week").to_pandas()

def fetch_rosters(season: int) -> pd.DataFrame:
    return nfl.load_rosters([season]).to_pandas()

def fetch_depth_charts() -> pd.DataFrame:
    df = get_all_depth_charts()
    if not df.empty:
        df["fetched_at"] = pd.Timestamp.now().isoformat()
    return df

def run_weekly_update() -> pd.DataFrame:
    """
    Runs the full weekly data refresh pipeline.
    Returns upcoming games ready for prediction.
    """
    season = get_current_season()
    print(f"\nWeekly update — {season} season")
    print("=" * 40)

    print("\n[1/5] Fetching nflverse data...")
    schedules    = fetch_schedules(season)
    player_stats = fetch_player_stats(season)
    team_stats   = fetch_team_stats(season)
    rosters      = fetch_rosters(season)
    print(f"  {len(schedules)} games, {len(player_stats)} player rows, "
          f"{len(team_stats)} team rows, {len(rosters)} roster entries")

    print("\n[2/5] Fetching ESPN depth charts...")
    depth_charts = fetch_depth_charts()
    print(f"  {len(depth_charts)} entries ({depth_charts['team'].nunique()} teams)")

    print("\n[3/5] Saving to current tables...")
    with get_connection() as conn:
        schedules.to_sql("schedules_current",    conn, if_exists="replace", index=False)
        player_stats.to_sql("player_stats_current", conn, if_exists="replace", index=False)
        team_stats.to_sql("team_stats_current",  conn, if_exists="replace", index=False)
        rosters.to_sql("rosters_current",        conn, if_exists="replace", index=False)
        depth_charts.to_sql("depth_charts_current", conn, if_exists="replace", index=False)
    print("  Done")

    print("\n[4/5] Merging completed games into historical tables...")
    n = merge_season_into_historical(schedules)
    print(f"  {n} new games merged")

    print("\n[5/5] Rebuilding ELO...")
    ratings, _ = build_elo_history()
    print(f"  ELO rebuilt for {len(ratings)} teams")

    upcoming = load_upcoming_games(schedules)
    print(f"\nWeekly update complete — {len(upcoming)} upcoming games ready")
    return upcoming


if __name__ == "__main__":
    upcoming = run_weekly_update()
    if not upcoming.empty:
        print("\nUpcoming games:")
        print(upcoming[["week", "home_team", "away_team",
                         "gameday", "gametime"]].to_string(index=False))