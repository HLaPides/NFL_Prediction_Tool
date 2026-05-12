import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import nflreadpy as nfl
from nflreadpy.utils_date import get_current_season
from data.sql_helpers import save, get_connection, merge_season_into_historical, load_upcoming_games
from data.depth_chart_getter import get_all_depth_charts
from data.injury_getter import fetch_injuries
from engine.elo_analyzer import elo_analyzer
from engine.model import load_training_data, train_final_model
from engine.defense import build_defense_ratings
from engine.predictor import run_predictions
from scripts.seed import build_rolling_stats, build_player_features

SKILL_POSITIONS = {"QB", "RB", "WR", "TE"}


# ── Data fetchers ─────────────────────────────────────────────────────────────
def get_schedules(season: int) -> pd.DataFrame:
    return nfl.load_schedules([season]).to_pandas()

def get_player_stats(season: int) -> pd.DataFrame:
    df = nfl.load_player_stats([season]).to_pandas()
    df = df[df["season_type"] == "REG"]
    return df[df["position"].isin(SKILL_POSITIONS)]

def get_team_stats(season: int) -> pd.DataFrame:
    return nfl.load_team_stats(seasons=[season], summary_level="week").to_pandas()

def get_rosters(season: int) -> pd.DataFrame:
    return nfl.load_rosters([season]).to_pandas()

def get_depth_charts() -> pd.DataFrame:
    df = get_all_depth_charts()
    if not df.empty:
        df["fetched_at"] = pd.Timestamp.now().isoformat()
    return df


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run_weekly_update() -> pd.DataFrame:
    season = get_current_season()
    print(f"\nWeekly update — {season} season")
    print("=" * 40)

    print("\n[1/9] Fetching nflverse data...")
    schedules    = get_schedules(season)
    player_stats = get_player_stats(season)
    team_stats   = get_team_stats(season)
    rosters      = get_rosters(season)
    print(f"  {len(schedules)} games, {len(player_stats)} player rows, "
          f"{len(team_stats)} team rows, {len(rosters)} roster entries")

    print("\n[2/9] Fetching ESPN depth charts...")
    depth_charts = get_depth_charts()
    print(f"  {len(depth_charts)} entries ({depth_charts['team'].nunique()} teams)")

    print("\n[3/9] Fetching injury report...")
    injuries = fetch_injuries()
    print(f"  {len(injuries)} injury entries")

    print("\n[4/9] Saving current tables...")
    with get_connection() as conn:
        schedules.to_sql("schedules_current",       conn, if_exists="replace", index=False)
        player_stats.to_sql("player_stats_current", conn, if_exists="replace", index=False)
        team_stats.to_sql("team_stats_current",     conn, if_exists="replace", index=False)
        rosters.to_sql("rosters_current",           conn, if_exists="replace", index=False)
        depth_charts.to_sql("depth_charts_current", conn, if_exists="replace", index=False)
        injuries.to_sql("injuries_current",         conn, if_exists="replace", index=False)
    print("  Done")

    print("\n[5/9] Merging completed games into historical tables...")
    n = merge_season_into_historical(schedules)
    print(f"  {n} new games merged")

    print("\n[6/9] Rebuilding ELO + features...")
    elo = elo_analyzer()
    ratings, _ = elo.build_history()
    print(f"  ELO rebuilt for {len(ratings)} teams")

    with get_connection() as conn:
        existing_rolling = pd.read_sql(
            "SELECT DISTINCT game_id FROM team_rolling_stats", conn
        )["game_id"].tolist()
        existing_player = pd.read_sql(
            "SELECT DISTINCT game_id FROM player_features", conn
        )["game_id"].tolist()

    new_rolling = [g for g in schedules["game_id"] if g not in existing_rolling]
    new_player  = [g for g in schedules["game_id"] if g not in existing_player]

    if new_rolling:
        rolling = build_rolling_stats()
        save(rolling, "team_rolling_stats")
        print(f"  Rolling stats rebuilt: {len(rolling)} rows")
    else:
        print("  Rolling stats up to date")

    if new_player:
        player_feats = build_player_features()
        save(player_feats, "player_features")
        print(f"  Player features rebuilt: {len(player_feats)} rows")
    else:
        print("  Player features up to date")

    print("\n[7/9] Retraining spread model...")
    df = load_training_data()
    train_final_model(df)
    print(f"  Model retrained on {len(df)} games")

    print("\n[8/9] Rebuilding defense ratings...")
    def_ratings = build_defense_ratings()
    save(def_ratings, "defense_ratings")
    print(f"  Defense ratings updated: {len(def_ratings):,} rows")

    print("\n[9/9] Running predictions (picks + props)...")
    run_predictions()

    upcoming = load_upcoming_games(schedules)
    print(f"\nWeekly update complete — {len(upcoming)} upcoming games ready")
    return upcoming


if __name__ == "__main__":
    upcoming = run_weekly_update()
    if not upcoming.empty:
        print("\nUpcoming games:")
        print(upcoming[["week", "home_team", "away_team",
                         "gameday", "gametime"]].to_string(index=False))