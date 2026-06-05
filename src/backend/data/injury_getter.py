"""
injury_getter.py — Fetch current NFL injury report from the ESPN API.

Pulls the live injury report for all 32 teams and returns a flat DataFrame
saved to the injuries_current table. Called by weekly_update.py each week.

The ESPN injuries API returns all players with an injury designation —
Out, Doubtful, Questionable, IR, PUP, Suspended, Probable. The resulting
table is used by:
    - predictor.py (build_injury_scores): compute weighted injury severity
      per team for the current week's game predictions
    - predict_season.py (build_injury_penalties): apply ELO penalties to
      teams with confirmed starter absences in the Super Bowl simulator

Usage:
    python3.11 data/injury_getter.py   # fetch, save to DB, print summary

    From code:
        from data.injury_getter import fetch_injuries
        injuries = fetch_injuries()    # returns DataFrame, does NOT save to DB
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import requests
import pandas as pd
from data.sql_helpers import save

ESPN_INJURIES_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries"

UNLIKELY_STATUSES  = {"Out", "Doubtful", "IR", "PUP", "Suspended"}
UNCERTAIN_STATUSES = {"Questionable"}


def fetch_injuries() -> pd.DataFrame:
    """
    Fetch the current NFL injury report from the ESPN API.

    Returns one row per injured player with their team (display name format,
    e.g. 'Kansas City Chiefs'), position, injury status, and short comment.

    The team_display column uses full team names — callers are responsible
    for mapping to abbreviations via TEAM_DISPLAY_NAMES if needed (see
    predictor.py and predict_season.py for the mapping dict).

    Returns:
        pd.DataFrame: Columns: team_display, player_name, position,
                      status, comment, date.
                      Empty DataFrame if the API returns no injury data.

    Raises:
        requests.HTTPError: On non-200 API response.
    """
    resp = requests.get(ESPN_INJURIES_URL, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    rows = []
    for team in data.get("injuries", []):
        team_name = team.get("displayName", "")
        for injury in team.get("injuries", []):
            athlete  = injury.get("athlete", {})
            position = athlete.get("position", {})
            rows.append({
                "team_display": team_name,
                "player_name":  athlete.get("displayName"),
                "position":     position.get("abbreviation") if isinstance(position, dict) else None,
                "status":       injury.get("status"),
                "comment":      injury.get("shortComment", ""),
                "date":         injury.get("date", ""),
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Fetching NFL injury report...")
    df = fetch_injuries()
    save(df, "injuries_current")
    print(f"Saved {len(df)} injury entries")

    print("\nStatus breakdown:")
    print(df["status"].value_counts().to_string())

    print("\nSample Out/Doubtful players:")
    print(df[df["status"].isin(UNLIKELY_STATUSES)][
        ["team_display", "player_name", "position", "status", "comment"]
    ].head(10).to_string(index=False))