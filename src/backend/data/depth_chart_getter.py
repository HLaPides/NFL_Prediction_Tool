"""
depth_chart_getter.py — Fetch current NFL depth charts from the ESPN API.

Hits the ESPN site API for each of the 32 NFL teams and returns skill position
depth chart data (QB, RB, WR, TE only). Called by weekly_update.py each week
to refresh depth_charts_current, which feeds into injury penalty calculations
in predict_season.py and predictor.py.

The ESPN depth chart API is unofficial but stable — same endpoint used by
ESPN's own fantasy tools. No auth required. Rate-limited by team loop so
failures on individual teams don't block the rest.

Usage:
    python3.11 data/depth_chart_getter.py   # fetch all 32 teams, print summary

    From code:
        from data.depth_chart_getter import get_all_depth_charts
        df = get_all_depth_charts()   # returns full depth chart DataFrame
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import requests
import pandas as pd

ESPN_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"

TEAM_ID_MAP = {
    "ARI": 22, "ATL": 1,  "BAL": 33, "BUF": 2,
    "CAR": 29, "CHI": 3,  "CIN": 4,  "CLE": 5,
    "DAL": 6,  "DEN": 7,  "DET": 8,  "GB":  9,
    "HOU": 34, "IND": 11, "JAX": 30, "KC":  12,
    "LA":  14, "LAC": 24, "LV":  13, "MIA": 15,
    "MIN": 16, "NE":  17, "NO":  18, "NYG": 19,
    "NYJ": 20, "PHI": 21, "PIT": 23, "SEA": 26,
    "SF":  25, "TB":  27, "TEN": 10, "WSH": 28,
}

SKILL_POSITIONS = {"QB", "RB", "WR", "TE"}


def get_depth_chart(team_abbr: str) -> pd.DataFrame:
    """
    Fetch the full depth chart for a single team from the ESPN API.

    Args:
        team_abbr: NFL team abbreviation (must be a key in TEAM_ID_MAP).

    Returns:
        pd.DataFrame: Columns: team, position, depth_order, player_id, player_name.
                      Only skill positions (QB/RB/WR/TE) are included.
                      Empty DataFrame if the team has no depth chart data.

    Raises:
        ValueError: If team_abbr is not in TEAM_ID_MAP.
        requests.HTTPError: On non-200 API response.
    """
    team_id = TEAM_ID_MAP.get(team_abbr)
    if team_id is None:
        raise ValueError(f"Unknown team: {team_abbr}")

    resp = requests.get(f"{ESPN_URL}/teams/{team_id}/depthcharts", timeout=10)
    resp.raise_for_status()
    raw  = resp.json()
    rows = []

    for group in raw.get("depthchart", []):
        for _, pos_data in group.get("positions", {}).items():
            position = pos_data.get("position", {}).get("abbreviation", "").upper()
            if position not in SKILL_POSITIONS:
                continue
            for depth_order, athlete in enumerate(pos_data.get("athletes", []), start=1):
                rows.append({
                    "team":        team_abbr,
                    "position":    position,
                    "depth_order": depth_order,
                    "player_id":   athlete.get("id"),
                    "player_name": athlete.get("displayName"),
                })

    return pd.DataFrame(rows)


def get_all_depth_charts() -> pd.DataFrame:
    """
    Fetch depth charts for all 32 NFL teams and concatenate into one DataFrame.

    Failures on individual teams are caught and logged — a single team's API
    error won't abort the full fetch. Teams that fail return no rows.

    Returns:
        pd.DataFrame: Full depth chart across all teams with columns:
                      team, position, depth_order, player_id, player_name.
                      Empty DataFrame only if all 32 teams fail.
    """
    frames = []
    for team_abbr in TEAM_ID_MAP:
        try:
            df = get_depth_chart(team_abbr)
            if not df.empty:
                frames.append(df)
                print(f"  {team_abbr} — {len(df)} players")
        except requests.HTTPError as e:
            print(f"  {team_abbr} — HTTP error: {e}")
        except Exception as e:
            print(f"  {team_abbr} — failed: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


if __name__ == "__main__":
    df = get_all_depth_charts()
    print(f"\nStarters only:")
    print(df[df["depth_order"] == 1].groupby("position")["team"].count())