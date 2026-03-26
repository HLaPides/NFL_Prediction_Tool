import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import requests
import pandas as pd

ESPN_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"

# Mapping from nflverse abbreviations to ESPN numeric team IDs
TEAM_ID_MAP = {
    "ARI": 22, "ATL": 1,  "BAL": 33, "BUF": 2,
    "CAR": 29, "CHI": 3,  "CIN": 4,  "CLE": 5,
    "DAL": 6,  "DEN": 7,  "DET": 8,  "GB":  9,
    "HOU": 34, "IND": 11, "JAX": 30, "KC":  12,
    "LA":  14, "LAC": 24, "LV":  13, "MIA": 15,
    "MIN": 16, "NE":  17, "NO":  18, "NYG": 19,
    "NYJ": 20, "PHI": 21, "PIT": 23, "SEA": 26,
    "SF":  25, "TB":  27, "TEN": 10, "WSH": 28
}

SKILL_POSITIONS = {"QB", "RB", "WR", "TE"}

def get_depth_chart_raw(team_abr: str) -> dict:
    team_id = TEAM_ID_MAP.get(team_abr)
    if (team_id is None):
        raise ValueError(f"Unknown team abbreviation: {team_abr}")
    url = f"{ESPN_URL}/teams/{team_id}/depthcharts"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()

def parse_depth_chart(raw: dict, team_abbr: str) -> pd.DataFrame:
    rows = []

    for group in raw.get("depthchart", []):
        positions = group.get("positions", {})

        for pos_key, pos_data in positions.items():
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


def get_depth_chart(team_abbr: str) -> pd.DataFrame:
    raw = get_depth_chart_raw(team_abbr)
    return parse_depth_chart(raw, team_abbr)


def get_starters(team_abbr: str) -> pd.DataFrame:
    df = get_depth_chart(team_abbr)
    if df.empty:
        return df
    return df[df["depth_order"]==1].reset_index(drop=True)


def get_all_depth_charts() -> pd.DataFrame:
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

def get_week_starters(home_team: str, away_team: str) -> dict[str, pd.DataFrame]:
    return {
        "home": get_starters(home_team),
        "away": get_starters(away_team),
    }

if __name__ == "__main__":
    df = get_all_depth_charts()
    print(f"\nStarters only:")
    starters = df[df["depth_order"] == 1]
    print(starters.groupby("position")["team"].count())
