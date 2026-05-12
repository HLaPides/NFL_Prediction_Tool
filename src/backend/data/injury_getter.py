\import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import requests
import pandas as pd
from data.sql_helpers import save

ESPN_INJURIES_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries"

UNLIKELY_STATUSES  = {"Out", "Doubtful", "IR", "PUP", "Suspended"}
UNCERTAIN_STATUSES = {"Questionable"}


def fetch_injuries() -> pd.DataFrame:
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


def get_team_injury_flags(injuries: pd.DataFrame, team_display: str) -> dict:
    team_inj = injuries[injuries["team_display"] == team_display]
    return {
        "has_out":              any(team_inj["status"].isin(UNLIKELY_STATUSES)),
        "has_questionable":     any(team_inj["status"].isin(UNCERTAIN_STATUSES)),
        "out_players":          team_inj[team_inj["status"].isin(UNLIKELY_STATUSES)]["player_name"].tolist(),
        "questionable_players": team_inj[team_inj["status"].isin(UNCERTAIN_STATUSES)]["player_name"].tolist(),
    }


def get_qb_status(injuries: pd.DataFrame, qb_name: str) -> str:
    last_name = qb_name.split(".")[-1].strip() if "." in qb_name else qb_name
    match = injuries[injuries["player_name"].str.contains(last_name, case=False, na=False)]
    if match.empty:
        return "Active"
    return match.iloc[0]["status"]


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