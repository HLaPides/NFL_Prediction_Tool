import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from data.sql_helpers import get_connection, save

# Mock QB prop predictions for testing the UI
mock_qbs = [
    {"player_name": "P.Mahomes",   "team": "KC",  "opponent": "BUF", "week": 15, "is_home": 1,
     "pred_passing_yards": 287.3, "pred_passing_tds": 2.1, "pred_attempts": 34.2, "implied_team_total": 27.5, "spread_line": -3.5},
    {"player_name": "J.Allen",     "team": "BUF", "opponent": "KC",  "week": 15, "is_home": 0,
     "pred_passing_yards": 264.8, "pred_passing_tds": 1.9, "pred_attempts": 36.1, "implied_team_total": 24.5, "spread_line":  3.5},
    {"player_name": "L.Jackson",   "team": "BAL", "opponent": "PIT", "week": 15, "is_home": 1,
     "pred_passing_yards": 231.4, "pred_passing_tds": 1.7, "pred_attempts": 28.6, "implied_team_total": 26.0, "spread_line": -7.0},
    {"player_name": "J.Burrow",    "team": "CIN", "opponent": "CLE", "week": 15, "is_home": 1,
     "pred_passing_yards": 278.9, "pred_passing_tds": 2.2, "pred_attempts": 37.4, "implied_team_total": 25.5, "spread_line": -5.5},
    {"player_name": "J.Goff",      "team": "DET", "opponent": "MIN", "week": 15, "is_home": 0,
     "pred_passing_yards": 252.1, "pred_passing_tds": 1.8, "pred_attempts": 33.8, "implied_team_total": 23.0, "spread_line":  2.5},
    {"player_name": "J.Hurts",     "team": "PHI", "opponent": "DAL", "week": 15, "is_home": 1,
     "pred_passing_yards": 219.6, "pred_passing_tds": 1.6, "pred_attempts": 29.2, "implied_team_total": 27.0, "spread_line": -6.0},
    {"player_name": "B.Purdy",     "team": "SF",  "opponent": "LAR", "week": 15, "is_home": 1,
     "pred_passing_yards": 244.3, "pred_passing_tds": 1.9, "pred_attempts": 31.5, "implied_team_total": 24.5, "spread_line": -4.5},
    {"player_name": "C.Stroud",    "team": "HOU", "opponent": "TEN", "week": 15, "is_home": 1,
     "pred_passing_yards": 261.7, "pred_passing_tds": 1.8, "pred_attempts": 35.0, "implied_team_total": 25.0, "spread_line": -8.5},
    {"player_name": "D.Prescott",  "team": "DAL", "opponent": "PHI", "week": 15, "is_home": 0,
     "pred_passing_yards": 247.2, "pred_passing_tds": 1.7, "pred_attempts": 34.6, "implied_team_total": 21.0, "spread_line":  6.0},
    {"player_name": "J.Love",      "team": "GB",  "opponent": "SEA", "week": 15, "is_home": 1,
     "pred_passing_yards": 256.4, "pred_passing_tds": 1.9, "pred_attempts": 33.1, "implied_team_total": 24.0, "spread_line": -3.0},
    {"player_name": "T.Lawrence",  "team": "JAX", "opponent": "IND", "week": 15, "is_home": 0,
     "pred_passing_yards": 238.9, "pred_passing_tds": 1.5, "pred_attempts": 32.4, "implied_team_total": 20.5, "spread_line":  1.5},
    {"player_name": "S.Darnold",   "team": "MIN", "opponent": "DET", "week": 15, "is_home": 1,
     "pred_passing_yards": 243.6, "pred_passing_tds": 1.7, "pred_attempts": 30.9, "implied_team_total": 22.5, "spread_line": -2.5},
    {"player_name": "A.Richardson","team": "IND", "opponent": "JAX", "week": 15, "is_home": 1,
     "pred_passing_yards": 198.4, "pred_passing_tds": 1.4, "pred_attempts": 26.8, "implied_team_total": 22.0, "spread_line": -1.5},
    {"player_name": "D.Carr",      "team": "NO",  "opponent": "ATL", "week": 15, "is_home": 0,
     "pred_passing_yards": 229.1, "pred_passing_tds": 1.5, "pred_attempts": 31.7, "implied_team_total": 19.5, "spread_line":  3.0},
    {"player_name": "R.Wilson",    "team": "PIT", "opponent": "BAL", "week": 15, "is_home": 0,
     "pred_passing_yards": 213.7, "pred_passing_tds": 1.4, "pred_attempts": 28.3, "implied_team_total": 18.5, "spread_line":  7.0},
    {"player_name": "G.Smith",     "team": "SEA", "opponent": "GB",  "week": 15, "is_home": 0,
     "pred_passing_yards": 234.2, "pred_passing_tds": 1.6, "pred_attempts": 32.5, "implied_team_total": 21.5, "spread_line":  3.0},
]

df = pd.DataFrame(mock_qbs)
df["game_id"] = df.apply(lambda r: f"2025_15_{r['team']}_{r['opponent']}", axis=1)

with get_connection() as conn:
    df.to_sql("qb_prop_predictions", conn, if_exists="replace", index=False)

print(f"Inserted {len(df)} mock QB prop predictions")
print(df[["player_name", "team", "opponent", "pred_passing_yards", "pred_passing_tds"]].to_string(index=False))