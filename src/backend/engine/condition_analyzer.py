import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from data.sql_helpers import get_connection

DOME_TEMP = 72
DOME_WIND = 0
COLD_THRESHOLD = 40
HIGH_WIND = 15
SEVERE_WIND = 25

DOME_ROOFS = {"dome", "closed", "retractable"}

#Analyzes wind, temperature, turf surface, and roof for a given game
class ConditionAnalyzer:

    def __init__(self):
        with get_connection() as conn:
            self._conditions = pd.read_sql("SELECT * FROM game_conditions", conn)

    def _compute(
        self,
        roof:    str,
        temp:    float,
        wind:    float,
        surface: str,
    ) -> dict:
        is_dome = str(roof).lower() in DOME_ROOFS if roof else False

        temp = DOME_TEMP if is_dome else (temp if temp is not None else DOME_TEMP)
        wind = DOME_WIND if is_dome else (wind if wind is not None else DOME_WIND)

        return {
            "is_dome":     int(is_dome),
            "temp":        round(temp, 1),
            "is_cold":     int(temp < COLD_THRESHOLD),
            "wind":        round(wind, 1),
            "high_wind":   int(wind >= HIGH_WIND),
            "severe_wind": int(wind >= SEVERE_WIND),
            "is_turf":     int(str(surface).lower() in {"astroturf", "fieldturf",
                                                         "a_turf", "matrixturf",
                                                         "sportturf", "artificalturf"})
        }

    #Analyzes historical game
    def analyze_game(self, game_id: str) -> dict:
        row = self._conditions[self._conditions["game_id"] == game_id]
        if row.empty:
            raise ValueError("CONDITION_ANALZYER ISSUE: unknown game id")
        r = row.iloc[0]
        return self._compute(
            roof = r.get("roof"),
            temp = r.get("temp"),
            wind = r.get("wind"),
            surface = r.get("surface")
        )
    
    #Analyzes upcoming game with info available
    def analyze_upcoming(
        self,
        roof:    str   = None,
        temp:    float = None,
        wind:    float = None,
        surface: str   = None,
    ) -> dict:
        return self._compute(roof=roof, temp=temp, wind=wind, surface=surface)

if __name__ == "__main__":
    analyzer = ConditionAnalyzer()

    print("Sample historical game conditions:\n")
    with get_connection() as conn:
        sample_ids = pd.read_sql("""
            SELECT game_id, roof, temp, wind, surface
            FROM game_conditions
            LIMIT 5
        """, conn)

    for _, row in sample_ids.iterrows():
        result = analyzer.analyze_game(row["game_id"])
        print(f"  {row['game_id']}")
        print(f"    roof={row['roof']} temp={row['temp']} wind={row['wind']} surface={row['surface']}")
        print(f"    → {result}\n")

    print("Live upcoming game (outdoor, cold, windy):")
    result = analyzer.analyze_upcoming("outdoors", 28.0, 22.0, "grass")
    print(f"  → {result}")

    print("\nLive upcoming game (dome):")
    result = analyzer.analyze_upcoming(roof="dome")
    print(f"  → {result}")