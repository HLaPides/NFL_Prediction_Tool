
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.sql_helpers import load_current_elo_ratings, get_connection
from engine.elo import predict_game
from condition_analyzer import ConditionAnalyzer

import pandas as pd


#Note, these are the weights for weather conditions. They have been zeroed out until I can use a model to accuratly weight them
DOME_ADJ        = 0
COLD_ADJ        = 0
HIGH_WIND_ADJ   = 0
SEVERE_WIND_ADJ = 0
TURF_ADJ        = 0

#The Main engine that compiles all of the features. Currently there is elo and conditions. Missing players and teamForm
class MainEngine:

    def __init__(self):
        self.conditions = ConditionAnalyzer()
        self.ratings    = load_current_elo_ratings()

    def predict(
        self,
        home_team: str,
        away_team: str,
        season:    int,
        week:      int,
        game_id:   str = None,
        # Live mode — pass conditions directly if no game_id
        roof:    str   = None,
        temp:    float = None,
        wind:    float = None,
        surface: str   = None,
    ) -> dict:
        #For historical pass in game id, for live pass in everything else

        elo = predict_game(home_team, away_team, self.ratings)

        if game_id:
            cond = self.conditions.analyze_game(game_id)
        else:
            cond = self.conditions.analyze_upcoming(
                roof=roof, temp=temp, wind=wind, surface=surface
            )

        adj = 0.0
        if cond["high_wind"]:
            adj += HIGH_WIND_ADJ
        if cond["severe_wind"]:
            adj += SEVERE_WIND_ADJ
        if cond["is_cold"]:
            adj += COLD_ADJ
        if cond["is_turf"]:
            adj += TURF_ADJ

        adjusted_prob = round(max(0.01, min(0.99, elo["prob_home"] + adj)), 3)

        return {
            "home_team":      home_team,
            "away_team":      away_team,
            "elo_prob":       elo["prob_home"],
            "adjusted_prob":  adjusted_prob,
            "implied_spread": elo["implied_spread"],
            "conditions":     cond,
            "personnel":      {},   # PersonnelAnalyzer — coming soon
            "form":           {},   # TeamFormAnalyzer  — coming soon
        }

#Generates all predictions in given week, uses game_ifs from DB to look it up
    def predict_week(self, season: int, week: int) -> list[dict]:
        with get_connection() as conn:
            games = pd.read_sql(f"""
                SELECT s.game_id, s.home_team, s.away_team
                FROM schedules s
                WHERE s.season = {season} AND s.week = {week}
                AND s.game_type = 'REG'
            """, conn)

        return [
            self.predict(
                home_team = row["home_team"],
                away_team = row["away_team"],
                season    = season,
                week      = week,
                game_id   = row["game_id"],
            )
            for _, row in games.iterrows()
        ]



if __name__ == "__main__":
    engine = MainEngine()

    print("Single game prediction — KC vs BUF (live, cold + windy):")
    pred = engine.predict(
        home_team = "KC",
        away_team = "BUF",
        season    = 2025,
        week      = 10,
        roof      = "outdoors",
        temp      = 32.0,
        wind      = 18.0,
        surface   = "grass",
    )
    print(f"  ELO prob:      {pred['elo_prob']}")
    print(f"  Adjusted prob: {pred['adjusted_prob']}")
    print(f"  Implied spread:{pred['implied_spread']}")
    print(f"  Conditions:    {pred['conditions']}")

    print("\nWeek 1 2024 predictions:")
    week1 = engine.predict_week(season=2024, week=1)
    for p in week1:
        print(f"  {p['home_team']:4s} vs {p['away_team']:4s} — "
              f"ELO: {p['elo_prob']:.3f}  Adj: {p['adjusted_prob']:.3f}  "
              f"Spread: {p['implied_spread']:+.1f}")