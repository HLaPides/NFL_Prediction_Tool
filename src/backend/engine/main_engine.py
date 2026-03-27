import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from data.sql_helpers import load_current_elo_ratings, get_connection
from engine.elo_analyzer import elo_analyzer
from engine.condition_analyzer import ConditionAnalyzer

# Condition weights — zeroed out until Bayesian model learns proper values
DOME_ADJ        = 0.0
COLD_ADJ        = 0.0
HIGH_WIND_ADJ   = 0.0
SEVERE_WIND_ADJ = 0.0
TURF_ADJ        = 0.0


class MainEngine:
#Missing teamform and player analytics

    def __init__(self, elo: elo_analyzer = None, conditions: ConditionAnalyzer = None):
        self.elo        = elo or elo_analyzer()
        self.conditions = conditions or ConditionAnalyzer()
        self.elo.ratings = load_current_elo_ratings()

    def predict(
        self,
        home_team: str,
        away_team: str,
        season:    int,
        week:      int,
        game_id:   str   = None,
        roof:      str   = None,
        temp:      float = None,
        wind:      float = None,
        surface:   str   = None,
    ) -> dict:
        #Pass game id for historical game, everything else for live
        pred = self.elo.predict(home_team, away_team)

        cond = (self.conditions.analyze_game(game_id) if game_id
                else self.conditions.analyze_upcoming(roof=roof, temp=temp,
                                                      wind=wind, surface=surface))

        adj = 0.0
        if cond["is_dome"]:     adj += DOME_ADJ
        if cond["is_cold"]:     adj += COLD_ADJ
        if cond["severe_wind"]: adj += SEVERE_WIND_ADJ
        elif cond["high_wind"]: adj += HIGH_WIND_ADJ
        if cond["is_turf"]:     adj += TURF_ADJ

        return {
            "home_team":      home_team,
            "away_team":      away_team,
            "elo_prob":       pred["prob_home"],
            "adjusted_prob":  round(max(0.01, min(0.99, pred["prob_home"] + adj)), 3),
            "implied_spread": pred["implied_spread"],
            "conditions":     cond,
            "personnel":      {},
            "form":           {},
        }

    #Predicts all games in a week
    def predict_week(self, season: int, week: int) -> list[dict]:
        with get_connection() as conn:
            games = pd.read_sql(f"""
                SELECT game_id, home_team, away_team
                FROM schedules
                WHERE season = {season} AND week = {week}
                AND game_type = 'REG'
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
    for p in engine.predict_week(season=2024, week=1):
        print(f"  {p['home_team']:4s} vs {p['away_team']:4s} — "
              f"ELO: {p['elo_prob']:.3f}  Adj: {p['adjusted_prob']:.3f}  "
              f"Spread: {p['implied_spread']:+.1f}")