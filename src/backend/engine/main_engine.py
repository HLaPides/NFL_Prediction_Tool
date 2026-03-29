import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from data.sql_helpers import load_current_elo_ratings, get_connection
from engine.elo_analyzer import elo_analyzer
from engine.condition_analyzer import condition_analyzer
from engine.players.player_analyzer import player_analyzer

# Condition weights — zeroed out until Bayesian model learns proper values
DOME_ADJ        = 0.0
COLD_ADJ        = 0.0
HIGH_WIND_ADJ   = 0.0
SEVERE_WIND_ADJ = 0.0
TURF_ADJ        = 0.0


class MainEngine:
    """
    Orchestrates all prediction components into a single adjusted probability.

    Current components:
        - EloEngine       (base probability)
        - ConditionsAnalyzer (weather + surface features)

    Planned:
        - PersonnelAnalyzer
        - TeamFormAnalyzer
    """

    def __init__(self, elo: elo_analyzer = None, conditions: condition_analyzer = None,
                 players: player_analyzer = None):
        self.elo        = elo        or elo_analyzer()
        self.conditions = conditions or condition_analyzer()
        self.players    = players    or player_analyzer()
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
        """
        Historical mode — pass game_id to look up conditions from DB.
        Live mode       — pass roof, temp, wind, surface directly.
        """
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

        personnel = self.players.evaluate_matchup(home_team, away_team, week, season)

        return {
            "home_team":      home_team,
            "away_team":      away_team,
            "elo_prob":       pred["prob_home"],
            "adjusted_prob":  round(max(0.01, min(0.99, pred["prob_home"] + adj)), 3),
            "implied_spread": pred["implied_spread"],
            "conditions":     cond,
            "personnel":      personnel,
            "form":           {},
        }

    def predict_week(self, season: int, week: int) -> list[dict]:
        """Generates predictions for all games in a given week."""
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
    for side in ["home", "away"]:
        p = pred["personnel"][side]
        print(f"\n  {p['team']} offense:")
        print(f"    QB  {p['offense']['qb'].get('player_name','?'):20s}  EPA/att: {p['offense']['qb']['epa_per_att']:+.4f}")
        print(f"    WR1 {p['offense']['wr'].get('player_name','?'):20s}  Yards: {p['offense']['wr']['receiving_yards']:.1f}")
        print(f"    TE1 {p['offense']['te'].get('player_name','?'):20s}  Yards: {p['offense']['te']['receiving_yards']:.1f}")
        print(f"    RB1 {p['offense']['rb'].get('player_name','?'):20s}  Rush: {p['offense']['rb']['rush_yards']:.1f}")
        print(f"  {p['team']} defense:")
        print(f"    Pts: {p['defense']['pts_allowed']:.1f}  Pass: {p['defense']['pass_yards_allowed']:.1f}  Rush: {p['defense']['rush_yards_allowed']:.1f}")

    print("\nWeek 1 2024 predictions:")
    for p in engine.predict_week(season=2024, week=1):
        print(f"  {p['home_team']:4s} vs {p['away_team']:4s} — "
              f"ELO: {p['elo_prob']:.3f}  Adj: {p['adjusted_prob']:.3f}  "
              f"Spread: {p['implied_spread']:+.1f}")