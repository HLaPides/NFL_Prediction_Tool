import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import math
import pandas as pd
from data.sql_helpers import load_schedules, save

RELOCATIONS = {
    "LA":  "STL",
    "LAC": "SD",
    "LV":  "OAK",
}


class elo_analyzer:
    def __init__(
        self,
        initial_elo:      float = 1500,
        k_factor:         float = 20,
        home_advantage:   float = 48,
        reversion_factor: float = 0.70,
    ):
        self.initial_elo      = initial_elo
        self.k_factor         = k_factor
        self.home_advantage   = home_advantage
        self.reversion_factor = reversion_factor
        self.ratings: dict[str, float] = {}


    def win_probability(self, home_rating: float, away_rating: float) -> float:
        return 1.0 / (1 + math.pow(10, (away_rating - home_rating) / 400))
    

    def update_ratings(self, rating_home, rating_away, home_score, away_score):
        adjusted_home = rating_home + self.home_advantage
        prob_home     = self.win_probability(adjusted_home, rating_away)
        prob_away     = 1.0 - prob_home

        point_diff   = home_score - away_score
        if point_diff > 0:
            outcome_home = 1.0
        elif point_diff < 0:
            outcome_home = 0.0
        else:
            outcome_home = 0.5

        mov = min(math.log(abs(point_diff) + 1), 2.5)
        k   = self.k_factor * mov

        new_home = rating_home + k * (outcome_home - prob_home)
        new_away = rating_away + k * ((1 - outcome_home) - prob_away)
        return new_home, new_away

    def apply_season_reversion(self) -> None:
        self.ratings = {
            team: rating * self.reversion_factor + self.initial_elo * (1 - self.reversion_factor)
            for team, rating in self.ratings.items()
        }

    def _init_team(self, team: str) -> None:
        if team not in self.ratings:
            predecessor = RELOCATIONS.get(team)
            self.ratings[team] = self.ratings.get(predecessor, self.initial_elo)

    #Builds elo history
    def build_history(self, schedules: pd.DataFrame = None) -> tuple[dict, pd.DataFrame]:
        if schedules is None:
            schedules = load_schedules()

        schedules = schedules[schedules["game_type"].isin(["REG", "WC", "DIV", "CON", "SB"])]
        schedules = schedules.sort_values(["season", "week"]).reset_index(drop=True)

        self.ratings   = {}
        history:    list[dict] = []
        current_season = None

        for _, game in schedules.iterrows():
            home   = game["home_team"]
            away   = game["away_team"]
            season = game["season"]

            self._init_team(home)
            self._init_team(away)

            if season != current_season:
                if current_season is not None:
                    self.apply_season_reversion()
                current_season = season

            elo_home_before = self.ratings[home]
            elo_away_before = self.ratings[away]

            new_home, new_away = self.update_ratings(
                self.ratings[home], self.ratings[away],
                int(game["home_score"]), int(game["away_score"])
            )
            self.ratings[home] = new_home
            self.ratings[away] = new_away

            history.append({"game_id": game["game_id"], "season": season,
                             "week": game["week"], "team": home, "opponent": away,
                             "elo_before": round(elo_home_before, 1), "elo_after": round(new_home, 1)})
            history.append({"game_id": game["game_id"], "season": season,
                             "week": game["week"], "team": away, "opponent": home,
                             "elo_before": round(elo_away_before, 1), "elo_after": round(new_away, 1)})

        history_df = pd.DataFrame(history)
        save(history_df, "elo_history")
        print(f"Saved {len(history_df)} rows to elo_history.")

        for old_team in RELOCATIONS.values():
            self.ratings.pop(old_team, None)

        return self.ratings, history_df


    #Predict using current elo ratings
    def predict(self, home_team: str, away_team: str) -> dict:
        r_home = self.ratings.get(home_team, self.initial_elo)
        r_away = self.ratings.get(away_team, self.initial_elo)

        adj_home   = r_home + self.home_advantage
        prob_home  = self.win_probability(adj_home, r_away)
        elo_diff   = adj_home - r_away

        return {
            "home_team":      home_team,
            "away_team":      away_team,
            "home_elo":       round(r_home, 1),
            "away_elo":       round(r_away, 1),
            "prob_home":      round(prob_home, 3),
            "prob_away":      round(1.0 - prob_home, 3),
            "implied_spread": round(elo_diff / 25.0, 1),
        }
if __name__ == "__main__":
    elo = elo_analyzer()
    elo.build_history()