"""
elo_analyzer.py — ELO rating system for all NFL teams, 1999–present.

Builds and maintains a complete ELO history for every franchise, handling
team relocations (Raiders OAK→LV, Chargers SD→LAC, Rams STL→LA) by
inheriting the predecessor's rating.

ELO mechanics:
    - 400-point scale win probability formula (standard chess ELO)
    - Home advantage: +48 ELO points added to home team before probability calc
    - K-factor: 20 × margin-of-victory multiplier (log scale, capped at 2.5x)
      so blowouts shift ratings more than squeakers
    - Season reversion: 30% regression toward 1500 between seasons
      (prevents compounding of early-season noise)

Usage:
    elo = elo_analyzer()
    ratings, history_df = elo.build_history()  # builds from DB, saves elo_history table

Writes to DB:
    elo_history — one row per (game_id, team) with elo_before and elo_after
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import math
import pandas as pd
from data.sql_helpers import load_schedules, save

# Teams that relocated — new team inherits predecessor's ELO
RELOCATIONS = {
    "LA":  "STL",   # Rams
    "LAC": "SD",    # Chargers
    "LV":  "OAK",   # Raiders
}


class elo_analyzer:
    def __init__(
        self,
        initial_elo:      float = 1500,
        k_factor:         float = 20,
        home_advantage:   float = 48,
        reversion_factor: float = 0.70,
    ):
        """
        Args:
            initial_elo: Starting ELO for teams with no history (default 1500).
            k_factor: Base K-factor before MOV adjustment (default 20).
            home_advantage: Flat ELO bonus for home team (default 48 ≈ 2.8 pts).
            reversion_factor: Fraction of rating retained between seasons.
                              0.70 = 30% regression toward 1500 each offseason.
        """
        self.initial_elo      = initial_elo
        self.k_factor         = k_factor
        self.home_advantage   = home_advantage
        self.reversion_factor = reversion_factor
        self.ratings: dict[str, float] = {}

    def win_probability(self, home_rating: float, away_rating: float) -> float:
        """
        Compute home team win probability using standard ELO formula.
        home_rating should already include home_advantage adjustment.
        """
        return 1.0 / (1 + math.pow(10, (away_rating - home_rating) / 400))

    def update_ratings(self, rating_home, rating_away, home_score, away_score):
        """
        Update ELO ratings after a completed game.

        K-factor is multiplied by log(|margin| + 1), capped at 2.5x.
        This ensures a 35-point blowout moves ratings more than a 1-point win,
        without letting extreme margins dominate.

        Returns:
            tuple: (new_home_elo, new_away_elo)
        """
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
        """
        Regress all ratings 30% toward 1500 at the start of each new season.
        Prevents multi-year compounding of early-season noise and keeps
        ratings meaningful for teams that have turned over their roster.
        """
        self.ratings = {
            team: rating * self.reversion_factor + self.initial_elo * (1 - self.reversion_factor)
            for team, rating in self.ratings.items()
        }

    def _init_team(self, team: str) -> None:
        """
        Initialize a team's ELO if not already tracked.
        Relocated teams inherit the predecessor's current rating.
        """
        if team not in self.ratings:
            predecessor = RELOCATIONS.get(team)
            self.ratings[team] = self.ratings.get(predecessor, self.initial_elo)

    def build_history(self, schedules: pd.DataFrame = None) -> tuple[dict, pd.DataFrame]:
        """
        Build complete ELO history from all available schedule data.

        Processes games in chronological order. Applies season reversion
        at the start of each new season. Saves results to elo_history table.

        Args:
            schedules: Optional pre-loaded schedules DataFrame.
                       If None, loads from DB via load_schedules().

        Returns:
            tuple: (current_ratings dict, history_df DataFrame)
                   history_df has one row per (game_id, team) with
                   elo_before (used as model feature) and elo_after.
        """
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

        # Remove legacy relocated team entries from current ratings
        for old_team in RELOCATIONS.values():
            self.ratings.pop(old_team, None)

        return self.ratings, history_df

    def predict(self, home_team: str, away_team: str) -> dict:
        """
        Predict win probabilities and implied spread for a matchup.

        Uses current ratings (after build_history() has been called).
        Implied spread = ELO differential / 25 (empirically calibrated).

        Returns:
            dict: home_team, away_team, home_elo, away_elo,
                  prob_home, prob_away, implied_spread
        """
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