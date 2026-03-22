import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import math
import pandas as pd
from data.sql_helpers import load_schedules, save

INITIAL_ELO = 1500 #How much elo each team starts at
K_FACTOR = 20 #How much each match impacts elo. For reference chess has a k factor 30
HOME_ADVANTAGE = 48 #Home teams a bonus to elo when playing at home to represent homefield advantage
REVERSION_FACTOR = .80 #After each season a team loses 20% of their elo. Idea is to reflect preseason changes

RELOCATIONS = {
    "LA":  "STL",  # Rams:     STL -> LA
    "LAC": "SD",   # Chargers:  SD -> LAC
    "LV":  "OAK",  # Raiders:  OAK -> LV 
}

#Calculates win probablitity based on elo
def calculate_win_probability(home_rating: float, away_rating: float):
    return 1.0 / (1 + math.pow(10,(away_rating - home_rating)/400))

#Multiples elo by point_difference. A blowout is going to affect elo more than a 
def margin_multiplier(point_difference: int):
    return min(math.log(abs(point_difference) + 1), 2.5)

#Updates elo rating. Note the home team gets a temp HOME_ADVANTAGE elo boost
def update_rating(rating_home: float, rating_away: float, home_score: int, away_score: int) -> tuple[float, float]:
    adjusted_home = rating_home + HOME_ADVANTAGE

    prob_home = calculate_win_probability(adjusted_home, rating_away)
    prob_away = 1 - prob_home

    point_diff = home_score - away_score

    if (point_diff>0):
        outcome_home = 1.0
    elif(point_diff<0):
        outcome_home = 0.0
    else:
        outcome_home = 0.5
    
    point_diff_multiplier = margin_multiplier(outcome_home)
    k = K_FACTOR * point_diff_multiplier

    new_home = rating_home + K_FACTOR * (outcome_home-prob_home)
    new_away = rating_away + K_FACTOR * ((1-outcome_home) - prob_away)

    return new_home, new_away

def apply_season_regression(ratings: dict[str, float]) -> dict[str, float]:
    return {
        team: rating * REVERSION_FACTOR + INITIAL_ELO * (1 - REVERSION_FACTOR)
        for team, rating in ratings.items()
    }


def build_elo_history(schedules: pd.DataFrame = None) -> tuple[dict, pd.DataFrame]:
    """
    Replays every NFL game from 1999 to present in chronological order,
    updating ELO ratings after each game.

    Returns:
        ratings  : dict of current ELO ratings {team: rating}
        history  : DataFrame of every team's rating after every game
                   columns: game_id, season, week, team, elo_before, elo_after
    """
    if schedules is None:
        schedules = load_schedules()

    # Only use regular season + playoffs, sorted chronologically
    schedules = schedules[schedules["game_type"].isin(["REG", "POST", "WC", "DIV", "CON", "SB"])]
    schedules = schedules.sort_values(["season", "week"]).reset_index(drop=True)

    ratings: dict[str, float] = {}
    history: list[dict] = []
    current_season = None

    for _, game in schedules.iterrows():
        home = game["home_team"]
        away = game["away_team"]
        season = game["season"]

        # Initialize any new team we haven't seen before
        if home not in ratings:
            ratings[home] = ratings.get(RELOCATIONS.get(home), INITIAL_ELO)
        if away not in ratings:
            ratings[away] = ratings.get(RELOCATIONS.get(away), INITIAL_ELO)
        
        # Apply mean reversion at the start of each new season
        if season != current_season:
            if current_season is not None:
                ratings = apply_season_regression(ratings)
            current_season = season

        elo_home_before = ratings[home]
        elo_away_before = ratings[away]

        new_home, new_away = update_rating(
            ratings[home], ratings[away],
            int(game["home_score"]), int(game["away_score"])
        )

        ratings[home] = new_home
        ratings[away] = new_away

        history.append({
            "game_id":    game["game_id"],
            "season":     season,
            "week":       game["week"],
            "team":       home,
            "opponent":   away,
            "elo_before": round(elo_home_before, 1),
            "elo_after":  round(new_home, 1),
        })
        history.append({
            "game_id":    game["game_id"],
            "season":     season,
            "week":       game["week"],
            "team":       away,
            "opponent":   home,
            "elo_before": round(elo_away_before, 1),
            "elo_after":  round(new_away, 1),
        })

    history_df = pd.DataFrame(history)
    save(history_df, "elo_history")
    for old_team in RELOCATIONS.values():
        ratings.pop(old_team, None)
    return ratings, history_df

def predict_game(
    home_team: str,
    away_team: str,
    ratings: dict[str, float],
) -> dict:
    """
    Predicts the outcome of a single game given current ELO ratings.
    Returns win probabilities and implied spread for both teams.
    """
    rating_home = ratings.get(home_team, INITIAL_ELO)
    rating_away = ratings.get(away_team, INITIAL_ELO)

    adjusted_home = rating_home + HOME_ADVANTAGE
    prob_home = calculate_win_probability(adjusted_home, rating_away)
    prob_away = 1.0 - prob_home

    # Convert ELO difference to implied point spread
    # ~25 ELO points ≈ 1 point on the spread
    elo_diff = adjusted_home - rating_away
    implied_spread = elo_diff / 25.0

    return {
        "home_team":      home_team,
        "away_team":      away_team,
        "home_elo":       round(rating_home, 1),
        "away_elo":       round(rating_away, 1),
        "prob_home":      round(prob_home, 3),
        "prob_away":      round(prob_away, 3),
        "implied_spread": round(implied_spread, 1),
    }
