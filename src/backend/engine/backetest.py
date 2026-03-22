
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from data.sql_helpers import load_schedules, load_game_odds, load_elo_history
from engine.elo import predict_game

#Backtest curtosey of Claude
def build_season_predictions(season: int) -> pd.DataFrame:
    """
    For every regular season game in a given season, look up each team's
    ELO rating going into that game (elo_before) and generate a prediction.
    Merges with actual results and Vegas lines for comparison.
    """
    sched = load_schedules(min_season=season)
    sched = sched[(sched["season"] == season) & (sched["game_type"] == "REG")]

    odds = load_game_odds(min_season=season)
    odds = odds[["game_id", "spread_line"]]

    history = load_elo_history()
    history = history[history["season"] == season]

    # Get each team's ELO going INTO each game (elo_before)
    home_elo = history[["game_id", "team", "elo_before"]].rename(
        columns={"team": "home_team", "elo_before": "home_elo_before"}
    )
    away_elo = history[["game_id", "team", "elo_before"]].rename(
        columns={"team": "away_team", "elo_before": "away_elo_before"}
    )

    df = sched.merge(odds, on="game_id", how="left")
    df = df.merge(home_elo, on=["game_id", "home_team"], how="left")
    df = df.merge(away_elo, on=["game_id", "away_team"], how="left")

    rows = []
    for _, game in df.iterrows():
        home_elo_val = game["home_elo_before"]
        away_elo_val = game["away_elo_before"]

        if pd.isna(home_elo_val) or pd.isna(away_elo_val):
            continue

        ratings = {game["home_team"]: home_elo_val, game["away_team"]: away_elo_val}
        pred = predict_game(game["home_team"], game["away_team"], ratings)

        actual_result   = game["result"]           # positive = home win
        vegas_spread    = game["spread_line"]      # negative = home favored
        elo_spread      = pred["implied_spread"]   # positive = home favored

        actual_winner   = game["home_team"] if actual_result > 0 else game["away_team"]
        predicted_winner = game["home_team"] if pred["prob_home"] > 0.5 else game["away_team"]
        vegas_winner = game["home_team"] if vegas_spread > 0 else game["away_team"]

        # Did the home team cover the Vegas spread?
        elo_covers_spread = (actual_result + vegas_spread) > 0

        rows.append({
            "week":             game["week"],
            "home_team":        game["home_team"],
            "away_team":        game["away_team"],
            "actual_result":    actual_result,
            "vegas_spread":     vegas_spread,
            "elo_spread":       round(elo_spread, 1),
            "spread_diff": round(elo_spread - vegas_spread, 1),  # positive = ELO thinks home is stronger than Vegas
            "actual_winner":    actual_winner,
            "predicted_winner": predicted_winner,
            "vegas_winner":     vegas_winner,
            "elo_correct":      predicted_winner == actual_winner,
            "vegas_correct":    vegas_winner == actual_winner,
        })

    return pd.DataFrame(rows)


def print_accuracy_report(df: pd.DataFrame, season: int) -> None:
    total         = len(df)
    elo_correct   = df["elo_correct"].sum()
    vegas_correct = df["vegas_correct"].sum()

    print(f"\n{'='*50}")
    print(f"Backtest results — {season} regular season")
    print(f"{'='*50}")
    print(f"Games evaluated:       {total}")
    print(f"ELO correct:           {elo_correct}/{total} ({elo_correct/total*100:.1f}%)")
    print(f"Vegas correct:         {vegas_correct}/{total} ({vegas_correct/total*100:.1f}%)")
    print(f"ELO vs Vegas edge:     {(elo_correct - vegas_correct)/total*100:+.1f}%")
    print(f"\nAvg ELO spread:        {df['elo_spread'].abs().mean():.1f} pts")
    print(f"Avg Vegas spread:      {df['vegas_spread'].abs().mean():.1f} pts")
    print(f"Avg spread diff:       {df['spread_diff'].abs().mean():.1f} pts")

    print(f"\nBiggest ELO vs Vegas disagreements:")
    disagreements = df.reindex(df["spread_diff"].abs().sort_values(ascending=False).index)
    print(disagreements[["week", "home_team", "away_team", "elo_spread", "vegas_spread", "actual_result"]].head(10).to_string(index=False))


if __name__ == "__main__":
    for season in range(1999,2025):
        print(f"Running backtest for {season} season...")
        df = build_season_predictions(season)
        print_accuracy_report(df, season)