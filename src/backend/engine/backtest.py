
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from data.sql_helpers import load_schedules, load_game_odds, load_elo_history, list_tables
from engine.elo_analyzer import elo_analyzer
from engine.condition_analyzer import ConditionAnalyzer
from engine.main_engine import DOME_ADJ, COLD_ADJ, HIGH_WIND_ADJ, SEVERE_WIND_ADJ, TURF_ADJ


def build_season_predictions(season: int, engine: elo_analyzer, conditions: ConditionAnalyzer) -> pd.DataFrame:
    sched = load_schedules(min_season=season)
    sched = sched[(sched["season"] == season) & (sched["game_type"] == "REG")]

    odds = load_game_odds(min_season=season)
    odds = odds[["game_id", "spread_line"]]

    history = load_elo_history()
    history = history[history["season"] == season]

    home_elo = history[["game_id", "team", "elo_before"]].rename(
        columns={"team": "home_team", "elo_before": "home_elo_before"}
    )
    away_elo = history[["game_id", "team", "elo_before"]].rename(
        columns={"team": "away_team", "elo_before": "away_elo_before"}
    )

    df  = sched.merge(odds, on="game_id", how="left")
    df  = df.merge(home_elo, on=["game_id", "home_team"], how="left")
    df  = df.merge(away_elo, on=["game_id", "away_team"], how="left")
    rows = []

    for _, game in df.iterrows():
        home_elo_val = game["home_elo_before"]
        away_elo_val = game["away_elo_before"]

        if pd.isna(home_elo_val) or pd.isna(away_elo_val):
            continue

        engine.ratings = {game["home_team"]: home_elo_val, game["away_team"]: away_elo_val}
        pred           = engine.predict(game["home_team"], game["away_team"])

        cond = conditions.analyze_game(game["game_id"])
        adj  = 0.0
        if cond["is_dome"]:     adj += DOME_ADJ
        if cond["is_cold"]:     adj += COLD_ADJ
        if cond["severe_wind"]: adj += SEVERE_WIND_ADJ
        elif cond["high_wind"]: adj += HIGH_WIND_ADJ
        if cond["is_turf"]:     adj += TURF_ADJ

        adjusted_prob = max(0.01, min(0.99, pred["prob_home"] + adj))
        actual_result = game["result"]
        vegas_spread  = game["spread_line"]
        elo_spread    = pred["implied_spread"]

        actual_winner = game["home_team"] if actual_result > 0 else game["away_team"]
        elo_winner    = game["home_team"] if pred["prob_home"] > 0.5 else game["away_team"]
        adj_winner    = game["home_team"] if adjusted_prob > 0.5 else game["away_team"]
        vegas_winner  = game["home_team"] if vegas_spread > 0 else game["away_team"]

        rows.append({
            "week":             game["week"],
            "home_team":        game["home_team"],
            "away_team":        game["away_team"],
            "actual_result":    actual_result,
            "vegas_spread":     vegas_spread,
            "elo_spread":       round(elo_spread, 1),
            "spread_diff":      round(elo_spread - vegas_spread, 1),
            "actual_winner":    actual_winner,
            "predicted_winner": adj_winner,
            "elo_winner":       elo_winner,
            "vegas_winner":     vegas_winner,
            "elo_correct":      elo_winner == actual_winner,
            "adj_correct":      adj_winner == actual_winner,
            "vegas_correct":    vegas_winner == actual_winner,
        })

    return pd.DataFrame(rows)


def print_accuracy_report(df: pd.DataFrame, season: int) -> None:
    total         = len(df)
    elo_correct   = df["elo_correct"].sum()
    adj_correct   = df["adj_correct"].sum()
    vegas_correct = df["vegas_correct"].sum()

    print(f"\n{'='*50}")
    print(f"Backtest results — {season} regular season")
    print(f"{'='*50}")
    print(f"Games evaluated:       {total}")
    print(f"ELO correct:           {elo_correct}/{total} ({elo_correct/total*100:.1f}%)")
    print(f"ELO + conditions:      {adj_correct}/{total} ({adj_correct/total*100:.1f}%)")
    print(f"Vegas correct:         {vegas_correct}/{total} ({vegas_correct/total*100:.1f}%)")
    print(f"Adj vs Vegas edge:     {(adj_correct - vegas_correct)/total*100:+.1f}%")
    print(f"\nAvg ELO spread:        {df['elo_spread'].abs().mean():.1f} pts")
    print(f"Avg Vegas spread:      {df['vegas_spread'].abs().mean():.1f} pts")
    print(f"Avg spread diff:       {df['spread_diff'].abs().mean():.1f} pts")

    if adj_correct > vegas_correct:
        print(f"\nJaginator beat Vegas for the season")

    print(f"\nBiggest ELO vs Vegas disagreements:")
    disagreements = df.reindex(df["spread_diff"].abs().sort_values(ascending=False).index)
    print(disagreements[[
        "week", "home_team", "away_team",
        "elo_spread", "vegas_spread", "actual_result"
    ]].head(10).to_string(index=False))


if __name__ == "__main__":
    engine = elo_analyzer()
    conditions = ConditionAnalyzer()
    for season in range(1999, 2026):
        print(f"Running backtest for {season} season...")
        df = build_season_predictions(season, engine, conditions)
        print_accuracy_report(df, season)