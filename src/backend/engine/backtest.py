import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from data.sql_helpers import load_schedules, load_game_odds, load_elo_history, get_connection
from engine.elo_analyzer import elo_analyzer
from engine.condition_analyzer import condition_analyzer
from engine.main_engine import DOME_ADJ, COLD_ADJ, HIGH_WIND_ADJ, SEVERE_WIND_ADJ, TURF_ADJ
from engine.players.player_analyzer import player_analyzer

#
# ── Scoring weights ───────────────────────────────────────────────────────────
# Each component is normalized to a comparable scale then weighted.
# Final score delta is multiplied by PLAYER_SCALE to get prob adjustment.
W_QB    = 0.40   # EPA/attempt — most predictive single signal
W_WR    = 0.15   # WR receiving yards per game
W_TE    = 0.10   # TE receiving yards per game
W_RB    = 0.10   # RB total yards per game (rush + rec)
W_DEF   = 0.25   # Defensive quality (pts allowed + turnovers forced)

PLAYER_SCALE = 0.25   # Converts normalized score delta → probability shift


def _offense_score(off: dict) -> float:
    """
    Returns a single offensive quality number from all four position slots.
    Each metric is divided by a league-average baseline so deltas are unit-free.
    """
    qb  = off.get("qb",  {})
    wr  = off.get("wr",  {})
    te  = off.get("te",  {})
    rb  = off.get("rb",  {})

    # Normalize each metric against a rough league average
    qb_score = qb.get("epa_per_att",    0.15) / 0.15          # league avg ~0.15
    wr_score = wr.get("receiving_yards", 60.0) / 60.0          # league avg ~60 yds/gm
    te_score = te.get("receiving_yards", 45.0) / 45.0          # league avg ~45 yds/gm
    rb_score = (rb.get("rush_yards", 60.0) +
                rb.get("rec_yards",  15.0)) / 75.0             # league avg ~75 yds/gm

    return (W_QB * qb_score +
            W_WR * wr_score +
            W_TE * te_score +
            W_RB * rb_score)


def _defense_score(d: dict) -> float:
    """
    Returns a defensive quality number. Higher = better defense.
    pts_allowed is inverted (lower allowed = better).
    turnovers_forced is a bonus.
    """
    pts  = d.get("pts_allowed",      22.0)
    to   = d.get("turnovers_forced", 25.0)

    pts_score = (22.0 / max(pts, 1.0))   # league avg ~22; better D scores > 1
    to_score  = to / 25.0                 # league avg ~25 turnovers/season

    return (pts_score * 0.7) + (to_score * 0.3)

def _has_live_data(matchup: dict, min_games: int = 3) -> bool:
    """
    Returns True only if both teams have enough in-season games to trust
    the player evaluations. Prevents career-average leakage in early weeks
    and historical seasons where we have no real weekly data.
    """
    for side in ("home", "away"):
        qb_games = matchup[side]["offense"]["qb"].get("games_used", 0)
        if qb_games < min_games:
            return False
    return True

def compute_player_adjustment(matchup: dict) -> float:
    """
    Returns a probability adjustment for the home team based on the full
    player matchup. Positive = home team has the better personnel edge.
    """
    home_off = _offense_score(matchup["home"]["offense"])
    away_off = _offense_score(matchup["away"]["offense"])

    home_def = _defense_score(matchup["home"]["defense"])
    away_def = _defense_score(matchup["away"]["defense"])

    # Offense delta: home offense vs away offense
    off_delta = home_off - away_off

    # Defense delta: home defense suppresses away offense (positive = home advantage)
    def_delta = home_def - away_def

    raw = (W_QB + W_WR + W_TE + W_RB) * off_delta + W_DEF * def_delta
    return raw * PLAYER_SCALE


def build_season_predictions(
    season: int,
    engine: elo_analyzer,
    conditions: condition_analyzer,
    players: player_analyzer
) -> pd.DataFrame:
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

    df = sched.merge(odds, on="game_id", how="left")
    df = df.merge(home_elo, on=["game_id", "home_team"], how="left")
    df = df.merge(away_elo, on=["game_id", "away_team"], how="left")

    rows = []
    for _, game in df.iterrows():
        home_elo_val = game["home_elo_before"]
        away_elo_val = game["away_elo_before"]

        if pd.isna(home_elo_val) or pd.isna(away_elo_val):
            continue

        # ── ELO base prediction ───────────────────────────────────────────────
        engine.ratings = {game["home_team"]: home_elo_val, game["away_team"]: away_elo_val}
        pred = engine.predict(game["home_team"], game["away_team"])

        # ── Conditions adjustment ─────────────────────────────────────────────
        cond = conditions.analyze_game(game["game_id"])
        cond_adj = 0.0
        if cond["is_dome"]:     cond_adj += DOME_ADJ
        if cond["is_cold"]:     cond_adj += COLD_ADJ
        if cond["severe_wind"]: cond_adj += SEVERE_WIND_ADJ
        elif cond["high_wind"]: cond_adj += HIGH_WIND_ADJ
        if cond["is_turf"]:     cond_adj += TURF_ADJ

        # ── Player adjustment ─────────────────────────────────────────────────
        player_adj = 0.0
        try:
            matchup    = players.evaluate_matchup(
                game["home_team"], game["away_team"],
                int(game["week"]), season,
            )
            if _has_live_data(matchup):
                player_adj = compute_player_adjustment(matchup)
        except Exception:
            pass

        # ── Final adjusted probability ────────────────────────────────────────
        total_adj     = cond_adj + player_adj
        adjusted_prob = max(0.01, min(0.99, pred["prob_home"] + total_adj))

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
            "cond_adj":         round(cond_adj, 4),
            "player_adj":       round(player_adj, 4),
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
    print(f"ELO + conditions + QB: {adj_correct}/{total} ({adj_correct/total*100:.1f}%)")
    print(f"Vegas correct:         {vegas_correct}/{total} ({vegas_correct/total*100:.1f}%)")
    print(f"Adj vs Vegas edge:     {(adj_correct - vegas_correct)/total*100:+.1f}%")
    print(f"\nAvg ELO spread:        {df['elo_spread'].abs().mean():.1f} pts")
    print(f"Avg Vegas spread:      {df['vegas_spread'].abs().mean():.1f} pts")
    print(f"Avg spread diff:       {df['spread_diff'].abs().mean():.1f} pts")
    print(f"Avg cond adj:          {df['cond_adj'].abs().mean():.4f}")
    print(f"Avg player adj:        {df['player_adj'].abs().mean():.4f}")

    if adj_correct > vegas_correct:
        print(f"\nJaginator beat Vegas for the season")

    print(f"\nBiggest ELO vs Vegas disagreements:")
    disagreements = df.reindex(df["spread_diff"].abs().sort_values(ascending=False).index)
    print(disagreements[[
        "week", "home_team", "away_team",
        "elo_spread", "vegas_spread", "actual_result"
    ]].head(10).to_string(index=False))


if __name__ == "__main__":
    engine     = elo_analyzer()
    conditions = condition_analyzer()
    players    = player_analyzer()

    matchup = players.evaluate_matchup("KC", "BUF", 10, 2025)

    print("KC QB games_used:", matchup["home"]["offense"]["qb"]["games_used"])
    print("KC QB source:", matchup["home"]["offense"]["qb"]["source"])
    print("BUF QB games_used:", matchup["away"]["offense"]["qb"]["games_used"])
    print("BUF QB source:", matchup["away"]["offense"]["qb"]["source"])

    df = build_season_predictions(2023, engine, conditions, players)

    flipped = df[df["elo_winner"] != df["predicted_winner"]]
    print(f"Games where player adj changed the pick: {len(flipped)}")
    print(f"Of those, adj was correct: {flipped['adj_correct'].sum()}")
    print(f"Of those, elo was correct: {flipped['elo_correct'].sum()}")
    print(f"\nSample flipped games:")
    print(flipped[["week","home_team","away_team","elo_spread","player_adj","actual_result"]].head(15).to_string(index=False))

"""
    for season in range(2020, 2026):
        print(f"Running backtest for {season} season...")
        df = build_season_predictions(season, engine, conditions, players)
        print_accuracy_report(df, season)
"""