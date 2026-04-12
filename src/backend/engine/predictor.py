import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import math
import pandas as pd
import numpy as np
from datetime import datetime
from data.sql_helpers import get_connection, save
from engine.model import load_trained_model, FEATURE_COLS

# ── Config ────────────────────────────────────────────────────────────────────
HOME_ADVANTAGE = 48.0   # ELO points of home field advantage


# ── Data loaders ──────────────────────────────────────────────────────────────
def _load_current_elo() -> dict[str, float]:
    with get_connection() as conn:
        df = pd.read_sql("""
            SELECT h.team, h.elo_after
            FROM elo_history h
            INNER JOIN (
                SELECT team, MAX(season * 100 + week) AS max_key
                FROM elo_history GROUP BY team
            ) latest ON h.team = latest.team
              AND (h.season * 100 + h.week) = latest.max_key
        """, conn)
    return dict(zip(df["team"], df["elo_after"]))


def _load_current_rolling() -> pd.DataFrame:
    """Most recent rolling stats row per team."""
    with get_connection() as conn:
        return pd.read_sql("""
            SELECT tr.*
            FROM team_rolling_stats tr
            INNER JOIN (
                SELECT team, MAX(s.season * 100 + s.week) AS max_key
                FROM team_rolling_stats tr
                JOIN schedules s ON s.game_id = tr.game_id
                GROUP BY tr.team
            ) latest ON tr.team = latest.team
            JOIN schedules s ON s.game_id = tr.game_id
              AND (s.season * 100 + s.week) = latest.max_key
        """, conn)


def _load_current_player_features() -> pd.DataFrame:
    """Most recent player_features row per home and away team."""
    with get_connection() as conn:
        # Get the latest game_id per team (as home or away)
        home = pd.read_sql("""
            SELECT pf.*, s.home_team AS team,
                   s.season * 100 + s.week AS sort_key
            FROM player_features pf
            JOIN schedules s ON s.game_id = pf.game_id
            ORDER BY sort_key DESC
        """, conn)
        away = pd.read_sql("""
            SELECT pf.*, s.away_team AS team,
                   s.season * 100 + s.week AS sort_key
            FROM player_features pf
            JOIN schedules s ON s.game_id = pf.game_id
            ORDER BY sort_key DESC
        """, conn)
    # Keep most recent row per team
    home = home.groupby("team").first().reset_index()
    away = away.groupby("team").first().reset_index()
    return home, away


def load_upcoming_games() -> pd.DataFrame:
    with get_connection() as conn:
        df = pd.read_sql("""
            SELECT s.game_id, s.season, s.week,
                   s.home_team, s.away_team,
                   s.gameday, s.gametime,
                   s.home_rest, s.away_rest,
                   o.spread_line AS vegas_spread,
                   c.roof, c.wind
            FROM schedules_current s
            LEFT JOIN game_odds o        ON o.game_id = s.game_id
            LEFT JOIN game_conditions c  ON c.game_id = s.game_id
            WHERE s.home_score IS NULL
            ORDER BY s.week, s.gameday, s.gametime
        """, conn)
    return df


# ── Feature builder ───────────────────────────────────────────────────────────
def build_features(game: pd.Series,
                   elo_map: dict,
                   rolling: pd.DataFrame,
                   home_pf: pd.DataFrame,
                   away_pf: pd.DataFrame) -> dict:
    ht = game["home_team"]
    at = game["away_team"]

    home_elo = elo_map.get(ht, 1500.0)
    away_elo = elo_map.get(at, 1500.0)

    hr = rolling[rolling["team"] == ht].iloc[0] if not rolling[rolling["team"] == ht].empty else {}
    ar = rolling[rolling["team"] == at].iloc[0] if not rolling[rolling["team"] == at].empty else {}
    hp = home_pf[home_pf["team"] == ht].iloc[0] if not home_pf[home_pf["team"] == ht].empty else {}
    ap = away_pf[away_pf["team"] == at].iloc[0] if not away_pf[away_pf["team"] == at].empty else {}

    def r(row, col, default=0.0):
        try:
            v = row[col]
            return float(v) if pd.notna(v) else default
        except Exception:
            return default

    is_dome   = 1 if str(game.get("roof", "")).lower() in ("dome", "closed") else 0
    high_wind = 1 if float(game.get("wind") or 0) > 15 else 0
    rest_diff = float(game.get("home_rest") or 7) - float(game.get("away_rest") or 7)

    return {
        "elo_diff":             home_elo - away_elo,
        "rest_diff":            rest_diff,
        "is_dome":              is_dome,
        "high_wind":            high_wind,
        "qb_epa_diff":          r(hp, "home_qb_epa")       - r(ap, "away_qb_epa"),
        "wr_yards_diff":        r(hp, "home_wr_yards")      - r(ap, "away_wr_yards"),
        "te_yards_diff":        r(hp, "home_te_yards")      - r(ap, "away_te_yards"),
        "rb_rush_diff":         r(hp, "home_rb_rush_yards") - r(ap, "away_rb_rush_yards"),
        "rb_rec_diff":          r(hp, "home_rb_rec_yards")  - r(ap, "away_rb_rec_yards"),
        "roll_pass_epa_diff":   r(hr, "roll_passing_epa")   - r(ar, "roll_passing_epa"),
        "roll_rush_epa_diff":   r(hr, "roll_rushing_epa")   - r(ar, "roll_rushing_epa"),
        "roll_pass_yards_diff": r(hr, "roll_passing_yards") - r(ar, "roll_passing_yards"),
        "roll_rush_yards_diff": r(hr, "roll_rushing_yards") - r(ar, "roll_rushing_yards"),
        "roll_def_sacks_diff":  r(hr, "roll_def_sacks")     - r(ar, "roll_def_sacks"),
        "roll_def_ints_diff":   r(hr, "roll_def_ints")      - r(ar, "roll_def_ints"),
        "roll_def_tfl_diff":    r(hr, "roll_def_tfl")       - r(ar, "roll_def_tfl"),
        "roll_turnovers_diff":  r(hr, "roll_turnovers")     - r(ar, "roll_turnovers"),
        "roll_penalty_diff":    r(hr, "roll_penalty_yards") - r(ar, "roll_penalty_yards"),
        "roll_first_downs_diff":r(hr, "roll_first_downs")   - r(ar, "roll_first_downs"),
    }


# ── Win probability ───────────────────────────────────────────────────────────
def spread_to_win_prob(spread: float, elo_diff: float) -> float:
    adj = elo_diff + HOME_ADVANTAGE
    return round(1.0 / (1 + math.pow(10, -adj / 400)), 3)


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run_predictions() -> pd.DataFrame:
    print("Loading model...")
    base, meta = load_trained_model()

    print("Loading current data...")
    elo_map          = _load_current_elo()
    rolling          = _load_current_rolling()
    home_pf, away_pf = _load_current_player_features()
    upcoming         = load_upcoming_games()

    if upcoming.empty:
        print("No upcoming games found — run weekly_update.py first.")
        return pd.DataFrame()

    print(f"Predicting {len(upcoming)} games...\n")
    rows = []

    for _, game in upcoming.iterrows():
        try:
            features = build_features(game, elo_map, rolling, home_pf, away_pf)
            X        = pd.DataFrame([features])[FEATURE_COLS].fillna(0)

            base_pred    = float(base.predict(X)[0])
            vegas_spread = float(game["vegas_spread"]) if pd.notna(game.get("vegas_spread")) else 0.0
            elo_diff     = features["elo_diff"]

            meta_X       = np.array([[base_pred, vegas_spread, elo_diff]])
            model_spread = float(meta.predict(meta_X)[0])
            spread_gap   = round(model_spread - vegas_spread, 1) if pd.notna(game.get("vegas_spread")) else None
            home_win_prob= spread_to_win_prob(model_spread, elo_diff)

            rows.append({
                "game_id":          game["game_id"],
                "season":           game["season"],
                "week":             game["week"],
                "gameday":          game.get("gameday"),
                "gametime":         game.get("gametime"),
                "home_team":        game["home_team"],
                "away_team":        game["away_team"],
                "model_spread":     round(model_spread, 1),
                "vegas_spread":     vegas_spread if pd.notna(game.get("vegas_spread")) else None,
                "spread_gap":       spread_gap,
                "home_win_prob":    home_win_prob,
                "away_win_prob":    round(1 - home_win_prob, 3),
                "predicted_winner": game["home_team"] if model_spread > 0 else game["away_team"],
                "generated_at":     datetime.now().isoformat(),
            })

            vs = f"{vegas_spread:+.1f}" if pd.notna(game.get("vegas_spread")) else "  ?"
            print(f"  {game['home_team']:4s} vs {game['away_team']:4s}  "
                  f"model: {model_spread:+.1f}  vegas: {vs}  "
                  f"gap: {spread_gap or '?'}  "
                  f"home win: {home_win_prob*100:.1f}%")

        except Exception as e:
            print(f"  Skipping {game['game_id']}: {e}")

    df = pd.DataFrame(rows)
    if not df.empty:
        save(df, "weekly_predictions")
        print(f"\nSaved {len(df)} predictions → weekly_predictions table.")
    return df


if __name__ == "__main__":
    run_predictions()