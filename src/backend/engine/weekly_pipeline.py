import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from datetime import datetime
from data.sql_helpers import get_connection, save
from engine.model import load_trained_model, FEATURE_COLS

# ── Config ────────────────────────────────────────────────────────────────────
# Adjust these without touching the prediction logic
WIN_PROB_SCALE = 25.0   # ELO points per point of spread (for win prob conversion)
HOME_ADVANTAGE = 48.0   # ELO home field advantage points


# ── Load upcoming games ───────────────────────────────────────────────────────
def load_upcoming_games() -> pd.DataFrame:
    """
    Returns games from schedules_current that haven't been played yet.
    Joins Vegas lines from game_odds where available.
    """
    with get_connection() as conn:
        df = pd.read_sql("""
            SELECT
                s.game_id, s.season, s.week,
                s.home_team, s.away_team,
                s.gameday, s.gametime,
                o.spread_line AS vegas_spread
            FROM schedules_current s
            LEFT JOIN game_odds o ON o.game_id = s.game_id
            WHERE s.home_score IS NULL
            ORDER BY s.week, s.gameday, s.gametime
        """, conn)
    return df


# ── Build feature vector for a single game ───────────────────────────────────
def build_game_features(game: pd.Series) -> dict:
    """
    Assembles the full feature vector for an upcoming game using
    current ELO ratings, rolling stats, and player features.
    Returns a dict with all FEATURE_COLS keys.
    """
    with get_connection() as conn:
        # ELO ratings
        elo = pd.read_sql("""
            SELECT team, elo_after
            FROM elo_history
            WHERE (team, season * 100 + week) IN (
                SELECT team, MAX(season * 100 + week)
                FROM elo_history GROUP BY team
            )
        """, conn)
        elo_map = dict(zip(elo["team"], elo["elo_after"]))

        # Rolling stats
        roll = pd.read_sql("""
            SELECT tr.*, s.season, s.week
            FROM team_rolling_stats tr
            JOIN schedules s ON s.game_id = tr.game_id
            WHERE tr.team IN (?, ?)
            ORDER BY s.season DESC, s.week DESC
        """, conn, params=(game["home_team"], game["away_team"]))

        home_roll = roll[roll["team"] == game["home_team"]].iloc[0] if not roll[roll["team"] == game["home_team"]].empty else {}
        away_roll = roll[roll["team"] == game["away_team"]].iloc[0] if not roll[roll["team"] == game["away_team"]].empty else {}

        # Player features
        pf = pd.read_sql("""
            SELECT pf.*
            FROM player_features pf
            JOIN schedules s ON s.game_id = pf.game_id
            WHERE s.home_team IN (?, ?) OR s.away_team IN (?, ?)
            ORDER BY s.season DESC, s.week DESC
            LIMIT 20
        """, conn, params=(game["home_team"], game["away_team"],
                           game["home_team"], game["away_team"]))

        # Conditions for upcoming game
        cond = pd.read_sql("""
            SELECT roof, wind FROM game_conditions
            WHERE game_id = ?
        """, conn, params=(game["game_id"],))

    home_elo = elo_map.get(game["home_team"], 1500)
    away_elo = elo_map.get(game["away_team"], 1500)
    elo_diff = home_elo - away_elo

    # Get most recent player features for each team
    home_pf = pf[(pf["game_id"].isin(
        pd.read_sql(f"SELECT game_id FROM schedules WHERE home_team='{game['home_team']}' OR away_team='{game['home_team']}' ORDER BY season DESC, week DESC LIMIT 5",
                    __import__('sqlite3').connect(str(Path(__file__).resolve().parents[1] / "data" / "jaginator.db")))["game_id"]
    ))].head(1) if not pf.empty else pd.DataFrame()

    def get_roll(row, col, default=0.0):
        try:
            return float(row[col]) if row is not None and col in row else default
        except Exception:
            return default

    hr = home_roll if isinstance(home_roll, pd.Series) else {}
    ar = away_roll if isinstance(away_roll, pd.Series) else {}

    is_dome   = 1 if not cond.empty and cond.iloc[0]["roof"] in ("dome", "closed") else 0
    high_wind = 1 if not cond.empty and float(cond.iloc[0]["wind"] or 0) > 15 else 0

    # Latest player features — fall back to 0 if not available
    def latest_pf(team: str, col: str) -> float:
        rows = pf[pf.get("home_team", pd.Series()) == team] if "home_team" in pf.columns else pd.DataFrame()
        if rows.empty:
            rows = pf[pf.get("away_team", pd.Series()) == team] if "away_team" in pf.columns else pd.DataFrame()
        return float(rows.iloc[0][col]) if not rows.empty and col in rows.columns else 0.0

    home_qb_epa      = latest_pf(game["home_team"], "home_qb_epa")
    away_qb_epa      = latest_pf(game["away_team"], "away_qb_epa")
    home_wr_yards    = latest_pf(game["home_team"], "home_wr_yards")
    away_wr_yards    = latest_pf(game["away_team"], "away_wr_yards")
    home_te_yards    = latest_pf(game["home_team"], "home_te_yards")
    away_te_yards    = latest_pf(game["away_team"], "away_te_yards")
    home_rb_rush     = latest_pf(game["home_team"], "home_rb_rush_yards")
    away_rb_rush     = latest_pf(game["away_team"], "away_rb_rush_yards")
    home_rb_rec      = latest_pf(game["home_team"], "home_rb_rec_yards")
    away_rb_rec      = latest_pf(game["away_team"], "away_rb_rec_yards")

    return {
        "elo_diff":             elo_diff,
        "rest_diff":            0.0,   # not known pre-game, default to 0
        "is_dome":              is_dome,
        "high_wind":            high_wind,
        "qb_epa_diff":          home_qb_epa      - away_qb_epa,
        "wr_yards_diff":        home_wr_yards     - away_wr_yards,
        "te_yards_diff":        home_te_yards     - away_te_yards,
        "rb_rush_diff":         home_rb_rush      - away_rb_rush,
        "rb_rec_diff":          home_rb_rec       - away_rb_rec,
        "roll_pass_epa_diff":   get_roll(hr, "roll_passing_epa")  - get_roll(ar, "roll_passing_epa"),
        "roll_rush_epa_diff":   get_roll(hr, "roll_rushing_epa")  - get_roll(ar, "roll_rushing_epa"),
        "roll_pass_yards_diff": get_roll(hr, "roll_passing_yards") - get_roll(ar, "roll_passing_yards"),
        "roll_rush_yards_diff": get_roll(hr, "roll_rushing_yards") - get_roll(ar, "roll_rushing_yards"),
        "roll_def_sacks_diff":  get_roll(hr, "roll_def_sacks")    - get_roll(ar, "roll_def_sacks"),
        "roll_def_ints_diff":   get_roll(hr, "roll_def_ints")     - get_roll(ar, "roll_def_ints"),
        "roll_def_tfl_diff":    get_roll(hr, "roll_def_tfl")      - get_roll(ar, "roll_def_tfl"),
        "roll_turnovers_diff":  get_roll(hr, "roll_turnovers")    - get_roll(ar, "roll_turnovers"),
        "roll_penalty_diff":    get_roll(hr, "roll_penalty_yards") - get_roll(ar, "roll_penalty_yards"),
        "roll_first_downs_diff":get_roll(hr, "roll_first_downs")  - get_roll(ar, "roll_first_downs"),
    }


# ── Win probability from spread ───────────────────────────────────────────────
def spread_to_win_prob(spread: float) -> float:
    """Converts implied spread to home win probability using logistic curve."""
    import math
    adj = spread * WIN_PROB_SCALE / HOME_ADVANTAGE
    return round(1.0 / (1 + math.pow(10, -adj / 400)), 3)


# ── Run predictions ───────────────────────────────────────────────────────────
def run_predictions() -> pd.DataFrame:
    """
    Main entry point. Loads upcoming games, builds features,
    runs through saved model, writes to weekly_predictions table.
    """
    base, meta = load_trained_model()
    upcoming   = load_upcoming_games()

    if upcoming.empty:
        print("No upcoming games found.")
        return pd.DataFrame()

    rows = []
    for _, game in upcoming.iterrows():
        try:
            features = build_game_features(game)
            X = pd.DataFrame([features])[FEATURE_COLS].fillna(0)

            base_pred = base.predict(X)[0]
            meta_X    = np.array([[
                base_pred,
                game["vegas_spread"] if pd.notna(game["vegas_spread"]) else 0.0,
                features["elo_diff"],
            ]])
            model_spread = float(meta.predict(meta_X)[0])

            vegas_spread   = float(game["vegas_spread"]) if pd.notna(game["vegas_spread"]) else None
            spread_gap     = round(model_spread - vegas_spread, 1) if vegas_spread is not None else None
            home_win_prob  = spread_to_win_prob(model_spread)
            predicted_winner = game["home_team"] if model_spread > 0 else game["away_team"]

            rows.append({
                "game_id":           game["game_id"],
                "season":            game["season"],
                "week":              game["week"],
                "gameday":           game["gameday"],
                "gametime":          game["gametime"],
                "home_team":         game["home_team"],
                "away_team":         game["away_team"],
                "model_spread":      round(model_spread, 1),
                "vegas_spread":      vegas_spread,
                "spread_gap":        spread_gap,
                "home_win_prob":     home_win_prob,
                "away_win_prob":     round(1 - home_win_prob, 3),
                "predicted_winner":  predicted_winner,
                "generated_at":      datetime.now().isoformat(),
            })
            print(f"  {game['home_team']:4s} vs {game['away_team']:4s} — "
                  f"model: {model_spread:+.1f}  vegas: {vegas_spread or '?':>5}  "
                  f"gap: {spread_gap or '?'}")

        except Exception as e:
            print(f"  Skipping {game['game_id']}: {e}")

    df = pd.DataFrame(rows)
    if not df.empty:
        save(df, "weekly_predictions")
        print(f"\nSaved {len(df)} predictions to weekly_predictions table.")
    return df


if __name__ == "__main__":
    print("Running weekly predictions...")
    df = run_predictions()
    if not df.empty:
        print(f"\n{'Team':<20} {'Spread':>8} {'Vegas':>8} {'Gap':>6} {'Win%':>6}")
        print("-" * 50)
        for _, r in df.iterrows():
            print(f"{r['home_team']:4s} vs {r['away_team']:4s}        "
                  f"{r['model_spread']:>+7.1f}  {str(r['vegas_spread'] or '?'):>7}  "
                  f"{str(r['spread_gap'] or '?'):>5}  {r['home_win_prob']*100:>5.1f}%")