"""
predictor.py — Generate weekly game predictions and player props.

Called at the end of weekly_update.py (step 9) after all data has already
been fetched, merged, and the model retrained. Do NOT call this directly
as a replacement for weekly_update.py — predictions against stale data
are meaningless.

Core flow:
    run_predictions()
        └── run_game_predictions()   → weekly_predictions table
        └── run_props_predictions()  → qb/rb/wr_prop_predictions tables

Game prediction logic:
    - Loads the trained XGBoost model from trained_base_model.pkl
    - Builds a feature vector for each upcoming game from current DB state
      (ELO, rolling stats, player features, injury scores, clutch scores,
       game conditions, defense ratings, Vegas spread)
    - Predicts ATS margin (result - vegas_spread), converts back to spread
    - Computes bootstrap confidence (std across 25 bootstrap models)
    - During playoff weeks (19-22), silently switches to the logistic
      playoff model for win probability and pick direction; spread model
      still runs for reference

Injury scoring:
    Position weights × status severity (Out=1.0, Doubtful=0.75,
    Questionable=0.25) summed per team. QB out flagged separately.

Clutch scoring:
    Win rate in close games (≤7 pts) over the last 8 games.
    Falls back to 0.5 (neutral) if fewer than 3 close games in window.

Writes to DB tables:
    weekly_predictions, playoff_predictions,
    qb_prop_predictions, rb_prop_predictions, wr_prop_predictions
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import json
import pandas as pd
import numpy as np
from datetime import datetime
from data.sql_helpers import get_connection, save, merge_season_into_historical
from engine.model import load_trained_model, load_training_data, train_final_model, FEATURE_COLS
from engine.props.qb_props import load_props_model as load_qb_model, predict_current_week as predict_qb
from engine.props.rb_props import load_props_model as load_rb_model, predict_current_week as predict_rb
from engine.props.wr_props import load_props_model as load_wr_model, predict_current_week as predict_wr

# ── Config ────────────────────────────────────────────────────────────────────
WIN_PROB_SCALE    = 25.0
HOME_ADVANTAGE    = 48.0
SKILL_POSITIONS   = {"QB", "RB", "WR", "TE"}
CLOSE_GAME_MARGIN = 7
CLUTCH_ROLL       = 8
PLAYOFF_WEEKS     = {19, 20, 21, 22}

TEAM_DISPLAY_NAMES = {
    "Arizona Cardinals":   "ARI", "Atlanta Falcons":      "ATL", "Baltimore Ravens":   "BAL",
    "Buffalo Bills":       "BUF", "Carolina Panthers":    "CAR", "Chicago Bears":      "CHI",
    "Cincinnati Bengals":  "CIN", "Cleveland Browns":     "CLE", "Dallas Cowboys":     "DAL",
    "Denver Broncos":      "DEN", "Detroit Lions":        "DET", "Green Bay Packers":  "GB",
    "Houston Texans":      "HOU", "Indianapolis Colts":   "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs":  "KC",  "Los Angeles Rams":     "LA",  "Los Angeles Chargers": "LAC",
    "Las Vegas Raiders":   "LV",  "Miami Dolphins":       "MIA", "Minnesota Vikings":  "MIN",
    "New England Patriots":"NE",  "New Orleans Saints":   "NO",  "New York Giants":    "NYG",
    "New York Jets":       "NYJ", "Philadelphia Eagles":  "PHI", "Pittsburgh Steelers":"PIT",
    "Seattle Seahawks":    "SEA", "San Francisco 49ers":  "SF",  "Tampa Bay Buccaneers":"TB",
    "Tennessee Titans":    "TEN", "Washington Commanders":"WAS",
}

POSITION_WEIGHTS = {
    "QB": 10.0, "WR": 2.0, "RB": 2.0, "TE": 1.5,
    "T":   1.0, "G":  0.5, "C":  0.5,
    "CB":  1.0, "S":  1.0, "LB": 0.8, "DE": 0.8, "DT": 0.5,
}

STATUS_SEVERITY = {
    "Out": 1.0, "Doubtful": 0.75, "IR": 1.0,
    "PUP": 1.0, "Suspended": 1.0, "Questionable": 0.25,
}

QB_OUT_STATUSES = {"Out", "Doubtful", "IR", "PUP"}


# ── Playoff detection ─────────────────────────────────────────────────────────
def is_playoff_week(upcoming: pd.DataFrame) -> bool:
    """
    Return True if any upcoming game is a playoff game.
    Checks week numbers first (19-22), then falls back to game_type in DB.
    """
    if upcoming.empty:
        return False
    if upcoming["week"].isin(PLAYOFF_WEEKS).any():
        return True
    try:
        with get_connection() as conn:
            game_ids     = upcoming["game_id"].tolist()
            placeholders = ",".join(["%s"] * len(game_ids))
            types = pd.read_sql(f"""
                SELECT game_type FROM schedules_current
                WHERE game_id IN ({placeholders})
                  AND game_type IN ('WC','DIV','CON','SB')
            """, conn, params=game_ids)
        return not types.empty
    except Exception:
        return False


# ── Playoff predictions table ─────────────────────────────────────────────────
def ensure_playoff_table():
    """Create playoff_predictions table if it doesn't exist."""
    with get_connection() as conn:
        conn.exec_driver_sql("""
            CREATE TABLE IF NOT EXISTS playoff_predictions (
                game_id            TEXT PRIMARY KEY,
                season             INTEGER,
                week               INTEGER,
                home_team          TEXT,
                away_team          TEXT,
                home_win_prob      REAL,
                away_win_prob      REAL,
                playoff_pick       TEXT,
                playoff_confidence REAL,
                features_json      TEXT
            )
        """)
        conn.commit()


def save_playoff_prediction(game_id, season, week, home_team, away_team, pred, features):
    """Upsert a playoff prediction row into playoff_predictions."""
    ensure_playoff_table()
    pick = home_team if pred["playoff_pick"] == "home" else away_team
    with get_connection() as conn:
        conn.exec_driver_sql("""
            INSERT INTO playoff_predictions
            (game_id, season, week, home_team, away_team,
             home_win_prob, away_win_prob, playoff_pick,
             playoff_confidence, features_json)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (game_id) DO UPDATE SET
                season              = EXCLUDED.season,
                week                = EXCLUDED.week,
                home_team           = EXCLUDED.home_team,
                away_team           = EXCLUDED.away_team,
                home_win_prob       = EXCLUDED.home_win_prob,
                away_win_prob       = EXCLUDED.away_win_prob,
                playoff_pick        = EXCLUDED.playoff_pick,
                playoff_confidence  = EXCLUDED.playoff_confidence,
                features_json       = EXCLUDED.features_json
        """, (
            game_id, season, week, home_team, away_team,
            pred["home_win_prob"], pred["away_win_prob"],
            pick, pred["confidence"],
            json.dumps({k: float(v) for k, v in features.items()
                        if isinstance(v, (int, float))})
        ))
        conn.commit()


# ── Clutch scores ─────────────────────────────────────────────────────────────
def build_clutch_scores() -> dict:
    """
    Compute each team's win rate in close games (≤7 pts) over the last
    CLUTCH_ROLL games. Returns { team: clutch_rate } where 0.5 is neutral.
    Falls back to 0.5 if fewer than 3 close games in window.
    """
    try:
        with get_connection() as conn:
            games = pd.read_sql("""
                SELECT game_id, season, week, home_team, away_team, result
                FROM schedules
                WHERE game_type IN ('REG','WC','DIV','CON','SB')
                  AND result IS NOT NULL
                ORDER BY season, week
            """, conn)
    except Exception as e:
        print(f"  Clutch scores: failed ({e}), using 0.5")
        return {}

    home_rows = games[["season","week","home_team","result"]].copy()
    home_rows = home_rows.rename(columns={"home_team":"team"})
    home_rows["team_won"] = (home_rows["result"] > 0).astype(float)
    home_rows["margin"]   = home_rows["result"].abs()

    away_rows = games[["season","week","away_team","result"]].copy()
    away_rows = away_rows.rename(columns={"away_team":"team"})
    away_rows["team_won"] = (away_rows["result"] < 0).astype(float)
    away_rows["margin"]   = away_rows["result"].abs()

    all_rows = pd.concat([home_rows, away_rows], ignore_index=True)
    all_rows = all_rows.sort_values(["team","season","week"]).reset_index(drop=True)
    all_rows["is_close"]  = (all_rows["margin"] <= CLOSE_GAME_MARGIN).astype(float)
    all_rows["close_win"] = all_rows["is_close"] * all_rows["team_won"]

    scores = {}
    for team, grp in all_rows.groupby("team"):
        last_n      = grp.tail(CLUTCH_ROLL)
        close_games = last_n["is_close"].sum()
        close_wins  = last_n["close_win"].sum()
        scores[team] = round(close_wins / close_games, 3) if close_games >= 3 else 0.5

    print(f"  Clutch scores built for {len(scores)} teams")
    return scores


# ── Injury scores ─────────────────────────────────────────────────────────────
def build_injury_scores() -> dict:
    """
    Compute a weighted injury severity score for each team from injuries_current.

    Each player's score = POSITION_WEIGHTS[pos] × STATUS_SEVERITY[status].
    Summed per team. QB out flagged separately as a binary flag.

    Returns:
        dict: { team: {"injury_score": float, "qb_out": int} }
              Only teams with injuries included. Missing teams → score of 0.
    """
    try:
        with get_connection() as conn:
            inj = pd.read_sql("SELECT * FROM injuries_current", conn)
    except Exception:
        return {}

    if inj.empty or "team_display" not in inj.columns:
        return {}

    inj["team"] = inj["team_display"].map(TEAM_DISPLAY_NAMES)
    inj = inj[inj["team"].notna()].copy()

    if "position" not in inj.columns or inj["position"].isna().all():
        try:
            with get_connection() as conn:
                rosters = pd.read_sql("SELECT full_name, position FROM rosters_current", conn)
            inj = inj.merge(rosters, left_on="player_name", right_on="full_name", how="left")
        except Exception:
            inj["position"] = None

    inj["pos_weight"] = inj["position"].map(POSITION_WEIGHTS).fillna(0.5)
    inj["severity"]   = inj["status"].map(STATUS_SEVERITY).fillna(0.0)
    inj["inj_score"]  = inj["pos_weight"] * inj["severity"]

    scores = {}
    for team, grp in inj.groupby("team"):
        qb_rows = grp[grp["position"] == "QB"]
        qb_out  = int(any(qb_rows["status"].isin(QB_OUT_STATUSES)))
        scores[team] = {
            "injury_score": round(float(grp["inj_score"].sum()), 2),
            "qb_out":       qb_out,
        }

    print(f"  Injury scores built for {len(scores)} teams")
    return scores


# ── Load upcoming games ───────────────────────────────────────────────────────
def load_upcoming_games() -> pd.DataFrame:
    """Load all unplayed games from schedules_current with Vegas spread."""
    with get_connection() as conn:
        return pd.read_sql("""
            SELECT s.game_id, s.season, s.week,
                   s.home_team, s.away_team,
                   s.gameday, s.gametime, s.game_type,
                   o.spread_line AS vegas_spread
            FROM schedules_current s
            LEFT JOIN game_odds o ON o.game_id = s.game_id
            WHERE s.home_score IS NULL
            ORDER BY s.week, s.gameday, s.gametime
        """, conn)


# ── Build feature vector ──────────────────────────────────────────────────────
def build_game_features(game: pd.Series,
                        inj_scores: dict = None,
                        clutch_scores: dict = None) -> dict:
    """
    Build a FEATURE_COLS-aligned feature dict for a single upcoming game.

    Pulls from current DB state:
        - ELO ratings (most recent from elo_history)
        - Rolling team stats (most recent from team_rolling_stats)
        - Player features (most recent from player_features)
        - Game conditions (roof, wind from game_conditions)
        - Defense ratings (most recent from defense_ratings)
        - Injury and clutch scores (pre-computed, passed in)

    Args:
        game: Row from load_upcoming_games() DataFrame.
        inj_scores: Output of build_injury_scores().
        clutch_scores: Output of build_clutch_scores().

    Returns:
        dict: Feature name → value, keys matching FEATURE_COLS exactly.
    """
    inj_scores    = inj_scores    or {}
    clutch_scores = clutch_scores or {}

    with get_connection() as conn:
        elo = pd.read_sql("""
            SELECT team, elo_after FROM elo_history
            WHERE (team, season * 100 + week) IN (
                SELECT team, MAX(season * 100 + week) FROM elo_history GROUP BY team
            )
        """, conn)
        elo_map = dict(zip(elo["team"], elo["elo_after"]))

        roll = pd.read_sql("""
            SELECT tr.*, s.season, s.week
            FROM team_rolling_stats tr
            JOIN schedules s ON s.game_id = tr.game_id
            WHERE tr.team IN (%s, %s)
            ORDER BY s.season DESC, s.week DESC
        """, conn, params=(game["home_team"], game["away_team"]))

        home_roll = roll[roll["team"] == game["home_team"]].iloc[0] \
                    if not roll[roll["team"] == game["home_team"]].empty else {}
        away_roll = roll[roll["team"] == game["away_team"]].iloc[0] \
                    if not roll[roll["team"] == game["away_team"]].empty else {}

        pf = pd.read_sql("""
            SELECT pf.* FROM player_features pf
            JOIN schedules s ON s.game_id = pf.game_id
            WHERE s.home_team IN (%s,%s) OR s.away_team IN (%s,%s)
            ORDER BY s.season DESC, s.week DESC LIMIT 20
        """, conn, params=(game["home_team"], game["away_team"],
                           game["home_team"], game["away_team"]))

        cond = pd.read_sql("""
            SELECT roof, wind FROM game_conditions WHERE game_id = %s
        """, conn, params=(game["game_id"],))

        def_ratings = pd.read_sql("""
            SELECT dr.* FROM defense_ratings dr
            INNER JOIN (
                SELECT team, MAX(season * 100 + week) AS max_key
                FROM defense_ratings GROUP BY team
            ) latest ON dr.team = latest.team
              AND (dr.season * 100 + dr.week) = latest.max_key
            WHERE dr.team IN (%s, %s)
        """, conn, params=(game["home_team"], game["away_team"]))

    home_elo = elo_map.get(game["home_team"], 1500)
    away_elo = elo_map.get(game["away_team"], 1500)

    def get_roll(row, col, default=0.0):
        try:   return float(row[col]) if row is not None and col in row else default
        except: return default

    hr  = home_roll if isinstance(home_roll, pd.Series) else {}
    ar  = away_roll if isinstance(away_roll, pd.Series) else {}
    hdr = def_ratings[def_ratings["team"] == game["home_team"]]
    adr = def_ratings[def_ratings["team"] == game["away_team"]]

    is_dome   = 1 if not cond.empty and cond.iloc[0]["roof"] in ("dome","closed") else 0
    high_wind = 1 if not cond.empty and float(cond.iloc[0]["wind"] or 0) > 15 else 0

    def latest_pf(team, col):
        rows = pf[pf.get("home_team", pd.Series()) == team] \
               if "home_team" in pf.columns else pd.DataFrame()
        if rows.empty:
            rows = pf[pf.get("away_team", pd.Series()) == team] \
                   if "away_team" in pf.columns else pd.DataFrame()
        return float(rows.iloc[0][col]) if not rows.empty and col in rows.columns else 0.0

    home_inj    = inj_scores.get(game["home_team"],    {"injury_score": 0.0, "qb_out": 0})
    away_inj    = inj_scores.get(game["away_team"],    {"injury_score": 0.0, "qb_out": 0})
    home_clutch = clutch_scores.get(game["home_team"], 0.5)
    away_clutch = clutch_scores.get(game["away_team"], 0.5)
    vegas_spread = float(game["vegas_spread"]) if pd.notna(game["vegas_spread"]) else 0.0

    return {
        "vegas_spread":               vegas_spread,
        "elo_diff":                   home_elo - away_elo,
        "roll_pass_yards_diff":       get_roll(hr,"roll_passing_yards") - get_roll(ar,"roll_passing_yards"),
        "injury_score_diff":          home_inj["injury_score"] - away_inj["injury_score"],
        "roll_def_tfl_diff":          get_roll(hr,"roll_def_tfl")       - get_roll(ar,"roll_def_tfl"),
        "roll_pass_epa_diff":         get_roll(hr,"roll_passing_epa")   - get_roll(ar,"roll_passing_epa"),
        "te_yards_diff":              latest_pf(game["home_team"],"home_te_yards")  - latest_pf(game["away_team"],"away_te_yards"),
        "qb_epa_diff":                latest_pf(game["home_team"],"home_qb_epa")    - latest_pf(game["away_team"],"away_qb_epa"),
        "wr_yards_diff":              latest_pf(game["home_team"],"home_wr_yards")  - latest_pf(game["away_team"],"away_wr_yards"),
        "rb_rush_diff":               latest_pf(game["home_team"],"home_rb_rush_yards") - latest_pf(game["away_team"],"away_rb_rush_yards"),
        "roll_first_downs_diff":      get_roll(hr,"roll_first_downs")   - get_roll(ar,"roll_first_downs"),
        "roll_def_sacks_diff":        get_roll(hr,"roll_def_sacks")     - get_roll(ar,"roll_def_sacks"),
        "is_dome":                    is_dome,
        "roll_rush_epa_diff":         get_roll(hr,"roll_rushing_epa")   - get_roll(ar,"roll_rushing_epa"),
        "rest_diff":                  0.0,
        "roll_rush_yards_diff":       get_roll(hr,"roll_rushing_yards") - get_roll(ar,"roll_rushing_yards"),
        "roll_penalty_diff":          get_roll(hr,"roll_penalty_yards") - get_roll(ar,"roll_penalty_yards"),
        "roll_def_ints_diff":         get_roll(hr,"roll_def_ints")      - get_roll(ar,"roll_def_ints"),
        "clutch_rate_diff":           home_clutch - away_clutch,
        "rb_rec_diff":                latest_pf(game["home_team"],"home_rb_rec_yards") - latest_pf(game["away_team"],"away_rb_rec_yards"),
        "high_wind":                  high_wind,
    }


# ── Win probability from spread ───────────────────────────────────────────────
def spread_to_win_prob(spread: float) -> float:
    """Convert a point spread to a win probability using ELO-style scaling."""
    import math
    adj = spread * WIN_PROB_SCALE / HOME_ADVANTAGE
    return round(1.0 / (1 + math.pow(10, -adj / 400)), 3)


# ── Run game predictions ──────────────────────────────────────────────────────
def run_game_predictions() -> pd.DataFrame:
    """
    Generate spread predictions for all upcoming games and save to weekly_predictions.

    Process per game:
        1. Build feature vector from current DB state
        2. Predict ATS margin via trained XGBoost, convert back to spread
        3. Score confidence via 25 pre-fit bootstrap models (std of predictions)
        4. During playoff weeks: override win prob and pick with playoff model
        5. Save playoff picks to playoff_predictions table

    Returns:
        pd.DataFrame: One row per upcoming game with all prediction columns.
                      Empty DataFrame if no upcoming games.
    """
    base, _       = load_trained_model()
    upcoming      = load_upcoming_games()
    inj_scores    = build_injury_scores()
    clutch_scores = build_clutch_scores()

    if upcoming.empty:
        print("  No upcoming games found.")
        return pd.DataFrame()

    playoff_mode   = is_playoff_week(upcoming)
    playoff_model  = None
    playoff_scaler = None

    if playoff_mode:
        print("  Playoff week detected — loading playoff model...")
        try:
            from engine.playoff_model import load_playoff_model, predict_playoff_game
            playoff_model, playoff_scaler = load_playoff_model()
            ensure_playoff_table()
            print("  Playoff model loaded.")
        except FileNotFoundError:
            print("  Playoff model not found — run engine/playoff_model.py to train.")
            playoff_mode = False
        except Exception as e:
            print(f"  Playoff model failed ({e}) — using main model.")
            playoff_mode = False
    else:
        print("  Regular season week.")

    # Fit bootstrap models once upfront — O(N_BOOTS) fits not O(N_BOOTS * N_games)
    print("  Computing confidence scores (bootstrap)...")
    boot_models = []
    try:
        from engine.model import load_training_data, _make_base
        train_df = load_training_data()
        rng      = np.random.default_rng(42)
        N_BOOTS  = 25
        for _ in range(N_BOOTS):
            idx    = rng.integers(0, len(train_df), size=len(train_df))
            sample = train_df.iloc[idx]
            m      = _make_base()
            m.fit(
                sample[FEATURE_COLS].fillna(0),
                sample["result"] - sample["vegas_spread"].fillna(0),
            )
            boot_models.append(m)
        print(f"  {N_BOOTS} bootstrap models ready")
    except Exception as e:
        print(f"  Confidence unavailable: {e}")

    def get_confidence(X_game: pd.DataFrame) -> float | None:
        if not boot_models:
            return None
        preds = [float(m.predict(X_game)[0]) for m in boot_models]
        return round(float(np.std(preds)), 3)

    rows = []
    for _, game in upcoming.iterrows():
        try:
            features     = build_game_features(game, inj_scores, clutch_scores)
            X            = pd.DataFrame([features])[FEATURE_COLS].fillna(0)

            vegas_spread = features["vegas_spread"]
            ats_pred     = float(base.predict(X)[0])
            model_spread = round(ats_pred + vegas_spread, 1)
            confidence   = get_confidence(X)

            spread_gap       = round(model_spread - vegas_spread, 1) if vegas_spread else None
            home_win_prob    = spread_to_win_prob(model_spread)
            predicted_winner = game["home_team"] if model_spread > 0 else game["away_team"]

            if playoff_mode and playoff_model is not None:
                try:
                    from engine.playoff_model import predict_playoff_game
                    po_pred          = predict_playoff_game(features, playoff_model, playoff_scaler)
                    home_win_prob    = po_pred["home_win_prob"]
                    predicted_winner = game["home_team"] if po_pred["playoff_pick"] == "home" \
                                       else game["away_team"]
                    save_playoff_prediction(
                        game["game_id"], int(game["season"]), int(game["week"]),
                        game["home_team"], game["away_team"], po_pred, features
                    )
                    print(f"  {game['home_team']:4s} vs {game['away_team']:4s}  "
                          f"playoff: {predicted_winner} ({home_win_prob:.0%})  "
                          f"spread: {model_spread:+.1f}  vegas: {vegas_spread or '?'}")
                except Exception as e:
                    print(f"  {game['home_team']:4s} vs {game['away_team']:4s}  "
                          f"playoff model failed ({e}), using main")
            else:
                edge_flag  = " ★" if spread_gap and abs(spread_gap) >= 3.0 else ""
                conf_label = f"  conf:{confidence:.2f}" if confidence is not None else ""
                print(f"  {game['home_team']:4s} vs {game['away_team']:4s}  "
                      f"model: {model_spread:+.1f}  vegas: {str(vegas_spread or '?'):>5}  "
                      f"gap: {str(spread_gap or '?'):>4}{edge_flag}{conf_label}")

            rows.append({
                "game_id":          game["game_id"],
                "season":           game["season"],
                "week":             game["week"],
                "gameday":          game["gameday"],
                "gametime":         game["gametime"],
                "home_team":        game["home_team"],
                "away_team":        game["away_team"],
                "game_type":        game.get("game_type", "REG"),
                "model_spread":     model_spread,
                "vegas_spread":     vegas_spread if vegas_spread else None,
                "spread_gap":       spread_gap,
                "home_win_prob":    home_win_prob,
                "away_win_prob":    round(1 - home_win_prob, 3),
                "predicted_winner": predicted_winner,
                "confidence":       confidence,
                "generated_at":     datetime.now().isoformat(),
            })

        except Exception as e:
            print(f"  Skipping {game['game_id']}: {e}")

    df = pd.DataFrame(rows)
    if not df.empty:
        save(df, "weekly_predictions")
        print(f"  Saved {len(df)} game predictions.")
    return df


# ── Props ─────────────────────────────────────────────────────────────────────
def run_props_predictions() -> None:
    """
    Generate QB, RB, and WR prop predictions for upcoming games.
    Each position loads its trained models and saves to the corresponding table.
    Failures are caught per position so one bad model doesn't block the others.
    """
    try:
        qb_models = {t: load_qb_model(t) for t in ["passing_yards","passing_tds","attempts","completions","passing_interceptions"]}
        qb_preds  = predict_qb(qb_models)
        if not qb_preds.empty:
            save(qb_preds, "qb_prop_predictions")
            print(f"  QB props saved: {len(qb_preds)} rows")
    except Exception as e:
        print(f"  QB props failed: {e}")

    try:
        rb_models = {t: load_rb_model(t) for t in ["rushing_yards","rushing_tds","carries","receptions","receiving_yards"]}
        rb_preds  = predict_rb(rb_models)
        if not rb_preds.empty:
            save(rb_preds, "rb_prop_predictions")
            print(f"  RB props saved: {len(rb_preds)} rows")
    except Exception as e:
        print(f"  RB props failed: {e}")

    try:
        wr_models = {t: load_wr_model(t) for t in ["receiving_yards","receiving_tds","receptions","targets"]}
        wr_preds  = predict_wr(wr_models)
        if not wr_preds.empty:
            save(wr_preds, "wr_prop_predictions")
            print(f"  WR props saved: {len(wr_preds)} rows")
    except Exception as e:
        print(f"  WR props failed: {e}")


# ── Main entry point ──────────────────────────────────────────────────────────
def run_predictions() -> pd.DataFrame:
    """
    Run game picks and player props for this week's upcoming games.

    Called by weekly_update.py after all data has been fetched, merged,
    and the model retrained. Does NOT re-fetch data or retrain — that is
    weekly_update.py's job (steps 1-8). This is step 9.

    Returns:
        pd.DataFrame: Game predictions written to weekly_predictions.
    """
    print("  Game picks...")
    df = run_game_predictions()
    print("  Props...")
    run_props_predictions()
    return df


if __name__ == "__main__":
    df = run_predictions()
    if not df.empty:
        print(f"\n{'Home':<6} {'Away':<6} {'Model':>8} {'Vegas':>8} {'Gap':>6} {'Win%':>6}")
        print("-" * 44)
        for _, r in df.iterrows():
            edge = " ★" if r["spread_gap"] and abs(r["spread_gap"]) >= 2.0 else ""
            print(f"{r['home_team']:<6} {r['away_team']:<6} "
                  f"{r['model_spread']:>+8.1f}  {str(r['vegas_spread'] or '?'):>7}  "
                  f"{str(r['spread_gap'] or '?'):>5}  {r['home_win_prob']*100:>5.1f}%{edge}")