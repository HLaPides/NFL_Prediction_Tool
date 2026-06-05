import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import json
import subprocess
from datetime import datetime
from typing import Optional

import pandas as pd
import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from data.sql_helpers import get_connection

app = FastAPI(title="Jaginator API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

pipeline_status = {"running": False, "last_run": None, "last_result": None}

PLAYOFF_GAME_TYPES = {"WC", "DIV", "CON", "SB"}

# ── Props config ──────────────────────────────────────────────────────────────
PROPS_CONFIG = {
    "qb": {
        "table":        "qb_prop_predictions",
        "default_stat": "passing_yards",
        "threshold":    25.0,
        "order_by":     "pred_passing_yards",
        "columns":      "player_name, team, opponent, week, is_home, spread_line, "
                        "implied_team_total, pred_passing_yards, pred_passing_tds, pred_attempts",
    },
    "rb": {
        "table":        "rb_prop_predictions",
        "default_stat": "rushing_yards",
        "threshold":    15.0,
        "order_by":     "pred_rushing_yards",
        "columns":      "player_name, team, opponent, week, is_home, spread_line, "
                        "implied_team_total, pred_rushing_yards, pred_rushing_tds, "
                        "pred_carries, pred_receptions, pred_receiving_yards",
    },
    "wr": {
        "table":        "wr_prop_predictions",
        "default_stat": "receiving_yards",
        "threshold":    15.0,
        "order_by":     "pred_receiving_yards",
        "columns":      "player_name, team, opponent, week, is_home, spread_line, "
                        "implied_team_total, pred_receiving_yards, pred_receiving_tds, "
                        "pred_receptions, pred_targets",
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _get_position_config(position: str) -> dict:
    cfg = PROPS_CONFIG.get(position.lower())
    if not cfg:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown position '{position}'. Valid: {list(PROPS_CONFIG.keys())}"
        )
    return cfg


def _prop_recommendation(model_pred: float, prop_line: float, threshold: float) -> dict:
    gap     = model_pred - prop_line
    abs_gap = abs(gap)
    if abs_gap < threshold:
        rec, conf = "NO BET", "low"
    elif gap > 0:
        rec, conf = "OVER",  "high" if abs_gap > threshold * 2 else "medium"
    else:
        rec, conf = "UNDER", "high" if abs_gap > threshold * 2 else "medium"
    return {"gap": round(gap, 1), "recommendation": rec, "confidence": conf}


def _read_table(query: str, params: tuple = ()) -> list:
    try:
        with get_connection() as conn:
            return pd.read_sql(query, conn, params=params or None).to_dict(orient="records")
    except Exception:
        return []


# ── Playoff model — reads from playoff_predictions table ──────────────────────
def _apply_playoff_model(records: list) -> list:
    """
    For playoff games in historical records, replace model_winner and
    home_win_prob using predictions stored in the playoff_predictions table.
    Falls back to recomputing with the model if not in the table.
    """
    if not records:
        return records

    # Get game types from schedules
    try:
        game_ids     = [r["game_id"] for r in records]
        placeholders = ",".join("?" * len(game_ids))
        with get_connection() as conn:
            game_types = pd.read_sql(
                f"SELECT game_id, game_type FROM schedules "
                f"WHERE game_id IN ({placeholders})",
                conn, params=game_ids
            )
        type_map = dict(zip(game_types["game_id"], game_types["game_type"]))
    except Exception:
        type_map = {}

    for r in records:
        r["game_type"] = type_map.get(r["game_id"], "REG")

    playoff_records = [r for r in records if r["game_type"] in PLAYOFF_GAME_TYPES]
    if not playoff_records:
        return records

    # Load stored playoff predictions
    try:
        po_ids       = [r["game_id"] for r in playoff_records]
        placeholders = ",".join("?" * len(po_ids))
        with get_connection() as conn:
            stored = pd.read_sql(
                f"SELECT * FROM playoff_predictions WHERE game_id IN ({placeholders})",
                conn, params=po_ids
            )
        stored_map = {row["game_id"]: row for _, row in stored.iterrows()}
    except Exception:
        stored_map = {}

    # For any playoff game not in stored_map, recompute using the model
    missing = [r for r in playoff_records if r["game_id"] not in stored_map]
    if missing:
        try:
            from engine.playoff_model import load_playoff_model, predict_playoff_game, PLAYOFF_FEATURES
            model, scaler = load_playoff_model()

            # Try to get features from SQL (best effort fallback)
            miss_ids     = [r["game_id"] for r in missing]
            placeholders = ",".join("?" * len(miss_ids))
            with get_connection() as conn:
                feat_df = pd.read_sql(f"""
                    SELECT
                        s.game_id,
                        (h.elo_before - a.elo_before)               AS elo_diff,
                        COALESCE(s.home_rest,7)-COALESCE(s.away_rest,7) AS rest_diff,
                        pf.home_qb_epa - pf.away_qb_epa             AS qb_epa_diff,
                        hr.roll_passing_epa - ar.roll_passing_epa    AS roll_pass_epa_diff,
                        hr.roll_first_downs - ar.roll_first_downs    AS roll_first_downs_diff,
                        0.0                                          AS clutch_rate_diff,
                        COALESCE(hd.roll_adj_def_pass_epa,0)
                            - COALESCE(ad.roll_adj_def_pass_epa,0)  AS roll_adj_def_pass_epa_diff,
                        0.0 AS injury_score_diff,
                        0.0 AS qb_out_diff
                    FROM schedules s
                    JOIN elo_history h  ON h.game_id = s.game_id AND h.team = s.home_team
                    JOIN elo_history a  ON a.game_id = s.game_id AND a.team = s.away_team
                    JOIN player_features pf ON pf.game_id = s.game_id
                    LEFT JOIN team_rolling_stats hr
                        ON hr.game_id = s.game_id AND hr.team = s.home_team
                    LEFT JOIN team_rolling_stats ar
                        ON ar.game_id = s.game_id AND ar.team = s.away_team
                    LEFT JOIN defense_ratings hd
                        ON hd.game_id = s.game_id AND hd.team = s.home_team
                    LEFT JOIN defense_ratings ad
                        ON ad.game_id = s.game_id AND ad.team = s.away_team
                    WHERE s.game_id IN ({placeholders})
                """, conn, params=miss_ids)
            feat_map = {row["game_id"]: row for _, row in feat_df.iterrows()}

            for r in missing:
                features = feat_map.get(r["game_id"], {})
                pred     = predict_playoff_game(features, model, scaler)
                pick     = r["home_team"] if pred["playoff_pick"] == "home" else r["away_team"]
                # Store as a mock row
                stored_map[r["game_id"]] = {
                    "home_win_prob":      pred["home_win_prob"],
                    "away_win_prob":      pred["away_win_prob"],
                    "playoff_pick":       pick,
                    "playoff_confidence": pred["confidence"],
                }
        except Exception:
            pass

    # Apply stored predictions to records
    for r in records:
        if r["game_type"] not in PLAYOFF_GAME_TYPES:
            continue
        stored_row = stored_map.get(r["game_id"])
        if stored_row is None:
            continue
        try:
            r["model_winner"]  = stored_row["playoff_pick"]
            r["home_win_prob"] = stored_row["home_win_prob"]
            r["away_win_prob"] = stored_row["away_win_prob"]
        except Exception:
            pass

    return records


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


# ── Game predictions ──────────────────────────────────────────────────────────
@app.get("/predictions")
def get_predictions(week: Optional[int] = None, season: Optional[int] = None):
    records = _read_table("SELECT * FROM weekly_predictions ORDER BY week, gameday, gametime")
    if week:   records = [r for r in records if r["week"]   == week]
    if season: records = [r for r in records if r["season"] == season]
    return records


@app.get("/predictions/historical")
def get_historical(season: Optional[int] = None, week: Optional[int] = None):
    records = _read_table("""
        SELECT game_id, season, week, home_team, away_team, result,
               vegas_spread, model_spread, spread_gap, confidence,
               model_winner, vegas_winner, actual_winner
        FROM model_predictions WHERE season = ? AND week = ? ORDER BY week
    """, (season, week))

    records = _apply_playoff_model(records)
    return records


# ── ELO ───────────────────────────────────────────────────────────────────────
@app.get("/elo")
def get_elo_ratings():
    return _read_table("""
        SELECT h.team, h.elo_after AS elo FROM elo_history h
        INNER JOIN (
            SELECT team, MAX(season * 100 + week) AS max_key
            FROM elo_history GROUP BY team
        ) latest ON h.team = latest.team
          AND (h.season * 100 + h.week) = latest.max_key
        WHERE h.team NOT IN ('OAK','SD','STL') ORDER BY elo DESC
    """)


# ── Model stats ───────────────────────────────────────────────────────────────
@app.get("/stats")
def get_model_stats():
    records = _read_table("""
        SELECT season, COUNT(*) as games,
               AVG(CASE WHEN model_winner = actual_winner THEN 1.0 ELSE 0.0 END) as model_accuracy,
               AVG(CASE WHEN vegas_winner = actual_winner THEN 1.0 ELSE 0.0 END) as vegas_accuracy
        FROM model_predictions GROUP BY season ORDER BY season
    """)
    if not records:
        return []
    total = {
        "season":         "All-time",
        "games":          sum(r["games"] for r in records),
        "model_accuracy": sum(r["model_accuracy"] for r in records) / len(records),
        "vegas_accuracy": sum(r["vegas_accuracy"] for r in records) / len(records),
    }
    return records + [total]


# ── Props ─────────────────────────────────────────────────────────────────────
@app.get("/props/{position}")
def get_props(position: str):
    cfg = _get_position_config(position)
    return _read_table(
        f"SELECT {cfg['columns']} FROM {cfg['table']} ORDER BY week, {cfg['order_by']} DESC"
    )


@app.get("/props/{position}/recommendation")
def get_recommendation(
    position:    str,
    player_name: str,
    prop_line:   float,
    stat:        Optional[str] = None,
):
    cfg       = _get_position_config(position)
    stat      = stat or cfg["default_stat"]
    threshold = cfg["threshold"]

    records = _read_table(f"SELECT * FROM {cfg['table']}")
    if not records:
        raise HTTPException(status_code=404, detail=f"No {position.upper()} prop data.")

    matches = [r for r in records if player_name.lower() in r.get("player_name","").lower()]
    if not matches:
        raise HTTPException(status_code=404, detail=f"Player '{player_name}' not found.")

    row      = matches[0]
    pred_col = f"pred_{stat}"
    if pred_col not in row:
        raise HTTPException(status_code=400, detail=f"Unknown stat '{stat}'.")

    model_pred = float(row[pred_col])
    return {
        "player_name":    row["player_name"],
        "team":           row["team"],
        "opponent":       row["opponent"],
        "week":           int(row["week"]),
        "position":       position.upper(),
        "stat":           stat,
        "model_pred":     round(model_pred, 1),
        "prop_line":      prop_line,
        "threshold_used": threshold,
        **_prop_recommendation(model_pred, prop_line, threshold),
    }


# ── Pipeline ──────────────────────────────────────────────────────────────────
def _run_pipeline():
    pipeline_status["running"]     = True
    pipeline_status["last_result"] = None
    backend_dir = Path(__file__).resolve().parent
    python      = sys.executable
    try:
        result = subprocess.run(
            [python, str(backend_dir / "engine" / "predictor.py")],
            capture_output=True, text=True, cwd=str(backend_dir)
        )
        if result.returncode != 0:
            raise RuntimeError(f"predictor failed:\n{result.stderr}")
        pipeline_status["last_result"] = "success"
    except Exception as e:
        pipeline_status["last_result"] = f"error: {str(e)}"
    finally:
        pipeline_status["running"]  = False
        pipeline_status["last_run"] = datetime.now().isoformat()


@app.post("/refresh")
def refresh(background_tasks: BackgroundTasks):
    if pipeline_status["running"]:
        raise HTTPException(status_code=409, detail="Pipeline already running")
    background_tasks.add_task(_run_pipeline)
    return {"status": "started"}


@app.get("/refresh/status")
def refresh_status():
    return pipeline_status


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)