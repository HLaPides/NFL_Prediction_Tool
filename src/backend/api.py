import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

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

# ── Props config ──────────────────────────────────────────────────────────────
# Adding a new position = add one entry here, nothing else changes
PROPS_CONFIG = {
    "qb": {
        "table":      "qb_prop_predictions",
        "default_stat": "passing_yards",
        "threshold":  25.0,
        "order_by":   "pred_passing_yards",
        "columns":    "player_name, team, opponent, week, is_home, spread_line, "
                      "implied_team_total, pred_passing_yards, pred_passing_tds, pred_attempts",
    },
    "rb": {
        "table":      "rb_prop_predictions",
        "default_stat": "rushing_yards",
        "threshold":  15.0,
        "order_by":   "pred_rushing_yards",
        "columns":    "player_name, team, opponent, week, is_home, spread_line, "
                      "implied_team_total, pred_rushing_yards, pred_rushing_tds, "
                      "pred_carries, pred_receptions, pred_receiving_yards",
    },
    "wr": {
        "table":      "wr_prop_predictions",
        "default_stat": "receiving_yards",
        "threshold":  15.0,
        "order_by":   "pred_receiving_yards",
        "columns":    "player_name, team, opponent, week, is_home, spread_line, "
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
    """Safe table read — returns empty list on any error."""
    try:
        with get_connection() as conn:
            return pd.read_sql(query, conn, params=params or None).to_dict(orient="records")
    except Exception:
        return []


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
    return _read_table("""
        SELECT game_id, season, week, home_team, away_team, result,
               vegas_spread, model_spread, spread_gap, confidence,
               model_winner, vegas_winner, actual_winner
        FROM model_predictions WHERE season = ? AND week = ? ORDER BY week
    """, (season, week))


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
    """Returns prop predictions for a position. position = qb | rb | wr"""
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
    """
    Returns OVER / UNDER / NO BET for a player's prop line.
    stat defaults to the position's primary stat (passing_yards, rushing_yards, receiving_yards).
    """
    cfg       = _get_position_config(position)
    stat      = stat or cfg["default_stat"]
    threshold = cfg["threshold"]

    records = _read_table(f"SELECT * FROM {cfg['table']}")
    if not records:
        raise HTTPException(status_code=404, detail=f"No {position.upper()} prop data. Run weekly update first.")

    matches = [r for r in records if player_name.lower() in r.get("player_name", "").lower()]
    if not matches:
        raise HTTPException(status_code=404, detail=f"Player '{player_name}' not found.")

    row      = matches[0]
    pred_col = f"pred_{stat}"
    if pred_col not in row:
        raise HTTPException(status_code=400, detail=f"Unknown stat '{stat}' for {position.upper()}.")

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
        for script, label in [
            (backend_dir / "scripts" / "weekly_update.py", "weekly_update"),
            (backend_dir / "engine"  / "predictor.py",     "predictor"),
        ]:
            result = subprocess.run(
                [python, str(script)],
                capture_output=True, text=True, cwd=str(backend_dir)
            )
            if result.returncode != 0:
                raise RuntimeError(f"{label} failed:\n{result.stderr}")
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