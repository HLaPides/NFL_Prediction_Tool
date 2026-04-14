import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import asyncio
import subprocess
from datetime import datetime
from typing import Optional

import pandas as pd
import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from data.sql_helpers import get_connection

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="Jaginator API", version="1.0.0")

# Allow Electron/React to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Track pipeline status
pipeline_status = {
    "running": False,
    "last_run": None,
    "last_result": None,
}


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


# ── Predictions ───────────────────────────────────────────────────────────────
@app.get("/predictions")
def get_predictions(week: Optional[int] = None, season: Optional[int] = None):
    """
    Returns weekly predictions from the DB.
    Optionally filter by week and/or season.
    """
    with get_connection() as conn:
        try:
            df = pd.read_sql("SELECT * FROM weekly_predictions ORDER BY week, gameday, gametime", conn)
        except Exception:
            return []

    if week:
        df = df[df["week"] == week]
    if season:
        df = df[df["season"] == season]

    return df.to_dict(orient="records")


# ── Historical predictions ────────────────────────────────────────────────────
@app.get("/predictions/historical")
def get_historical(season: Optional[int] = None, week: Optional[int] = None):
    print(f"HIT: season={season} week={week}")
    with get_connection() as conn:
        query = "SELECT game_id, season, week, home_team, away_team, result, vegas_spread, model_spread, spread_gap, confidence, model_winner, vegas_winner, actual_winner FROM model_predictions WHERE season = ? AND week = ? ORDER BY week"
        df = pd.read_sql(query, conn, params=(season, week))
    return df.to_dict(orient="records")

# ── ELO ratings ───────────────────────────────────────────────────────────────
@app.get("/elo")
def get_elo_ratings():
    """Returns current ELO ratings for all teams, sorted best to worst."""
    with get_connection() as conn:
        try:
            df = pd.read_sql("""
                SELECT h.team, h.elo_after AS elo
                FROM elo_history h
                INNER JOIN (
                    SELECT team, MAX(season * 100 + week) AS max_key
                    FROM elo_history GROUP BY team
                ) latest ON h.team = latest.team
                  AND (h.season * 100 + h.week) = latest.max_key
                WHERE h.team NOT IN ('OAK', 'SD', 'STL')
                ORDER BY elo DESC
            """, conn)
        except Exception:
            return []

    return df.to_dict(orient="records")


# ── Model stats ───────────────────────────────────────────────────────────────
@app.get("/stats")
def get_model_stats():
    """
    Returns all-time model accuracy stats for the About tab.
    """
    with get_connection() as conn:
        try:
            df = pd.read_sql("""
                SELECT season,
                       COUNT(*) as games,
                       AVG(CASE WHEN model_winner = actual_winner THEN 1.0 ELSE 0.0 END) as model_accuracy,
                       AVG(CASE WHEN vegas_winner = actual_winner THEN 1.0 ELSE 0.0 END) as vegas_accuracy
                FROM model_predictions
                GROUP BY season
                ORDER BY season
            """, conn)
        except Exception:
            return []

    # Add all-time row
    total = {
        "season":         "All-time",
        "games":          int(df["games"].sum()),
        "model_accuracy": float(df["model_accuracy"].mean()),
        "vegas_accuracy": float(df["vegas_accuracy"].mean()),
    }

    records = df.to_dict(orient="records")
    records.append(total)
    return records


# ── Pipeline refresh ──────────────────────────────────────────────────────────
def _run_pipeline():
    """Runs weekly_update + predictor as subprocesses."""
    pipeline_status["running"] = True
    pipeline_status["last_result"] = None

    backend_dir = Path(__file__).resolve().parent
    python      = sys.executable

    try:
        # Step 1: weekly update
        result = subprocess.run(
            [python, str(backend_dir / "scripts" / "weekly_update.py")],
            capture_output=True, text=True, cwd=str(backend_dir)
        )
        if result.returncode != 0:
            raise RuntimeError(f"weekly_update failed:\n{result.stderr}")

        # Step 2: predictor
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
    """
    Triggers the full weekly pipeline in the background.
    Returns immediately — poll /refresh/status to check progress.
    """
    if pipeline_status["running"]:
        raise HTTPException(status_code=409, detail="Pipeline already running")

    background_tasks.add_task(_run_pipeline)
    return {"status": "started"}


@app.get("/refresh/status")
def refresh_status():
    return pipeline_status


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=False) 