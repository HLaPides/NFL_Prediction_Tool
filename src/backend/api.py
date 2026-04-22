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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline_status = {
    "running": False,
    "last_run": None,
    "last_result": None,
}

# Recommendation threshold — model must disagree by this many yards to make a call
PROP_THRESHOLD = 25.0


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


# ── Predictions ───────────────────────────────────────────────────────────────
@app.get("/predictions")
def get_predictions(week: Optional[int] = None, season: Optional[int] = None):
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
        query = """
            SELECT game_id, season, week, home_team, away_team, result,
                   vegas_spread, model_spread, spread_gap, confidence,
                   model_winner, vegas_winner, actual_winner
            FROM model_predictions
            WHERE season = ? AND week = ?
            ORDER BY week
        """
        df = pd.read_sql(query, conn, params=(season, week))
    return df.to_dict(orient="records")


# ── ELO ratings ───────────────────────────────────────────────────────────────
@app.get("/elo")
def get_elo_ratings():
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

    total = {
        "season":         "All-time",
        "games":          int(df["games"].sum()),
        "model_accuracy": float(df["model_accuracy"].mean()),
        "vegas_accuracy": float(df["vegas_accuracy"].mean()),
    }

    records = df.to_dict(orient="records")
    records.append(total)
    return records


# ── QB Props ──────────────────────────────────────────────────────────────────
@app.get("/props/qb")
def get_qb_props():
    """
    Returns QB prop predictions for upcoming games.
    Reads from qb_prop_predictions table populated by props_qb.py.
    """
    with get_connection() as conn:
        try:
            df = pd.read_sql("""
                SELECT player_name, team, opponent, week,
                       pred_passing_yards, pred_passing_tds, pred_attempts,
                       implied_team_total, spread_line, is_home
                FROM qb_prop_predictions
                ORDER BY week, team
            """, conn)
        except Exception:
            return []

    if df.empty:
        return []

    return df.to_dict(orient="records")


@app.get("/props/qb/recommendation")
def get_qb_recommendation(player_name: str, prop_line: float, stat: str = "passing_yards"):
    """
    Given a player name, the book's prop line, and the stat type,
    returns a recommendation: OVER, UNDER, or NO BET.

    Example: /props/qb/recommendation?player_name=P.Mahomes&prop_line=275.5&stat=passing_yards
    """
    with get_connection() as conn:
        try:
            df = pd.read_sql("""
                SELECT player_name, team, opponent, week,
                       pred_passing_yards, pred_passing_tds, pred_attempts
                FROM qb_prop_predictions
            """, conn)
        except Exception:
            raise HTTPException(status_code=404, detail="No prop predictions found. Run weekly_update first.")

    if df.empty:
        raise HTTPException(status_code=404, detail="No prop predictions found.")

    # Find player (case-insensitive partial match)
    mask = df["player_name"].str.contains(player_name, case=False, na=False)
    player = df[mask]

    if player.empty:
        raise HTTPException(status_code=404, detail=f"Player '{player_name}' not found.")

    row = player.iloc[0]
    pred_col = f"pred_{stat}"

    if pred_col not in row:
        raise HTTPException(status_code=400, detail=f"Unknown stat '{stat}'. Use passing_yards, passing_tds, or attempts.")

    model_pred  = float(row[pred_col])
    gap         = model_pred - prop_line
    abs_gap     = abs(gap)

    if abs_gap < PROP_THRESHOLD:
        recommendation = "NO BET"
        confidence     = "low"
    elif gap > 0:
        recommendation = "OVER"
        confidence     = "high" if abs_gap > PROP_THRESHOLD * 2 else "medium"
    else:
        recommendation = "UNDER"
        confidence     = "high" if abs_gap > PROP_THRESHOLD * 2 else "medium"

    return {
        "player_name":     row["player_name"],
        "team":            row["team"],
        "opponent":        row["opponent"],
        "week":            int(row["week"]),
        "stat":            stat,
        "model_pred":      round(model_pred, 1),
        "prop_line":       prop_line,
        "gap":             round(gap, 1),
        "recommendation":  recommendation,
        "confidence":      confidence,
        "threshold_used":  PROP_THRESHOLD,
    }


@app.get("/props/qb/all_recommendations")
def get_all_recommendations(threshold: Optional[float] = None):
    """
    Returns recommendations for all QBs where a prop line is stored.
    Only useful once prop lines are populated from an external source.
    For now returns all predictions without a prop line comparison.
    """
    with get_connection() as conn:
        try:
            df = pd.read_sql("""
                SELECT player_name, team, opponent, week,
                       pred_passing_yards, pred_passing_tds, pred_attempts,
                       implied_team_total
                FROM qb_prop_predictions
                ORDER BY pred_passing_yards DESC
            """, conn)
        except Exception:
            return []

    return df.to_dict(orient="records")


# ── Pipeline refresh ──────────────────────────────────────────────────────────
def _run_pipeline():
    pipeline_status["running"] = True
    pipeline_status["last_result"] = None

    backend_dir = Path(__file__).resolve().parent
    python      = sys.executable

    try:
        result = subprocess.run(
            [python, str(backend_dir / "scripts" / "weekly_update.py")],
            capture_output=True, text=True, cwd=str(backend_dir)
        )
        if result.returncode != 0:
            raise RuntimeError(f"weekly_update failed:\n{result.stderr}")

        result = subprocess.run(
            [python, str(backend_dir / "engine" / "predictor.py")],
            capture_output=True, text=True, cwd=str(backend_dir)
        )
        if result.returncode != 0:
            raise RuntimeError(f"predictor failed:\n{result.stderr}")

        # Also run QB props prediction
        result = subprocess.run(
            [python, str(backend_dir / "engine" / "props_qb.py")],
            capture_output=True, text=True, cwd=str(backend_dir)
        )
        if result.returncode != 0:
            raise RuntimeError(f"props_qb failed:\n{result.stderr}")

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


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)