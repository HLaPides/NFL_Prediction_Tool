import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from data.sql_helpers import get_connection, save

ROLLING_WEEKS = 4


def build_rolling_stats() -> pd.DataFrame:
    with get_connection() as conn:
        stats = pd.read_sql("""
            SELECT
                t.season, t.week, t.team,
                s.game_id,
                s.home_team, s.away_team,
                s.home_score, s.away_score,
                t.passing_epa, t.rushing_epa,
                t.passing_yards, t.rushing_yards,
                t.passing_tds, t.rushing_tds,
                t.passing_interceptions,
                t.rushing_fumbles, t.receiving_fumbles,
                t.passing_first_downs, t.rushing_first_downs,
                t.receiving_first_downs,
                t.penalties, t.penalty_yards,
                t.def_sacks, t.def_interceptions,
                t.def_fumbles_forced, t.def_tackles_for_loss,
                t.def_qb_hits, t.def_pass_defended,
                CASE WHEN s.home_team = t.team THEN p.home_coach
                     ELSE p.away_coach END AS coach
            FROM team_stats_weekly t
            JOIN schedules s
                ON s.season = t.season
                AND s.week   = t.week
                AND (s.home_team = t.team OR s.away_team = t.team)
            LEFT JOIN game_personnel p ON p.game_id = s.game_id
            WHERE t.season_type IN ('REG', 'POST')
            ORDER BY t.season, t.week
        """, conn)

    # Compute point differential from each team's perspective
    stats["point_diff"] = np.where(
        stats["team"] == stats["home_team"],
        stats["home_score"] - stats["away_score"],
        stats["away_score"] - stats["home_score"]
    )

    # Build coach win history for fast lookup
    stats["won"] = (stats["point_diff"] > 0).astype(int)
    coach_history = stats[["season", "week", "team", "coach", "won"]].copy()

    rows = []

    for team, grp in stats.groupby("team"):
        grp = grp.sort_values(["season", "week"]).reset_index(drop=True)

        for i, row in grp.iterrows():
            past = grp[
                (grp["season"] < row["season"]) |
                ((grp["season"] == row["season"]) & (grp["week"] < row["week"]))
            ].tail(ROLLING_WEEKS)

            if past.empty:
                # Consecutive win/loss streak
                wins_losses = (past["point_diff"] > 0).astype(int).values
                streak = 0
            for result in reversed(wins_losses):
                if result == 1:
                    if streak >= 0:
                        streak += 1
                    else:
                        break
                else:
                    if streak <= 0:
                        streak -= 1
                    else:
                        break

                rows.append({"game_id": row["game_id"], "team": team})
                continue

            # Point differential momentum
            margins        = past["point_diff"].values
            avg_margin     = float(margins.mean())
            margin_trend   = float(margins[-1] - margins[0]) if len(margins) > 1 else 0.0
            blowout_wins   = float((margins >= 14).mean())
            blowout_losses = float((margins <= -14).mean())

            # Coach win rate — last 10 games for this coach
            current_coach = row.get("coach")
            if current_coach and not pd.isna(current_coach):
                coach_past = coach_history[
                    (coach_history["coach"] == current_coach) &
                    ((coach_history["season"] < row["season"]) |
                     ((coach_history["season"] == row["season"]) &
                      (coach_history["week"]   <  row["week"])))
                ].tail(10)
                coach_win_rate = float(coach_past["won"].mean()) if not coach_past.empty else np.nan
                coach_games    = len(coach_past)
            else:
                coach_win_rate = np.nan
                coach_games    = 0

            rows.append({
                "game_id":             row["game_id"],
                "team":                team,
                "roll_passing_epa":    past["passing_epa"].mean(),
                "roll_rushing_epa":    past["rushing_epa"].mean(),
                "roll_passing_yards":  past["passing_yards"].mean(),
                "roll_rushing_yards":  past["rushing_yards"].mean(),
                "roll_passing_tds":    past["passing_tds"].mean(),
                "roll_interceptions":  past["passing_interceptions"].mean(),
                "roll_def_sacks":      past["def_sacks"].mean(),
                "roll_def_ints":       past["def_interceptions"].mean(),
                "roll_def_fumbles":    past["def_fumbles_forced"].mean(),
                "roll_def_tfl":        past["def_tackles_for_loss"].mean(),
                "roll_def_qb_hits":    past["def_qb_hits"].mean(),
                "roll_def_pass_def":   past["def_pass_defended"].mean(),
                "roll_turnovers":      (past["passing_interceptions"] + past["rushing_fumbles"] + past["receiving_fumbles"]).mean(),
                "roll_first_downs":    (past["passing_first_downs"] + past["rushing_first_downs"] + past["receiving_first_downs"]).mean(),
                "roll_penalty_yards":  past["penalty_yards"].mean(),
                "roll_avg_margin":     avg_margin,
                "roll_margin_trend":   margin_trend,
                "roll_blowout_wins":   blowout_wins,
                "roll_blowout_losses": blowout_losses,
                "roll_streak":         streak,
                "roll_coach_win_rate": coach_win_rate,
                "roll_coach_games":    coach_games,
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Building rolling team stats with momentum...")
    df = build_rolling_stats()
    save(df, "team_rolling_stats")
    print(f"Saved {len(df)} rows to team_rolling_stats")
    print(df.dropna(subset=["roll_avg_margin"]).head(3).to_string())