import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from data.sql_helpers import get_connection

ROLLING_WEIGHT = [0.4, 0.3, 0.2, 0.1]

LEAGUE_AVG = {
    "pts_allowed":        22.0,
    "pass_yards_allowed": 230.0,
    "rush_yards_allowed": 115.0,
    "sacks":              2.5,
    "turnovers_forced":   1.4,
}



def _load_historical_team_stats() -> pd.DataFrame:
    """Season-level team stats — one row per team per season."""
    with get_connection() as conn:
        df = pd.read_sql("""
            SELECT season, team,
                   def_sacks,
                   def_interceptions,
                   def_fumbles_forced as def_fumbles
            FROM team_stats
        """, conn)
    df["turnovers_forced"] = df["def_interceptions"] + df["def_fumbles"]
    return df


def _load_current_team_stats() -> pd.DataFrame:
    """Weekly team stats for the current season."""
    with get_connection() as conn:
        df = pd.read_sql("""
            SELECT season, week, team, season_type,
                   opponent_team,
                   passing_yards,
                   rushing_yards,
                   def_sacks,
                   def_interceptions,
                   def_fumbles
            FROM team_stats_current
            WHERE season_type = 'REG'
        """, conn)
    df["turnovers_forced"] = df["def_interceptions"] + df["def_fumbles"]
    return df

def _weighted_avg(values: np.ndarray) -> float:
    last4 = values[::-1][:4]
    w     = np.array(ROLLING_WEIGHT[:len(last4)])
    return float(np.dot(last4, w / w.sum()))

class defense_evaluator:

    def __init__(self):
        self.historical = _load_historical_team_stats()
        self.current    = _load_current_team_stats()
        self._build_defense_view()

    def _build_defense_view(self) -> None:
        opp = self.current[["season", "week", "team", "passing_yards",
                             "rushing_yards"]].rename(columns={
            "team":          "opponent_team",
            "passing_yards": "pass_yards_allowed",
            "rushing_yards": "rush_yards_allowed",
        })
        self.current = self.current.merge(opp, on=["season", "week", "opponent_team"], how="left")

        with get_connection() as conn:
            sched = pd.read_sql("""
                SELECT season, week, home_team, away_team,
                       home_score, away_score
                FROM schedules_current
                WHERE home_score IS NOT NULL
            """, conn)

        home = sched[["season", "week", "home_team", "away_score"]].rename(
            columns={"home_team": "team", "away_score": "pts_allowed"}
        )
        away = sched[["season", "week", "away_team", "home_score"]].rename(
            columns={"away_team": "team", "home_score": "pts_allowed"}
        )
        pts = pd.concat([home, away])
        self.current = self.current.merge(pts, on=["season", "week", "team"], how="left")

    def analyze(self, team: str, up_to_week: int, season: int) -> dict:
        curr = self.current[
            (self.current["team"]   == team) &
            (self.current["season"] == season) &
            (self.current["week"]   <  up_to_week)
        ].sort_values("week")

        if curr.empty:
            hist = self.historical[self.historical["team"] == team]
            if hist.empty:
                return self._result(team, **LEAGUE_AVG, source="league_avg")

            last = hist[hist["season"] == hist["season"].max()]
            return self._result(
                team,
                pts_allowed        = LEAGUE_AVG["pts_allowed"], 
                pass_yards_allowed = LEAGUE_AVG["pass_yards_allowed"],
                rush_yards_allowed = LEAGUE_AVG["rush_yards_allowed"],
                sacks              = float(last["def_sacks"].mean()),
                turnovers_forced   = float(last["turnovers_forced"].mean()),
                source             = "career",
                games_used         = 0,
            )

        n = len(curr)
        return self._result(
            team,
            pts_allowed        = _weighted_avg(curr["pts_allowed"].values),
            pass_yards_allowed = _weighted_avg(curr["pass_yards_allowed"].values),
            rush_yards_allowed = _weighted_avg(curr["rush_yards_allowed"].values),
            sacks              = _weighted_avg(curr["def_sacks"].values),
            turnovers_forced   = _weighted_avg(curr["turnovers_forced"].values),
            source             = "rolling" if n >= 4 else "limited",
            games_used         = n,
        )

    @staticmethod
    def _result(team, pts_allowed, pass_yards_allowed, rush_yards_allowed,
                sacks, turnovers_forced, source, games_used=0) -> dict:
        return {
            "team":               team,
            "pts_allowed":        round(float(pts_allowed),        1),
            "pass_yards_allowed": round(float(pass_yards_allowed), 1),
            "rush_yards_allowed": round(float(rush_yards_allowed), 1),
            "sacks":              round(float(sacks),              2),
            "turnovers_forced":   round(float(turnovers_forced),   2),
            "source":             source,
            "games_used":         games_used,
        }


if __name__ == "__main__":
    da = defense_evaluator()

    print("Defensive ratings going into week 10 of 2025:\n")
    for team in ["HOU", "PHI", "BUF", "NE", "CHI", "DEN", "SEA"]:
        r = da.analyze(team, up_to_week=17, season=2025)
        print(f"  {team:4s}  "
              f"Pts: {r['pts_allowed']:5.1f}  "
              f"Pass: {r['pass_yards_allowed']:6.1f}  "
              f"Rush: {r['rush_yards_allowed']:5.1f}  "
              f"Sacks: {r['sacks']:.2f}  "
              f"TO: {r['turnovers_forced']:.2f}  "
              f"({r['source']}, {r['games_used']} games)")

    print("\nWeek 1 fallback:\n")
    for team in ["KC", "PHI", "BUF"]:
        r = da.analyze(team, up_to_week=1, season=2025)
        print(f"  {team:4s}  "
              f"Pts: {r['pts_allowed']:5.1f}  "
              f"Pass: {r['pass_yards_allowed']:6.1f}  "
              f"Rush: {r['rush_yards_allowed']:5.1f}  "
              f"({r['source']}, {r['games_used']} games)")