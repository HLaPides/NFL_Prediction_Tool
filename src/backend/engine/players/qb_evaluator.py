import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from data.sql_helpers import get_connection

MIN_ATTEMPTS   = 10
LEAGUE_AVG_EPA = 0.07
ROLLING_WEIGHT = [0.4, 0.3, 0.2, 0.1]
CAREER_BLEND   = {0: 1.0, 1: 0.7, 2: 0.5, 3: 0.3}


def _load_qb_stats(table: str) -> pd.DataFrame:
    with get_connection() as conn:
        df = pd.read_sql(f"""
            SELECT player_id, player_name, season, week, team,
                   attempts, passing_epa, rushing_epa
            FROM {table}
            WHERE position = 'QB'
            AND attempts >= {MIN_ATTEMPTS}
        """, conn)
    df["rushing_epa"] = df["rushing_epa"].fillna(0)
    df["epa_per_att"] = (df["passing_epa"] + df["rushing_epa"]) / df["attempts"]
    return df


def _weighted_avg(values: np.ndarray) -> float:
    last4 = values[::-1][:4]
    w     = np.array(ROLLING_WEIGHT[:len(last4)])
    return float(np.dot(last4, w / w.sum()))


def _career_avg(player_id: str, historical: pd.DataFrame) -> float | None:
    games = historical[historical["player_id"] == player_id]
    if games.empty:
        return None
    games = games.sort_values(["season", "week"])
    w = np.arange(1, len(games) + 1, dtype=float)
    return float(np.dot(games["epa_per_att"].values, w / w.sum()))


class qb_evaluator:
    #Evaluates QB quality as a weighted EPA/attempt metric.
    def __init__(self):
        self.historical = _load_qb_stats("player_stats")
        self.current    = _load_qb_stats("player_stats_current")

    def evaluate(self, player_id: str, up_to_week: int, season: int) -> dict:
        curr   = self.current[
            (self.current["player_id"] == player_id) &
            (self.current["season"]    == season) &
            (self.current["week"]      <  up_to_week)
        ].sort_values("week")

        career = _career_avg(player_id, self.historical, before_season=season)
        n      = len(curr)

        if n >= 4:
            return self._result(player_id, _weighted_avg(curr["epa_per_att"].values), "rolling", n)

        if career is None:
            epa = _weighted_avg(curr["epa_per_att"].values) if n > 0 else LEAGUE_AVG_EPA
            return self._result(player_id, epa, "rookie", n)

        if n == 0:
            return self._result(player_id, career, "career", 0)

        cw  = CAREER_BLEND[n]
        epa = cw * career + (1 - cw) * _weighted_avg(curr["epa_per_att"].values)
        return self._result(player_id, epa, "blended", n)

    def get_team_qb(self, team: str, up_to_week: int, season: int) -> dict:
        curr = self.current[
            (self.current["team"]   == team) &
            (self.current["season"] == season) &
            (self.current["week"]   <  up_to_week)
        ]

        if not curr.empty:
            pid, name = curr.groupby(["player_id", "player_name"])["attempts"].sum().idxmax()
            result    = self.evaluate(pid, up_to_week, season)
            result["player_name"] = name
            return result

        # No current season data — fall back to last season's starter
        hist = self.historical[self.historical["team"] == team]
        if hist.empty:
            return self._result(None, LEAGUE_AVG_EPA, "no_data", 0, player_name="Unknown")

        last_season = hist["season"].max()
        last        = hist[hist["season"] == last_season]
        pid, name   = last.groupby(["player_id", "player_name"])["attempts"].sum().idxmax()
        result      = self.evaluate(pid, up_to_week, season)
        result["player_name"] = name
        return result

    @staticmethod
    def _result(player_id, epa, source, games, player_name=None) -> dict:
        r = {
            "player_id":   player_id,
            "epa_per_att": round(float(epa), 4),
            "source":      source,
            "games_used":  games,
        }
        if player_name is not None:
            r["player_name"] = player_name
        return r


if __name__ == "__main__":
    ev = qb_evaluator()

    print("QB ratings going into week 16 of 2025:\n")
    for team in ["JAX", "MIA", "NYG", "NE", "CHI", "DEN", "SEA"]:
        r = ev.get_team_qb(team, up_to_week=16, season=2025)
        print(f"  {team:4s}  {r.get('player_name', '?'):20s}  "
              f"EPA/att: {r['epa_per_att']:+.4f}  "
              f"({r['source']}, {r['games_used']} games)")

    print("\nWeek 1 fallback (career avg):\n")
    for team in ["JAX", "MIA", "NYJ"]:
        r = ev.get_team_qb(team, up_to_week=1, season=2025)
        print(f"  {team:4s}  {r.get('player_name', '?'):20s}  "
              f"EPA/att: {r['epa_per_att']:+.4f}  "
              f"({r['source']}, {r['games_used']} games)")