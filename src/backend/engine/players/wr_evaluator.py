import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from data.sql_helpers import get_connection


MIN_TARGETS     = 2       # minimum targets to count a game
LEAGUE_AVG_YARDS  = 45.0  # league average receiving yards/game for WR
LEAGUE_AVG_SHARE  = 0.15  # league average target share for WR
ROLLING_WEIGHT  = [0.4, 0.3, 0.2, 0.1]
CAREER_BLEND    = {0: 1.0, 1: 0.7, 2: 0.5, 3: 0.3}

def _load_wr_stats(table: str) -> pd.DataFrame:
    with get_connection() as conn:
        df = pd.read_sql(f"""
            SELECT player_id, player_name, season, week, team,
                   targets, receiving_yards, receiving_tds, target_share
            FROM {table}
            WHERE position = 'WR'
            AND targets >= {MIN_TARGETS}
        """, conn)
    df["target_share"] = df["target_share"].fillna(0)
    return df

def _weighted_avg(values: np.ndarray) -> float:
    last4 = values[::-1][:4]
    w     = np.array(ROLLING_WEIGHT[:len(last4)])
    return float(np.dot(last4, w / w.sum()))

def _career_avg(player_id: str, historical: pd.DataFrame) -> dict | None:
    games = historical[historical["player_id"] == player_id]
    if games.empty:
        return None
    games = games.sort_values(["season", "week"])
    w = np.arange(1, len(games)+1 ,dtype=float)
    w = w / w.sum()

    return {
        "receiving_yards": float(np.dot(games["receiving_yards"].values, w)),
        "target_share":    float(np.dot(games["target_share"].values, w)),
    }


class wr_evaluator:
        """
        Rookie, 0 games  → league averages
        Rookie, 1+ games → pure rolling average
        Veteran, 0 games → career averages
        Veteran, 1-3 games → blended career + current
        4+ games         → pure rolling average
        """

        def __init__(self):
            self.historical = _load_wr_stats("player_stats")
            self.current    = _load_wr_stats("player_stats_current")

        def evaluate(self, player_id: str, up_to_week: int, season: int) -> dict:
            curr = self.current[
                (self.current["player_id"] == player_id) &
                (self.current["season"]    == season) &
                (self.current["week"]      <  up_to_week)
            ].sort_values("week")

            career = _career_avg(player_id, self.historical)
            n      = len(curr)

            if n >= 4:
                return self._result(player_id,
                                    _weighted_avg(curr["receiving_yards"].values),
                                    _weighted_avg(curr["target_share"].values),
                                    "rolling", n)

            if career is None:
                if n > 0:
                    return self._result(player_id,
                                        _weighted_avg(curr["receiving_yards"].values),
                                        _weighted_avg(curr["target_share"].values),
                                        "rookie", n)
                return self._result(player_id, LEAGUE_AVG_YARDS, LEAGUE_AVG_SHARE, "rookie", 0)

            if n == 0:
                return self._result(player_id,
                                    career["receiving_yards"],
                                    career["target_share"],
                                    "career", 0)

            cw = CAREER_BLEND[n]
            yards = cw * career["receiving_yards"] + (1 - cw) * _weighted_avg(curr["receiving_yards"].values)
            share = cw * career["target_share"]    + (1 - cw) * _weighted_avg(curr["target_share"].values)
            return self._result(player_id, yards, share, "blended", n)
        
        def get_team_wr1(self, team: str, up_to_week: int, season: int) -> dict:
            curr = self.current[
                (self.current["team"]   == team) &
                (self.current["season"] == season) &
                (self.current["week"]   <  up_to_week)
            ]

            if not curr.empty:
                last_team = (self.current[self.current["season"] == season]
                            .sort_values("week")
                            .groupby("player_id")["team"]
                            .last())
                valid_pids = last_team[last_team == team].index
                curr = curr[curr["player_id"].isin(valid_pids)]

            if not curr.empty:
                pid, name = curr.groupby(["player_id", "player_name"])["target_share"].mean().idxmax()
                result    = self.evaluate(pid, up_to_week, season)
                result["player_name"] = name
                return result
            
            hist = self.historical[self.historical["team"] == team]
            if hist.empty:
                return self._result(None, LEAGUE_AVG_YARDS, LEAGUE_AVG_SHARE,
                                    "no_data", 0, player_name="Unknown")

            last_season = hist["season"].max()
            last        = hist[hist["season"] == last_season]
            pid, name   = last.groupby(["player_id", "player_name"])["target_share"].mean().idxmax()
            result      = self.evaluate(pid, up_to_week, season)
            result["player_name"] = name
            return result

        @staticmethod
        def _result(player_id, yards, share, source, games, player_name=None) -> dict:
            r = {
                "player_id":       player_id,
                "receiving_yards": round(float(yards), 2),
                "target_share":    round(float(share), 4),
                "source":          source,
                "games_used":      games,
            }
            if player_name is not None:
                r["player_name"] = player_name
            return r

if __name__ == "__main__":
    ev = wr_evaluator()

    print("WR1 ratings going into week 17 of 2025:\n")
    for team in ["JAX", "MIA", "LV", "NE", "CHI", "LA", "SEA"]:
        r = ev.get_team_wr1(team, up_to_week=10, season=2025)
        print(f"  {team:4s}  {r.get('player_name', '?'):20s}  "
              f"Yards: {r['receiving_yards']:6.1f}  "
              f"Share: {r['target_share']:.3f}  "
              f"({r['source']}, {r['games_used']} games)")

    print("\nWeek 1 fallback (career avg):\n")
    for team in ["KC", "PHI", "BUF"]:
        r = ev.get_team_wr1(team, up_to_week=1, season=2025)
        print(f"  {team:4s}  {r.get('player_name', '?'):20s}  "
              f"Yards: {r['receiving_yards']:6.1f}  "
              f"Share: {r['target_share']:.3f}  "
              f"({r['source']}, {r['games_used']} games)")