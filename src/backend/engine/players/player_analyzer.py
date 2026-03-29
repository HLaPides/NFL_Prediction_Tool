import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from engine.players.qb_evaluator import qb_evaluator
from engine.players.wr_evaluator import wr_evaluator
from engine.players.te_evaluator import te_evaluator
from engine.players.rb_evaluator import rb_evaluator
from engine.players.defense_evaluator import defense_evaluator
import pandas as pd


class player_analyzer:
    """
    Usage:
        analyzer = PlayerAnalyzer()
        result   = analyzer.evaluate_matchup("KC", "BUF", week=10, season=2025)
    """

    def __init__(self):
        self.qb      = qb_evaluator()
        self.wr      = wr_evaluator()
        self.te      = te_evaluator()
        self.rb      = rb_evaluator()
        self.defense = defense_evaluator()

    #Evaluates qb, we, te, rb, and defense
    def evaluate_team(self, team: str, up_to_week: int, season: int,
                  injuries: pd.DataFrame = None) -> dict:

        # Identify starters first so we can check injury status
        qb = self.qb.get_team_qb(team, up_to_week, season)
        wr = self.wr.get_team_wr1(team, up_to_week, season)
        te = self.te.get_team_te1(team, up_to_week, season)
        rb = self.rb.get_team_rb1(team, up_to_week, season)

        # If injury report provided, substitute any out/doubtful starters
        if injuries is not None and not injuries.empty:
            from data.injury_getter import get_qb_status, UNLIKELY_STATUSES

            qb_name   = qb.get("player_name", "")
            qb_status = get_qb_status(injuries, qb_name)
            if qb_status in UNLIKELY_STATUSES:
                qb = self.qb.get_team_qb(team, up_to_week, season,
                                        exclude_id=qb.get("player_id"))
                qb["injury_sub"] = True

            # WR
            wr_name = wr.get("player_name", "")
            wr_inj  = injuries[injuries["player_name"].str.contains(
                wr_name.split(".")[-1], case=False, na=False
            )] if wr_name else pd.DataFrame()
            if not wr_inj.empty and wr_inj.iloc[0]["status"] in UNLIKELY_STATUSES:
                wr = self.wr.get_team_wr1(team, up_to_week, season,
                                        exclude_id=wr.get("player_id"))
                wr["injury_sub"] = True

            # TE
            te_name = te.get("player_name", "")
            te_inj  = injuries[injuries["player_name"].str.contains(
                te_name.split(".")[-1], case=False, na=False
            )] if te_name else pd.DataFrame()
            if not te_inj.empty and te_inj.iloc[0]["status"] in UNLIKELY_STATUSES:
                te = self.te.get_team_te1(team, up_to_week, season,
                                        exclude_id=te.get("player_id"))
                te["injury_sub"] = True

            # RB
            rb_name = rb.get("player_name", "")
            rb_inj  = injuries[injuries["player_name"].str.contains(
                rb_name.split(".")[-1], case=False, na=False
            )] if rb_name else pd.DataFrame()
            if not rb_inj.empty and rb_inj.iloc[0]["status"] in UNLIKELY_STATUSES:
                rb = self.rb.get_team_rb1(team, up_to_week, season,
                                        exclude_id=rb.get("player_id"))
                rb["injury_sub"] = True

        df = self.defense.analyze(team, up_to_week, season)
        return {
            "team":    team,
            "offense": {"qb": qb, "wr": wr, "te": te, "rb": rb},
            "defense": df,
        }


    def evaluate_matchup(self, home_team: str, away_team: str,
                        week: int, season: int,
                        injuries: pd.DataFrame = None) -> dict:
        return {
            "home": self.evaluate_team(home_team, week, season, injuries),
            "away": self.evaluate_team(away_team, week, season, injuries),
        }


if __name__ == "__main__":
    analyzer = player_analyzer()

    print("KC vs BUF — week 10 of 2025\n")
    result = analyzer.evaluate_matchup("KC", "BUF", week=10, season=2025)

    for side in ["home", "away"]:
        team = result[side]["team"]
        off  = result[side]["offense"]
        defe = result[side]["defense"]

        print(f"  {team} offense:")
        print(f"    QB  {off['qb'].get('player_name','?'):20s}  EPA/att: {off['qb']['epa_per_att']:+.4f}")
        print(f"    WR1 {off['wr'].get('player_name','?'):20s}  Yards: {off['wr']['receiving_yards']:.1f}  Share: {off['wr']['target_share']:.3f}")
        print(f"    TE1 {off['te'].get('player_name','?'):20s}  Yards: {off['te']['receiving_yards']:.1f}  Share: {off['te']['target_share']:.3f}")
        print(f"    RB1 {off['rb'].get('player_name','?'):20s}  Rush: {off['rb']['rush_yards']:.1f}  Share: {off['rb']['carry_share']:.3f}")
        print(f"  {team} defense:")
        print(f"    Pts allowed: {defe['pts_allowed']:.1f}  "
              f"Pass: {defe['pass_yards_allowed']:.1f}  "
              f"Rush: {defe['rush_yards_allowed']:.1f}  "
              f"Sacks: {defe['sacks']:.2f}  "
              f"TO: {defe['turnovers_forced']:.2f}")
        print()