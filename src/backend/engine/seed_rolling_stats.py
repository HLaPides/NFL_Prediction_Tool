import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from data.sql_helpers import get_connection, save

ROLLING_WEEKS = 4


def build_rolling_stats() -> pd.DataFrame:
    """
    For each team/game, computes a 4-week rolling average of key stats
    using only games BEFORE that week (no leakage).
    Output: one row per team per game, joinable via game_id.
    """
    with get_connection() as conn:
        df = pd.read_sql("""
            SELECT season, week, team, game_id,
                   passing_epa, rushing_epa,
                   passing_yards, rushing_yards,
                   passing_tds, rushing_tds,
                   passing_interceptions,
                   def_sacks, def_interceptions,
                   def_fumbles_forced, def_tackles_for_loss,
                   def_qb_hits, def_pass_defended
            FROM team_stats_weekly
            WHERE season_type = 'REG'
            ORDER BY season, week
        """, conn)

    rows = []

    for team, grp in df.groupby("team"):
        grp = grp.sort_values(["season", "week"]).reset_index(drop=True)

        for i, row in grp.iterrows():
            # Only use games strictly before this one
            past = grp[
                (grp["season"] < row["season"]) |
                ((grp["season"] == row["season"]) & (grp["week"] < row["week"]))
            ].tail(ROLLING_WEEKS)

            if past.empty:
                # No prior data — use NaN, model will fill with 0
                rows.append({"game_id": row["game_id"], "team": team})
                continue

            rows.append({
                "game_id":             row["game_id"],
                "team":                team,
                "roll_passing_epa":    past["passing_epa"].mean(),
                "roll_rushing_epa":    past["rushing_epa"].mean(),
                "roll_passing_yards":  past["passing_yards"].mean(),
                "roll_rushing_yards":  past["rushing_yards"].mean(),
                "roll_passing_tds":    past["passing_tds"].mean(),
                "roll_rushing_tds":    past["rushing_tds"].mean(),
                "roll_interceptions":  past["passing_interceptions"].mean(),
                "roll_def_sacks":      past["def_sacks"].mean(),
                "roll_def_ints":       past["def_interceptions"].mean(),
                "roll_def_fumbles":    past["def_fumbles_forced"].mean(),
                "roll_def_tfl":        past["def_tackles_for_loss"].mean(),
                "roll_def_qb_hits":    past["def_qb_hits"].mean(),
                "roll_def_pass_def":   past["def_pass_defended"].mean(),
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Building rolling team stats...")
    df = build_rolling_stats()
    save(df, "team_rolling_stats")
    print(f"Saved {len(df)} rows to team_rolling_stats")
    print(df.dropna().head(3).to_string())