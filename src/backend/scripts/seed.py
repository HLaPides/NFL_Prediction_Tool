import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from data.sql_helpers import get_connection, save

ROLLING_WEEKS     = 4
UNLIKELY_STATUSES = {"Out", "Doubtful"}


# ══════════════════════════════════════════════════════════════════════════════
# Rolling team stats
# ══════════════════════════════════════════════════════════════════════════════

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

    stats["point_diff"] = np.where(
        stats["team"] == stats["home_team"],
        stats["home_score"] - stats["away_score"],
        stats["away_score"] - stats["home_score"]
    )
    stats["won"] = (stats["point_diff"] > 0).astype(int)
    coach_history = stats[["season", "week", "team", "coach", "won"]].copy()

    rows = []
    for team, grp in stats.groupby("team"):
        grp = grp.sort_values(["season", "week"]).reset_index(drop=True)

        for _, row in grp.iterrows():
            past = grp[
                (grp["season"] < row["season"]) |
                ((grp["season"] == row["season"]) & (grp["week"] < row["week"]))
            ].tail(ROLLING_WEEKS)

            if past.empty:
                rows.append({"game_id": row["game_id"], "team": team})
                continue

            # Consecutive win/loss streak
            margins    = past["point_diff"].values
            wins_losses = (margins > 0).astype(int)
            streak = 0
            for result in reversed(wins_losses):
                if result == 1:
                    if streak >= 0: streak += 1
                    else: break
                else:
                    if streak <= 0: streak -= 1
                    else: break

            avg_margin     = float(margins.mean())
            margin_trend   = float(margins[-1] - margins[0]) if len(margins) > 1 else 0.0
            blowout_wins   = float((margins >= 14).mean())
            blowout_losses = float((margins <= -14).mean())

            current_coach = row.get("coach")
            if current_coach and not pd.isna(current_coach):
                coach_past = coach_history[
                    (coach_history["coach"] == current_coach) &
                    ((coach_history["season"] < row["season"]) |
                     ((coach_history["season"] == row["season"]) &
                      (coach_history["week"]   < row["week"])))
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
                "roll_turnovers":      (past["passing_interceptions"] +
                                        past["rushing_fumbles"] +
                                        past["receiving_fumbles"]).mean(),
                "roll_first_downs":    (past["passing_first_downs"] +
                                        past["rushing_first_downs"] +
                                        past["receiving_first_downs"]).mean(),
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


# ══════════════════════════════════════════════════════════════════════════════
# Player features
# ══════════════════════════════════════════════════════════════════════════════

def _load_player_stats() -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql("""
            SELECT player_id, player_name, position, team, season, week, game_id,
                   attempts, passing_epa, rushing_epa,
                   receiving_yards, targets, target_share,
                   rushing_yards, carries
            FROM player_stats
            WHERE season_type = 'REG'
            AND position IN ('QB', 'WR', 'TE', 'RB')
            ORDER BY season, week
        """, conn)


def _load_injuries() -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql("""
            SELECT season, week, team, gsis_id, position, report_status
            FROM injuries_historical
            WHERE game_type = 'REG'
            AND report_status IN ('Out', 'Doubtful')
        """, conn)


def _get_injured_ids(injuries: pd.DataFrame, team: str,
                     season: int, week: int) -> set:
    mask = (
        (injuries["team"]   == team) &
        (injuries["season"] == season) &
        (injuries["week"]   == week)
    )
    return set(injuries[mask]["gsis_id"].dropna().tolist())


def _rolling_starter(df: pd.DataFrame, activity_col: str,
                     injured: set) -> tuple:
    """Returns (starter_id, healthy_df) for a position group."""
    healthy = df[~df["player_id"].isin(injured)]
    use     = healthy if not healthy.empty else df
    if use.empty:
        return None, use
    starter = (use.tail(ROLLING_WEEKS)
                  .groupby("player_id")[activity_col].sum()
                  .idxmax())
    return starter, use


def build_player_features() -> pd.DataFrame:
    print("  Loading player stats...")
    df = _load_player_stats()

    print("  Loading injury data...")
    injuries = _load_injuries()

    with get_connection() as conn:
        sched = pd.read_sql("""
            SELECT game_id, season, week, home_team, away_team
            FROM schedules
            WHERE game_type IN ('REG', 'WC', 'DIV', 'CON', 'SB')
            AND home_score IS NOT NULL
            ORDER BY season, week
        """, conn)

    qbs = df[df["position"] == "QB"]
    wrs = df[df["position"] == "WR"]
    tes = df[df["position"] == "TE"]
    rbs = df[df["position"] == "RB"]

    rows  = []
    total = len(sched)

    for i, (_, game) in enumerate(sched.iterrows(), 1):
        if i % 200 == 0:
            print(f"  {i}/{total} games...")

        season = game["season"]
        week   = game["week"]
        row    = {"game_id": game["game_id"], "season": season, "week": week}

        for side, team in (("home", game["home_team"]), ("away", game["away_team"])):
            injured = _get_injured_ids(injuries, team, season, week)

            def past_games(pos_df):
                return pos_df[
                    (pos_df["team"] == team) &
                    ((pos_df["season"] < season) |
                     ((pos_df["season"] == season) & (pos_df["week"] < week)))
                ]

            # QB
            tqb = past_games(qbs)
            starter, use = _rolling_starter(tqb, "attempts", injured)
            if starter and not use.empty:
                sqb = use[use["player_id"] == starter].tail(ROLLING_WEEKS)
                att = sqb["attempts"].mean()
                epa = (sqb["passing_epa"].mean() + sqb["rushing_epa"].fillna(0).mean())
                row[f"{side}_qb_epa"]     = round(epa / max(att, 1), 4) if att > 0 else np.nan
                row[f"{side}_qb_games"]   = len(use["week"].unique())
            else:
                row[f"{side}_qb_epa"]     = np.nan
                row[f"{side}_qb_games"]   = 0
            orig_starter = (tqb.tail(ROLLING_WEEKS).groupby("player_id")["attempts"].sum().idxmax()
                            if not tqb.empty else None)
            row[f"{side}_qb_injured"] = int(orig_starter in injured) if orig_starter else 0

            # WR
            twr = past_games(wrs)
            starter, use = _rolling_starter(twr, "targets", injured)
            if starter and not use.empty:
                swr = use[use["player_id"] == starter].tail(ROLLING_WEEKS)
                row[f"{side}_wr_yards"]        = swr["receiving_yards"].mean()
                row[f"{side}_wr_target_share"] = swr["target_share"].mean()
            else:
                row[f"{side}_wr_yards"]        = np.nan
                row[f"{side}_wr_target_share"] = np.nan
            orig_starter = (twr.tail(ROLLING_WEEKS).groupby("player_id")["targets"].sum().idxmax()
                            if not twr.empty else None)
            row[f"{side}_wr_injured"] = int(orig_starter in injured) if orig_starter else 0

            # TE
            tte = past_games(tes)
            starter, use = _rolling_starter(tte, "targets", injured)
            if starter and not use.empty:
                ste = use[use["player_id"] == starter].tail(ROLLING_WEEKS)
                row[f"{side}_te_yards"]        = ste["receiving_yards"].mean()
                row[f"{side}_te_target_share"] = ste["target_share"].mean()
            else:
                row[f"{side}_te_yards"]        = np.nan
                row[f"{side}_te_target_share"] = np.nan
            orig_starter = (tte.tail(ROLLING_WEEKS).groupby("player_id")["targets"].sum().idxmax()
                            if not tte.empty else None)
            row[f"{side}_te_injured"] = int(orig_starter in injured) if orig_starter else 0

            # RB
            trb = past_games(rbs)
            starter, use = _rolling_starter(trb, "carries", injured)
            if starter and not use.empty:
                srb = use[use["player_id"] == starter].tail(ROLLING_WEEKS)
                row[f"{side}_rb_rush_yards"]  = srb["rushing_yards"].mean()
                row[f"{side}_rb_rec_yards"]   = srb["receiving_yards"].mean()
                row[f"{side}_rb_carry_share"] = srb["carries"].sum() / max(
                    use.tail(ROLLING_WEEKS)["carries"].sum(), 1)
            else:
                row[f"{side}_rb_rush_yards"]  = np.nan
                row[f"{side}_rb_rec_yards"]   = np.nan
                row[f"{side}_rb_carry_share"] = np.nan
            orig_starter = (trb.tail(ROLLING_WEEKS).groupby("player_id")["carries"].sum().idxmax()
                            if not trb.empty else None)
            row[f"{side}_rb_injured"] = int(orig_starter in injured) if orig_starter else 0

        rows.append(row)

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point — run both seeds
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Seed DB tables from scratch")
    parser.add_argument("--rolling",  action="store_true", help="Rebuild team_rolling_stats")
    parser.add_argument("--players",  action="store_true", help="Rebuild player_features")
    parser.add_argument("--all",      action="store_true", help="Rebuild both (default)")
    args = parser.parse_args()

    run_rolling = args.all or args.rolling or not (args.rolling or args.players)
    run_players = args.all or args.players or not (args.rolling or args.players)

    if run_rolling:
        print("Building rolling team stats...")
        df = build_rolling_stats()
        save(df, "team_rolling_stats")
        print(f"Saved {len(df)} rows to team_rolling_stats")

    if run_players:
        print("\nBuilding player features...")
        df = build_player_features()
        save(df, "player_features")
        print(f"Saved {len(df)} rows to player_features")