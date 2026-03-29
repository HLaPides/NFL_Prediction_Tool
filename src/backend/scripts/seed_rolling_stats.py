import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from data.sql_helpers import get_connection, save

ROLLING_WEEKS = 4


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


def _rolling_avg(grp: pd.DataFrame, col: str, up_to_idx: int) -> float:
    """4-week rolling average of col strictly before up_to_idx."""
    past = grp[grp.index < up_to_idx][col].dropna().tail(ROLLING_WEEKS)
    return float(past.mean()) if not past.empty else np.nan


def _get_starter(grp: pd.DataFrame, up_to_idx: int, volume_col: str) -> str | None:
    """Returns player_id with most volume strictly before up_to_idx."""
    past = grp[grp.index < up_to_idx]
    if past.empty:
        return None
    totals = past.groupby("player_id")[volume_col].sum()
    return totals.idxmax() if not totals.empty else None


def build_player_features() -> pd.DataFrame:
    print("Loading player stats...")
    df = _load_player_stats()

    # Load all schedule game_ids to iterate over
    with get_connection() as conn:
        sched = pd.read_sql("""
            SELECT game_id, season, week, home_team, away_team
            FROM schedules
            WHERE game_type = 'REG'
            AND home_score IS NOT NULL
            ORDER BY season, week
        """, conn)

    # Pre-group by position and team for fast lookup
    qbs  = df[df["position"] == "QB"].copy()
    wrs  = df[df["position"] == "WR"].copy()
    tes  = df[df["position"] == "TE"].copy()
    rbs  = df[df["position"] == "RB"].copy()

    rows = []
    total = len(sched)

    for i, (_, game) in enumerate(sched.iterrows(), 1):
        if i % 200 == 0:
            print(f"  {i}/{total} games...")

        season = game["season"]
        week   = game["week"]

        row = {"game_id": game["game_id"], "season": season, "week": week}

        for side, team in (("home", game["home_team"]), ("away", game["away_team"])):

            # ── QB ────────────────────────────────────────────────────────
            tqb = qbs[(qbs["team"] == team) &
                      ((qbs["season"] < season) |
                       ((qbs["season"] == season) & (qbs["week"] < week)))]

            if not tqb.empty:
                # Starter = most attempts in last 4 games
                starter_id = (tqb.tail(ROLLING_WEEKS)
                                 .groupby("player_id")["attempts"].sum()
                                 .idxmax())
                sqb = tqb[tqb["player_id"] == starter_id].tail(ROLLING_WEEKS)
                passing_epa = sqb["passing_epa"].mean()
                rushing_epa = sqb["rushing_epa"].fillna(0).mean()
                attempts    = sqb["attempts"].mean()
                qb_epa = (passing_epa + rushing_epa) / max(attempts, 1) if attempts > 0 else np.nan
            else:
                qb_epa = np.nan

            row[f"{side}_qb_epa"]    = round(qb_epa, 4) if not np.isnan(qb_epa) else np.nan
            row[f"{side}_qb_games"]  = len(tqb["week"].unique()) if not tqb.empty else 0

            # ── WR ────────────────────────────────────────────────────────
            twr = wrs[(wrs["team"] == team) &
                      ((wrs["season"] < season) |
                       ((wrs["season"] == season) & (wrs["week"] < week)))]

            if not twr.empty:
                starter_id = (twr.tail(ROLLING_WEEKS)
                                 .groupby("player_id")["targets"].sum()
                                 .idxmax())
                swr = twr[twr["player_id"] == starter_id].tail(ROLLING_WEEKS)
                row[f"{side}_wr_yards"]        = swr["receiving_yards"].mean()
                row[f"{side}_wr_target_share"] = swr["target_share"].mean()
            else:
                row[f"{side}_wr_yards"]        = np.nan
                row[f"{side}_wr_target_share"] = np.nan

            # ── TE ────────────────────────────────────────────────────────
            tte = tes[(tes["team"] == team) &
                      ((tes["season"] < season) |
                       ((tes["season"] == season) & (tes["week"] < week)))]

            if not tte.empty:
                starter_id = (tte.tail(ROLLING_WEEKS)
                                 .groupby("player_id")["targets"].sum()
                                 .idxmax())
                ste = tte[tte["player_id"] == starter_id].tail(ROLLING_WEEKS)
                row[f"{side}_te_yards"]        = ste["receiving_yards"].mean()
                row[f"{side}_te_target_share"] = ste["target_share"].mean()
            else:
                row[f"{side}_te_yards"]        = np.nan
                row[f"{side}_te_target_share"] = np.nan

            # ── RB ────────────────────────────────────────────────────────
            trb = rbs[(rbs["team"] == team) &
                      ((rbs["season"] < season) |
                       ((rbs["season"] == season) & (rbs["week"] < week)))]

            if not trb.empty:
                starter_id = (trb.tail(ROLLING_WEEKS)
                                 .groupby("player_id")["carries"].sum()
                                 .idxmax())
                srb = trb[trb["player_id"] == starter_id].tail(ROLLING_WEEKS)
                row[f"{side}_rb_rush_yards"] = srb["rushing_yards"].mean()
                row[f"{side}_rb_rec_yards"]  = srb["receiving_yards"].mean()
                row[f"{side}_rb_carry_share"] = srb["carries"].sum() / max(
                    trb.tail(ROLLING_WEEKS)["carries"].sum(), 1)
            else:
                row[f"{side}_rb_rush_yards"]  = np.nan
                row[f"{side}_rb_rec_yards"]   = np.nan
                row[f"{side}_rb_carry_share"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Building rolling player features...")
    df = build_player_features()
    save(df, "player_features")
    print(f"Saved {len(df)} rows to player_features")
    print(df.dropna(subset=["home_qb_epa"]).head(3).to_string())