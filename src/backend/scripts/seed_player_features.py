import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from data.sql_helpers import get_connection, save

ROLLING_WEEKS    = 4
UNLIKELY_STATUSES = {"Out", "Doubtful"}


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
    """Returns player_ids listed as Out or Doubtful for this game."""
    mask = (
        (injuries["team"]   == team) &
        (injuries["season"] == season) &
        (injuries["week"]   == week)
    )
    return set(injuries[mask]["gsis_id"].dropna().tolist())


def build_player_features() -> pd.DataFrame:
    print("Loading player stats...")
    df = _load_player_stats()

    print("Loading injury data...")
    injuries = _load_injuries()

    with get_connection() as conn:
        sched = pd.read_sql("""
            SELECT game_id, season, week, home_team, away_team
            FROM schedules
            WHERE game_type IN ('REG', 'WC', 'DIV', 'CON', 'SB')
            AND home_score IS NOT NULL
            ORDER BY season, week
        """, conn)

    qbs = df[df["position"] == "QB"].copy()
    wrs = df[df["position"] == "WR"].copy()
    tes = df[df["position"] == "TE"].copy()
    rbs = df[df["position"] == "RB"].copy()

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

            # ── QB ────────────────────────────────────────────────────────
            tqb = qbs[
                (qbs["team"] == team) &
                ((qbs["season"] < season) |
                 ((qbs["season"] == season) & (qbs["week"] < week)))
            ]

            # Remove injured players
            tqb_healthy = tqb[~tqb["player_id"].isin(injured)]
            tqb_use     = tqb_healthy if not tqb_healthy.empty else tqb

            if not tqb_use.empty:
                starter_id = (tqb_use.tail(ROLLING_WEEKS)
                                     .groupby("player_id")["attempts"].sum()
                                     .idxmax())
                sqb        = tqb_use[tqb_use["player_id"] == starter_id].tail(ROLLING_WEEKS)
                passing_epa = sqb["passing_epa"].mean()
                rushing_epa = sqb["rushing_epa"].fillna(0).mean()
                attempts    = sqb["attempts"].mean()
                qb_epa      = (passing_epa + rushing_epa) / max(attempts, 1) if attempts > 0 else np.nan
            else:
                qb_epa = np.nan

            row[f"{side}_qb_epa"]   = round(qb_epa, 4) if not pd.isna(qb_epa) else np.nan
            row[f"{side}_qb_games"] = len(tqb_use["week"].unique()) if not tqb_use.empty else 0
            # Flag if the rolling starter was injured this week
            rolling_starter = (tqb.tail(ROLLING_WEEKS)
                                   .groupby("player_id")["attempts"].sum()
                                   .idxmax()) if not tqb.empty else None
            row[f"{side}_qb_injured"] = int(rolling_starter in injured) if rolling_starter else 0

            # ── WR ────────────────────────────────────────────────────────
            twr = wrs[
                (wrs["team"] == team) &
                ((wrs["season"] < season) |
                 ((wrs["season"] == season) & (wrs["week"] < week)))
            ]
            twr_healthy = twr[~twr["player_id"].isin(injured)]
            twr_use     = twr_healthy if not twr_healthy.empty else twr

            if not twr_use.empty:
                starter_id = (twr_use.tail(ROLLING_WEEKS)
                                     .groupby("player_id")["targets"].sum()
                                     .idxmax())
                swr = twr_use[twr_use["player_id"] == starter_id].tail(ROLLING_WEEKS)
                row[f"{side}_wr_yards"]        = swr["receiving_yards"].mean()
                row[f"{side}_wr_target_share"] = swr["target_share"].mean()
            else:
                row[f"{side}_wr_yards"]        = np.nan
                row[f"{side}_wr_target_share"] = np.nan

            wr_starter = (twr.tail(ROLLING_WEEKS)
                             .groupby("player_id")["targets"].sum()
                             .idxmax()) if not twr.empty else None
            row[f"{side}_wr_injured"] = int(wr_starter in injured) if wr_starter else 0

            # ── TE ────────────────────────────────────────────────────────
            tte = tes[
                (tes["team"] == team) &
                ((tes["season"] < season) |
                 ((tes["season"] == season) & (tes["week"] < week)))
            ]
            tte_healthy = tte[~tte["player_id"].isin(injured)]
            tte_use     = tte_healthy if not tte_healthy.empty else tte

            if not tte_use.empty:
                starter_id = (tte_use.tail(ROLLING_WEEKS)
                                     .groupby("player_id")["targets"].sum()
                                     .idxmax())
                ste = tte_use[tte_use["player_id"] == starter_id].tail(ROLLING_WEEKS)
                row[f"{side}_te_yards"]        = ste["receiving_yards"].mean()
                row[f"{side}_te_target_share"] = ste["target_share"].mean()
            else:
                row[f"{side}_te_yards"]        = np.nan
                row[f"{side}_te_target_share"] = np.nan

            te_starter = (tte.tail(ROLLING_WEEKS)
                             .groupby("player_id")["targets"].sum()
                             .idxmax()) if not tte.empty else None
            row[f"{side}_te_injured"] = int(te_starter in injured) if te_starter else 0

            # ── RB ────────────────────────────────────────────────────────
            trb = rbs[
                (rbs["team"] == team) &
                ((rbs["season"] < season) |
                 ((rbs["season"] == season) & (rbs["week"] < week)))
            ]
            trb_healthy = trb[~trb["player_id"].isin(injured)]
            trb_use     = trb_healthy if not trb_healthy.empty else trb

            if not trb_use.empty:
                starter_id = (trb_use.tail(ROLLING_WEEKS)
                                     .groupby("player_id")["carries"].sum()
                                     .idxmax())
                srb = trb_use[trb_use["player_id"] == starter_id].tail(ROLLING_WEEKS)
                row[f"{side}_rb_rush_yards"]  = srb["rushing_yards"].mean()
                row[f"{side}_rb_rec_yards"]   = srb["receiving_yards"].mean()
                row[f"{side}_rb_carry_share"] = srb["carries"].sum() / max(
                    trb_use.tail(ROLLING_WEEKS)["carries"].sum(), 1)
            else:
                row[f"{side}_rb_rush_yards"]  = np.nan
                row[f"{side}_rb_rec_yards"]   = np.nan
                row[f"{side}_rb_carry_share"] = np.nan

            rb_starter = (trb.tail(ROLLING_WEEKS)
                             .groupby("player_id")["carries"].sum()
                             .idxmax()) if not trb.empty else None
            row[f"{side}_rb_injured"] = int(rb_starter in injured) if rb_starter else 0

        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Building rolling player features with injury data...")
    df = build_player_features()
    save(df, "player_features")
    print(f"Saved {len(df)} rows to player_features")
    print(df.dropna(subset=["home_qb_epa"]).head(3).to_string())