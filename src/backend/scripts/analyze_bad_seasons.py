import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from engine.model import (
    load_training_data, walk_forward_backtest, calculate_edges,
)

BREAKEVEN    = 52.4


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading training data...")
    df = load_training_data()
    print(f"  {len(df)} games\n")

    print("Running walk-forward backtest...")
    results = walk_forward_backtest(df)
    edges   = calculate_edges(results)
    print(f"  {len(edges)} edge games total\n")

    edges["spread_mag"]     = edges["vegas_spread"].abs()
    edges["model_conf_abs"] = edges["spread_gap"].abs()
    edges["week_bucket"]    = pd.cut(edges["week"],
                                      bins=[0,4,8,12,16,22],
                                      labels=["Wk1-4","Wk5-8","Wk9-12","Wk13-16","Wk17+"])

    # Dynamically identify bad and good seasons
    season_stats = (
        edges.groupby("season")
        .agg(bets=("ats_win","count"), ats=("ats_win","mean"), roi=("payout","sum"))
        .reset_index()
    )
    season_stats["ats_pct"] = season_stats["ats"] * 100

    BAD_SEASONS  = season_stats[season_stats["ats_pct"] < BREAKEVEN]["season"].tolist()
    GOOD_SEASONS = season_stats[season_stats["ats_pct"] >= BREAKEVEN]["season"].tolist()

    print(f"{'='*70}")
    print(f"SEASON BREAKDOWN — ALL SEASONS")
    print(f"{'='*70}")
    print(f"  {'Season':>8} {'Bets':>6} {'ATS%':>8} {'ROI':>10} {'Status':>10}")
    print(f"  {'-'*46}")
    for _, row in season_stats.iterrows():
        status = "BAD ✗" if row["ats_pct"] < BREAKEVEN else "good ✓"
        color_marker = " ←" if row["ats_pct"] < BREAKEVEN else ""
        print(f"  {int(row['season']):>8} {int(row['bets']):>6} {row['ats_pct']:>7.1f}%  "
              f"{row['roi']:>9.0f}  {status}{color_marker}")
    print(f"  {'─'*46}")
    print(f"  Bad seasons ({len(BAD_SEASONS)}):  {BAD_SEASONS}")
    print(f"  Good seasons ({len(GOOD_SEASONS)}): {GOOD_SEASONS}")

    bad  = edges[edges["season"].isin(BAD_SEASONS)]
    good = edges[edges["season"].isin(GOOD_SEASONS)]

    print(f"\n{'='*70}")
    print(f"BAD SEASONS COMBINED OVERVIEW ({len(BAD_SEASONS)} seasons)")
    print(f"{'='*70}")
    print(f"  {len(bad)} bets  {bad['ats_win'].mean()*100:.1f}% ATS  ROI: ${bad['payout'].sum():.0f}")
    print(f"  {len(good)} bets  {good['ats_win'].mean()*100:.1f}% ATS  ROI: ${good['payout'].sum():.0f}  (good seasons)")

    print(f"\n{'='*70}")
    print(f"WEEK DISTRIBUTION")
    print(f"{'='*70}")
    print(f"  {'Week':>8} {'2022 Bets':>10} {'2022 ATS%':>10} {'Good Bets':>10} {'Good ATS%':>10}")
    print(f"  {'-'*52}")
    for bucket in ["Wk1-4","Wk5-8","Wk9-12","Wk13-16","Wk17+"]:
        b = bad[bad["week_bucket"]  == bucket]
        g = good[good["week_bucket"] == bucket]
        b_ats = b["ats_win"].mean()*100 if len(b) > 0 else 0
        g_ats = g["ats_win"].mean()*100 if len(g) > 0 else 0
        print(f"  {bucket:>8} {len(b):>10} {b_ats:>9.1f}%  {len(g):>10} {g_ats:>9.1f}%")

    print(f"\n{'='*70}")
    print(f"SPREAD SIZE DISTRIBUTION")
    print(f"{'='*70}")
    print(f"  {'Spread':>12} {'2022 Bets':>10} {'2022 ATS%':>10} {'Good Bets':>10} {'Good ATS%':>10}")
    print(f"  {'-'*56}")
    for lo, hi in [(0,3),(3,5),(5,7),(7,10),(10,20)]:
        b = bad[(bad["spread_mag"]  >= lo) & (bad["spread_mag"]  < hi)]
        g = good[(good["spread_mag"] >= lo) & (good["spread_mag"] < hi)]
        b_ats = b["ats_win"].mean()*100 if len(b) > 0 else 0
        g_ats = g["ats_win"].mean()*100 if len(g) > 0 else 0
        print(f"  +{lo} to +{hi:<6}   {len(b):>10} {b_ats:>9.1f}%  {len(g):>10} {g_ats:>9.1f}%")

    print(f"\n{'='*70}")
    print(f"EDGE GAP DISTRIBUTION")
    print(f"{'='*70}")
    print(f"  {'Gap':>10} {'2022 Bets':>10} {'2022 ATS%':>10} {'Good Bets':>10} {'Good ATS%':>10}")
    print(f"  {'-'*54}")
    for lo, hi in [(2,3),(3,4),(4,5),(5,6),(6,20)]:
        b = bad[(bad["model_conf_abs"]  >= lo) & (bad["model_conf_abs"]  < hi)]
        g = good[(good["model_conf_abs"] >= lo) & (good["model_conf_abs"] < hi)]
        b_ats = b["ats_win"].mean()*100 if len(b) > 0 else 0
        g_ats = g["ats_win"].mean()*100 if len(g) > 0 else 0
        print(f"  {lo} to {hi:<6}   {len(b):>10} {b_ats:>9.1f}%  {len(g):>10} {g_ats:>9.1f}%")

    print(f"\n{'='*70}")
    print(f"HOME vs AWAY SPLIT")
    print(f"{'='*70}")
    for label, subset in [("2022", bad), ("Good seasons", good), ("All seasons", edges)]:
        home_bets = subset[subset["bet_home"] == True]
        away_bets = subset[subset["bet_home"] == False]
        h_ats = home_bets["ats_win"].mean()*100 if len(home_bets) > 0 else 0
        a_ats = away_bets["ats_win"].mean()*100 if len(away_bets) > 0 else 0
        print(f"  {label:<15}  home: {len(home_bets):>4} bets {h_ats:.1f}%  "
              f"away: {len(away_bets):>4} bets {a_ats:.1f}%")

    print(f"\n{'='*70}")
    print(f"WORST BETS IN 2022")
    print(f"{'='*70}")
    print(f"  {'Wk':>4} {'Home':>5} {'Away':>5} {'Vegas':>7} {'Model':>7} "
          f"{'Gap':>6} {'Result':>8}")
    print(f"  {'-'*48}")
    worst = bad[bad["ats_win"] == False].sort_values("model_conf_abs", ascending=False)
    for _, r in worst.head(15).iterrows():
        print(f"  {int(r['week']):>4} {r['home_team']:>5} {r['away_team']:>5} "
              f"{r['vegas_spread']:>6.1f}  {r['model_spread']:>6.1f}  "
              f"{r['spread_gap']:>5.1f}  {r['result']:>7.1f}")

    print(f"\n{'='*70}")
    print(f"POTENTIAL FILTERS — IMPACT ON 2022 vs ALL-TIME")
    print(f"{'='*70}")

    # Test filters
    filters = [
        ("Skip wk 1-3",      edges["week"] > 3),
        ("Skip wk 1-4",      edges["week"] > 4),
        ("Gap ≥ 3.0",        edges["model_conf_abs"] >= 3.0),
        ("Gap ≥ 3.5",        edges["model_conf_abs"] >= 3.5),
        ("Spread ≥ 3.0",     edges["spread_mag"] >= 3.0),
        ("Spread ≥ 4.0",     edges["spread_mag"] >= 4.0),
        ("Fav only",         edges["vegas_spread"] > 0),
        ("Dog only",         edges["vegas_spread"] < 0),
    ]

    print(f"  {'Filter':<20} {'All Bets':>9} {'All ATS%':>9} {'2022 Bets':>10} {'2022 ATS%':>10} {'All ROI':>10}")
    print(f"  {'-'*72}")
    base_ats = edges["ats_win"].mean()*100
    base_roi = edges["payout"].sum()
    print(f"  {'No filter':<20} {len(edges):>9} {base_ats:>8.1f}%  "
          f"{len(bad):>10} {bad['ats_win'].mean()*100:>9.1f}%  {base_roi:>9.0f}")
    for name, mask in filters:
        filtered     = edges[mask]
        filtered_bad = bad[mask[bad.index]]
        if len(filtered) == 0:
            continue
        ats     = filtered["ats_win"].mean()*100
        roi     = filtered["payout"].sum()
        bad_ats = filtered_bad["ats_win"].mean()*100 if len(filtered_bad) > 0 else 0
        marker  = " ←" if ats > base_ats else ""
        print(f"  {name:<20} {len(filtered):>9} {ats:>8.1f}%  "
              f"{len(filtered_bad):>10} {bad_ats:>9.1f}%  {roi:>9.0f}{marker}")

    print(f"\nDone. No DB writes.")