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

UNIT_BET  = 100
ODDS      = -110
BREAKEVEN = 52.4


def score(edges: pd.DataFrame, label: str) -> None:
    if len(edges) == 0:
        print(f"  {label:<45} no bets")
        return
    ats = edges["ats_win"].mean() * 100
    roi = edges["payout"].sum()
    edge = ats - BREAKEVEN
    marker = " ←" if ats > 54.8 else ""
    print(f"  {label:<45} {len(edges):>5} bets  {ats:>6.1f}% ATS  {roi:>+8.0f}  {edge:>+5.1f}%{marker}")


if __name__ == "__main__":
    print("Loading training data...")
    df = load_training_data()
    print(f"  {len(df)} games\n")

    print("Running walk-forward backtest...")
    results = walk_forward_backtest(df)
    edges   = calculate_edges(results)
    print(f"  {len(edges)} edge games\n")

    edges["spread_mag"]     = edges["vegas_spread"].abs()
    edges["model_conf_abs"] = edges["spread_gap"].abs()

    # Split fav vs dog
    is_fav = (
        ((edges["bet_home"] == True)  & (edges["vegas_spread"] > 0)) |
        ((edges["bet_home"] == False) & (edges["vegas_spread"] < 0))
    )
    is_dog = ~is_fav

    fav_bets = edges[is_fav].copy()
    dog_bets = edges[is_dog].copy()

    print(f"{'='*75}")
    print(f"BASELINE")
    print(f"{'='*75}")
    score(edges,    "All bets (current)")
    score(fav_bets, "Favorite bets only")
    score(dog_bets, "Underdog bets only")

    print(f"\n{'='*75}")
    print(f"FAVORITE FILTERS — combined with all dog bets")
    print(f"  Baseline: 726 bets  54.8% ATS  +$3,382")
    print(f"{'='*75}")

    # Test every combination of gap cap and spread cap on favorites
    for max_gap in [2.5, 3.0, 3.5, 4.0, None]:
        for max_spread in [5, 7, 10, None]:
            fav = fav_bets.copy()
            label_parts = []

            if max_gap is not None:
                fav = fav[fav["model_conf_abs"] <= max_gap]
                label_parts.append(f"gap≤{max_gap}")
            if max_spread is not None:
                fav = fav[fav["spread_mag"] <= max_spread]
                label_parts.append(f"spread≤{max_spread}")

            label = "Fav: " + (", ".join(label_parts) if label_parts else "no filter")
            combined = pd.concat([dog_bets, fav])
            score(combined, label)

    print(f"\n{'='*75}")
    print(f"WEEK FILTER ON FAVORITES — combined with all dog bets")
    print(f"{'='*75}")
    for min_week in [4, 5, 6, 7]:
        for max_week in [16, 18]:
            fav = fav_bets[(fav_bets["week"] > min_week) & (fav_bets["week"] <= max_week)]
            label = f"Fav: wk{min_week+1}-{max_week}"
            combined = pd.concat([dog_bets, fav])
            score(combined, label)

    print(f"\n{'='*75}")
    print(f"COMBINED — GAP + SPREAD + WEEK on favorites")
    print(f"{'='*75}")
    combos = [
        ("gap≤3.0 + spread≤7",          lambda f: f[(f["model_conf_abs"] <= 3.0) & (f["spread_mag"] <= 7)]),
        ("gap≤3.0 + spread≤7 + wk5-16", lambda f: f[(f["model_conf_abs"] <= 3.0) & (f["spread_mag"] <= 7) & (f["week"] >= 5) & (f["week"] <= 16)]),
        ("gap≤2.5 + spread≤7",          lambda f: f[(f["model_conf_abs"] <= 2.5) & (f["spread_mag"] <= 7)]),
        ("gap≤3.0 + spread≤5",          lambda f: f[(f["model_conf_abs"] <= 3.0) & (f["spread_mag"] <= 5)]),
        ("gap≤3.0 + wk5-16",            lambda f: f[(f["model_conf_abs"] <= 3.0) & (f["week"] >= 5) & (f["week"] <= 16)]),
        ("spread≤7 + wk5-16",           lambda f: f[(f["spread_mag"] <= 7) & (f["week"] >= 5) & (f["week"] <= 16)]),
        ("gap≤3.0 + spread≤7 + home",   lambda f: f[(f["model_conf_abs"] <= 3.0) & (f["spread_mag"] <= 7) & (f["bet_home"] == True)]),
    ]

    for label, fn in combos:
        fav = fn(fav_bets)
        combined = pd.concat([dog_bets, fav])
        score(combined, f"Fav: {label}")

    print(f"\n{'='*75}")
    print(f"SEASON BREAKDOWN — best config vs baseline")
    print(f"{'='*75}")
    # Find best config
    best_ats  = 0
    best_fav  = None
    best_name = ""
    for label, fn in combos:
        fav = fn(fav_bets)
        combined = pd.concat([dog_bets, fav])
        if len(combined) > 0 and combined["ats_win"].mean()*100 > best_ats:
            best_ats  = combined["ats_win"].mean()*100
            best_fav  = fav
            best_name = label

    if best_fav is not None:
        best_combined = pd.concat([dog_bets, best_fav])
        print(f"  Best config: {best_name}")
        print(f"\n  {'Season':>8} {'Bets':>6} {'ATS%':>8} {'ROI':>10} {'Baseline ATS':>14}")
        print(f"  {'-'*50}")
        for season in sorted(best_combined["season"].unique()):
            s    = best_combined[best_combined["season"] == season]
            base = edges[edges["season"] == season]
            ats  = s["ats_win"].mean()*100
            roi  = s["payout"].sum()
            b_ats = base["ats_win"].mean()*100
            marker = " ✗" if ats < BREAKEVEN else ""
            print(f"  {season:>8} {len(s):>6} {ats:>7.1f}%  {roi:>9.0f}  {b_ats:>13.1f}%{marker}")
        print(f"  {'─'*50}")
        print(f"  {'All-time':>8} {len(best_combined):>6} "
              f"{best_combined['ats_win'].mean()*100:>7.1f}%  "
              f"{best_combined['payout'].sum():>9.0f}  "
              f"{edges['ats_win'].mean()*100:>13.1f}%")

    print(f"\nDone. No DB writes.")