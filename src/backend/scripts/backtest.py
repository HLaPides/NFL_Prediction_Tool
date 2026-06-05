"""
backtest.py — Season-start sanity check for the spread model.

Runs a full walk-forward backtest across all historical seasons and prints
accuracy, ATS performance, and ROI under the current model params and edge
filter. Also saves results to the model_predictions table so the frontend
history page has data.

Run this at the start of each season to confirm the model hasn't drifted
before live games begin. Should reproduce ~66.4% accuracy and ~58.7% ATS
on the asymmetric edge filter.

Usage:
    python3.11 scripts/backtest.py

Output:
    - Per-season accuracy and vs-Vegas delta (printed)
    - Base edge report at threshold=2.5 (printed)
    - Sharp-filtered edge report (printed)
    - Stacked filter sensitivity table (printed)
    - Results saved to model_predictions table in DB
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from engine.model import (
    load_training_data,
    walk_forward_backtest,
    calculate_edges,
    apply_sharp_filter,
    print_report,
    print_edge_report,
    print_stacked_filter_report,
    save,
)

if __name__ == "__main__":
    print("Loading training data...")
    df = load_training_data()
    print(f"Loaded {len(df)} games ({df['season'].min()}–{df['season'].max()})")

    print("\nRunning walk-forward backtest...")
    results = walk_forward_backtest(df)
    print_report(results)

    print("\nBase edges:")
    edges = calculate_edges(results)
    print_edge_report(edges, label="Base edges (threshold=2.5)")

    print("\nSharp-filtered edges:")
    sharp = apply_sharp_filter(edges)
    print_edge_report(sharp, label="Sharp-filtered edges")

    print("\nStacked filter analysis:")
    print_stacked_filter_report(results)

    print("\nSaving to model_predictions...")
    save(results, "model_predictions")
    print("Done.")