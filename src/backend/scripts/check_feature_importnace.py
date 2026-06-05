import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from engine.model import load_training_data, FEATURE_COLS, _make_base

# ── Config ────────────────────────────────────────────────────────────────────
PCA_VARIANCE_TARGET = 0.95   # how much variance to explain
TOP_N_FEATURES      = 26     # show all features in importance ranking


# ── Feature Importance ────────────────────────────────────────────────────────
def run_feature_importance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Train a single model on all data and extract feature importance.
    Uses three methods:
      - weight:  number of times a feature is used in a split
      - gain:    average improvement in loss when feature is used
      - cover:   average number of samples affected by splits using this feature
    Gain is the most meaningful for predictive power.
    """
    X = df[FEATURE_COLS].fillna(0)
    y = df["result"]

    model = _make_base()
    model.fit(X, y)

    importance_types = ["weight", "gain", "cover"]
    results = {}
    for itype in importance_types:
        scores = model.get_booster().get_score(importance_type=itype)
        results[itype] = scores

    importance_df = pd.DataFrame({
        "feature": FEATURE_COLS,
    })

    for itype in importance_types:
        importance_df[itype] = importance_df["feature"].map(results[itype]).fillna(0)

    # Normalize each column to 0-100 for readability
    for itype in importance_types:
        max_val = importance_df[itype].max()
        if max_val > 0:
            importance_df[f"{itype}_pct"] = (importance_df[itype] / max_val * 100).round(1)
        else:
            importance_df[f"{itype}_pct"] = 0

    importance_df = importance_df.sort_values("gain", ascending=False).reset_index(drop=True)
    return importance_df


def print_feature_importance(importance_df: pd.DataFrame) -> None:
    print(f"\n{'='*75}")
    print("FEATURE IMPORTANCE")
    print(f"Gain = avg loss improvement per split (most meaningful)")
    print(f"Weight = split frequency  |  Cover = avg samples per split")
    print(f"{'='*75}")
    print(f"{'Rank':<5} {'Feature':<32} {'Gain%':>7} {'Weight%':>8} {'Cover%':>8}  Status")
    print(f"{'-'*70}")

    for i, row in importance_df.iterrows():
        gain_pct   = row["gain_pct"]
        weight_pct = row["weight_pct"]

        # Flag weak features
        if gain_pct < 5:
            status = "  << WEAK"
        elif gain_pct < 15:
            status = "  < low"
        elif gain_pct > 60:
            status = "  >> STRONG"
        else:
            status = ""

        bar = "█" * int(gain_pct / 5)
        print(f"{i+1:<5} {row['feature']:<32} {gain_pct:>6.1f}% {weight_pct:>7.1f}% {row['cover_pct']:>7.1f}%  {bar}{status}")

    weak = importance_df[importance_df["gain_pct"] < 5]
    if not weak.empty:
        print(f"\nWeak features (gain < 5% of max) — consider removing:")
        for _, row in weak.iterrows():
            print(f"  {row['feature']}")


# ── PCA ───────────────────────────────────────────────────────────────────────
def run_pca(df: pd.DataFrame) -> dict:
    """
    Run PCA on the feature matrix.
    Shows:
    - How many components explain PCA_VARIANCE_TARGET of variance
    - Which original features load most heavily onto the top components
    - Whether features are redundant (high correlation = fewer effective components)
    """
    X = df[FEATURE_COLS].fillna(0)

    # Standardize — PCA is sensitive to scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Full PCA
    pca = PCA()
    pca.fit(X_scaled)

    explained          = pca.explained_variance_ratio_
    cumulative         = np.cumsum(explained)
    n_components_95    = np.argmax(cumulative >= PCA_VARIANCE_TARGET) + 1
    n_components_80    = np.argmax(cumulative >= 0.80) + 1
    n_features         = len(FEATURE_COLS)
    effective_features = n_components_95

    return {
        "pca":               pca,
        "explained":         explained,
        "cumulative":        cumulative,
        "n_features":        n_features,
        "n_components_80":   n_components_80,
        "n_components_95":   n_components_95,
        "effective_features":effective_features,
        "components":        pca.components_,
        "feature_names":     FEATURE_COLS,
    }


def print_pca(pca_result: dict) -> None:
    explained    = pca_result["explained"]
    cumulative   = pca_result["cumulative"]
    n_features   = pca_result["n_features"]
    n_comp_80    = pca_result["n_components_80"]
    n_comp_95    = pca_result["n_components_95"]
    components   = pca_result["components"]
    feature_names = pca_result["feature_names"]

    print(f"\n{'='*75}")
    print("PCA — DIMENSIONALITY ANALYSIS")
    print(f"{'='*75}")
    print(f"Total features:          {n_features}")
    print(f"Components for 80% var:  {n_comp_80}  ({n_comp_80/n_features*100:.0f}% of features)")
    print(f"Components for 95% var:  {n_comp_95}  ({n_comp_95/n_features*100:.0f}% of features)")

    redundancy = (n_features - n_comp_95) / n_features * 100
    if redundancy > 40:
        assessment = "HIGH redundancy — significant feature overlap"
    elif redundancy > 20:
        assessment = "MODERATE redundancy — some features are correlated"
    else:
        assessment = "LOW redundancy — features are mostly independent"
    print(f"Redundancy assessment:   {assessment}")

    print(f"\nVariance explained per component:")
    print(f"{'Comp':<6} {'Var%':>6} {'Cumul%':>8}  {'Bar'}")
    print(f"{'-'*45}")
    for i in range(min(15, n_features)):
        var_pct   = explained[i] * 100
        cum_pct   = cumulative[i] * 100
        bar       = "█" * int(var_pct * 2)
        marker    = " << 80%" if i + 1 == n_comp_80 else (" << 95%" if i + 1 == n_comp_95 else "")
        print(f"PC{i+1:<4} {var_pct:>5.1f}% {cum_pct:>7.1f}%  {bar}{marker}")
    if n_features > 15:
        remaining = (1 - cumulative[14]) * 100
        print(f"...    {remaining:>5.1f}% remaining across {n_features-15} components")

    print(f"\nTop feature loadings per principal component:")
    for pc_idx in range(min(5, n_comp_95)):
        loadings    = components[pc_idx]
        sorted_idx  = np.argsort(np.abs(loadings))[::-1]
        top_features = [(feature_names[j], loadings[j]) for j in sorted_idx[:4]]
        top_str     = "  |  ".join([f"{f} ({v:+.2f})" for f, v in top_features])
        print(f"  PC{pc_idx+1}: {top_str}")

    # Correlation matrix summary
    print(f"\nHighly correlated feature pairs (|r| > 0.7):")
    import pandas as pd
    corr = pd.DataFrame(
        np.corrcoef(np.column_stack([pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)])),
    )
    # Compute actual correlation from the PCA input
    # We need to recompute from original data — flag this as a note
    print(f"  (Run with --correlation flag to see full correlation matrix)")
    print(f"  Key insight: if n_components_95 << n_features, you have redundant features")
    print(f"  {n_comp_95} components explain 95% of variance across {n_features} features")
    if n_comp_95 <= n_features * 0.6:
        print(f"  Recommendation: consider removing the weakest features identified above")
    else:
        print(f"  Recommendation: features are reasonably independent, keep all")


def print_correlation_summary(df: pd.DataFrame) -> None:
    """Show highly correlated feature pairs."""
    X = df[FEATURE_COLS].fillna(0)
    corr = pd.DataFrame(X, columns=FEATURE_COLS).corr()

    print(f"\n{'='*75}")
    print("HIGHLY CORRELATED FEATURE PAIRS (|r| > 0.7)")
    print("These pairs may be redundant — model uses both but gains little from one")
    print(f"{'='*75}")

    pairs = []
    for i in range(len(FEATURE_COLS)):
        for j in range(i + 1, len(FEATURE_COLS)):
            r = corr.iloc[i, j]
            if abs(r) > 0.7:
                pairs.append((FEATURE_COLS[i], FEATURE_COLS[j], r))

    if pairs:
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        for f1, f2, r in pairs:
            direction = "positive" if r > 0 else "negative"
            print(f"  {f1:<32}  {f2:<32}  r={r:+.3f} ({direction})")
    else:
        print("  No highly correlated pairs found — features are independent")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading training data...")
    df = load_training_data()
    print(f"  {len(df)} games, {len(FEATURE_COLS)} features")

    print("\nRunning feature importance...")
    importance_df = run_feature_importance(df)
    print_feature_importance(importance_df)

    print("\nRunning PCA...")
    pca_result = run_pca(df)
    print_pca(pca_result)

    print("\nRunning correlation analysis...")
    print_correlation_summary(df)

    print(f"\n{'='*75}")
    print("SUMMARY")
    print(f"{'='*75}")
    weak_features = importance_df[importance_df["gain_pct"] < 5]["feature"].tolist()
    n_comp_95     = pca_result["n_components_95"]
    n_features    = len(FEATURE_COLS)

    print(f"Features:          {n_features}")
    print(f"Weak features:     {len(weak_features)}")
    print(f"Effective dims:    {n_comp_95} (explain 95% variance)")

    if weak_features:
        print(f"\nTop candidates for removal:")
        for f in weak_features:
            row = importance_df[importance_df["feature"] == f].iloc[0]
            print(f"  {f:<32} gain={row['gain_pct']:.1f}%")
        print(f"\nRemoving weak features may improve generalization.")
        print(f"Test by removing them from FEATURE_COLS and running backtest.py")