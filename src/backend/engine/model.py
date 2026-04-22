import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pickle
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from data.sql_helpers import get_connection, save

# ── Constants ─────────────────────────────────────────────────────────────────
MIN_SEASON        = 1999
INJURY_MIN_SEASON = 2015
EDGE_THRESHOLD    = 2.5
MIN_TRAIN_SEASONS = 5
UNIT_BET          = 100
ODDS              = -110

POSITION_WEIGHTS = {
    'QB': 10.0,
    'WR':  2.0, 'RB':  2.0, 'TE': 1.5,
    'T':   1.0, 'G':   0.5, 'C':  0.5,
    'CB':  1.0, 'S':   1.0, 'LB': 0.8,
    'DE':  0.8, 'DT':  0.5,
}

STATUS_SEVERITY = {
    'Out':          1.00,
    'Doubtful':     0.75,
    'Questionable': 0.25,
    'Probable':     0.05,
    'Note':         0.00,
    None:           0.00,
}


# ── Injury feature builder ────────────────────────────────────────────────────
def build_injury_features() -> pd.DataFrame:
    with get_connection() as conn:
        inj = pd.read_sql("""
            SELECT season, week, team, position, report_status, full_name
            FROM injuries_historical
            WHERE season >= ?
              AND position IN ('QB','WR','RB','TE','T','G','C','CB','S','LB','DE','DT')
        """, conn, params=(INJURY_MIN_SEASON,))

        games = pd.read_sql("""
            SELECT game_id, season, week, home_team, away_team
            FROM schedules
            WHERE game_type IN ('REG','WC','DIV','CON','SB')
              AND season >= ?
        """, conn, params=(INJURY_MIN_SEASON,))

    inj["pos_weight"] = inj["position"].map(POSITION_WEIGHTS).fillna(0.5)
    inj["severity"]   = inj["report_status"].map(STATUS_SEVERITY).fillna(0.0)
    inj["inj_score"]  = inj["pos_weight"] * inj["severity"]

    team_inj = (inj.groupby(["season", "week", "team"])["inj_score"]
                   .sum().reset_index()
                   .rename(columns={"inj_score": "injury_score"}))

    qb_inj = inj[inj["position"] == "QB"].copy()
    qb_inj["qb_out"] = qb_inj["report_status"].isin(["Out", "Doubtful"]).astype(int)
    qb_team = (qb_inj.groupby(["season", "week", "team"])["qb_out"]
                      .max().reset_index())

    team_inj = team_inj.merge(qb_team, on=["season", "week", "team"], how="left")
    team_inj["qb_out"] = team_inj["qb_out"].fillna(0)

    games = games.merge(
        team_inj.rename(columns={"team": "home_team", "injury_score": "home_injury_score", "qb_out": "home_qb_out"}),
        on=["season", "week", "home_team"], how="left"
    )
    games = games.merge(
        team_inj.rename(columns={"team": "away_team", "injury_score": "away_injury_score", "qb_out": "away_qb_out"}),
        on=["season", "week", "away_team"], how="left"
    )

    for col in ["home_injury_score", "away_injury_score", "home_qb_out", "away_qb_out"]:
        games[col] = games[col].fillna(0)

    games["injury_score_diff"] = games["home_injury_score"] - games["away_injury_score"]
    games["qb_out_diff"]       = games["away_qb_out"] - games["home_qb_out"]

    return games[["game_id", "home_injury_score", "away_injury_score",
                   "injury_score_diff", "home_qb_out", "away_qb_out", "qb_out_diff"]]


# ── Data loading ──────────────────────────────────────────────────────────────
def load_training_data() -> pd.DataFrame:
    with get_connection() as conn:
        df = pd.read_sql("""
            SELECT
                s.game_id, s.season, s.week,
                s.home_team, s.away_team,
                s.result,
                s.div_game,
                COALESCE(s.home_rest, 7) - COALESCE(s.away_rest, 7) AS rest_diff,
                h.elo_before  AS home_elo,
                a.elo_before  AS away_elo,
                (h.elo_before - a.elo_before) AS elo_diff,
                o.spread_line AS vegas_spread,
                CASE WHEN c.roof IN ('dome', 'closed') THEN 1 ELSE 0 END AS is_dome,
                CASE WHEN CAST(COALESCE(c.wind, 0) AS FLOAT) > 15 THEN 1 ELSE 0 END AS high_wind,
                pf.home_qb_epa, pf.away_qb_epa,
                pf.home_wr_yards, pf.away_wr_yards,
                pf.home_te_yards, pf.away_te_yards,
                pf.home_rb_rush_yards, pf.away_rb_rush_yards,
                pf.home_rb_rec_yards, pf.away_rb_rec_yards,
                hr.roll_passing_epa    AS home_roll_pass_epa,
                hr.roll_rushing_epa    AS home_roll_rush_epa,
                hr.roll_passing_yards  AS home_roll_pass_yards,
                hr.roll_rushing_yards  AS home_roll_rush_yards,
                hr.roll_def_sacks      AS home_roll_def_sacks,
                hr.roll_def_ints       AS home_roll_def_ints,
                hr.roll_def_tfl        AS home_roll_def_tfl,
                hr.roll_turnovers      AS home_roll_turnovers,
                hr.roll_first_downs    AS home_roll_first_downs,
                hr.roll_penalty_yards  AS home_roll_penalty_yards,
                hr.roll_avg_margin     AS home_roll_avg_margin,
                hr.roll_margin_trend   AS home_roll_margin_trend,
                ar.roll_passing_epa    AS away_roll_pass_epa,
                ar.roll_rushing_epa    AS away_roll_rush_epa,
                ar.roll_passing_yards  AS away_roll_pass_yards,
                ar.roll_rushing_yards  AS away_roll_rush_yards,
                ar.roll_def_sacks      AS away_roll_def_sacks,
                ar.roll_def_ints       AS away_roll_def_ints,
                ar.roll_def_tfl        AS away_roll_def_tfl,
                ar.roll_turnovers      AS away_roll_turnovers,
                ar.roll_first_downs    AS away_roll_first_downs,
                ar.roll_penalty_yards  AS away_roll_penalty_yards,
                ar.roll_avg_margin     AS away_roll_avg_margin,
                ar.roll_margin_trend   AS away_roll_margin_trend
            FROM schedules s
            JOIN elo_history h  ON h.game_id = s.game_id AND h.team = s.home_team
            JOIN elo_history a  ON a.game_id = s.game_id AND a.team = s.away_team
            JOIN player_features pf ON pf.game_id = s.game_id
            LEFT JOIN game_odds o        ON o.game_id = s.game_id
            LEFT JOIN game_conditions c  ON c.game_id = s.game_id
            LEFT JOIN team_rolling_stats hr ON hr.game_id = s.game_id AND hr.team = s.home_team
            LEFT JOIN team_rolling_stats ar ON ar.game_id = s.game_id AND ar.team = s.away_team
            WHERE s.game_type IN ('REG', 'WC', 'DIV', 'CON', 'SB')
              AND s.result IS NOT NULL
              AND s.season >= ?
            ORDER BY s.season, s.week
        """, conn, params=(MIN_SEASON,))

    df["qb_epa_diff"]           = df["home_qb_epa"]            - df["away_qb_epa"]
    df["wr_yards_diff"]         = df["home_wr_yards"]           - df["away_wr_yards"]
    df["te_yards_diff"]         = df["home_te_yards"]           - df["away_te_yards"]
    df["rb_rush_diff"]          = df["home_rb_rush_yards"]      - df["away_rb_rush_yards"]
    df["rb_rec_diff"]           = df["home_rb_rec_yards"]       - df["away_rb_rec_yards"]
    df["roll_pass_epa_diff"]    = df["home_roll_pass_epa"]      - df["away_roll_pass_epa"]
    df["roll_rush_epa_diff"]    = df["home_roll_rush_epa"]      - df["away_roll_rush_epa"]
    df["roll_pass_yards_diff"]  = df["home_roll_pass_yards"]    - df["away_roll_pass_yards"]
    df["roll_rush_yards_diff"]  = df["home_roll_rush_yards"]    - df["away_roll_rush_yards"]
    df["roll_def_sacks_diff"]   = df["home_roll_def_sacks"]     - df["away_roll_def_sacks"]
    df["roll_def_ints_diff"]    = df["home_roll_def_ints"]      - df["away_roll_def_ints"]
    df["roll_def_tfl_diff"]     = df["home_roll_def_tfl"]       - df["away_roll_def_tfl"]
    df["roll_turnovers_diff"]   = df["home_roll_turnovers"]     - df["away_roll_turnovers"]
    df["roll_first_downs_diff"] = df["home_roll_first_downs"]   - df["away_roll_first_downs"]
    df["roll_penalty_diff"]     = df["home_roll_penalty_yards"] - df["away_roll_penalty_yards"]
    df["roll_avg_margin_diff"]  = df["home_roll_avg_margin"]    - df["away_roll_avg_margin"]
    df["roll_margin_trend_diff"]= df["home_roll_margin_trend"]  - df["away_roll_margin_trend"]

    # Merge injury features (2015+, zero-filled earlier)
    inj = build_injury_features()
    df  = df.merge(inj, on="game_id", how="left")
    for col in ["home_injury_score", "away_injury_score",
                "injury_score_diff", "home_qb_out", "away_qb_out", "qb_out_diff"]:
        df[col] = df[col].fillna(0)

    return df


# ── Feature columns ───────────────────────────────────────────────────────────
FEATURE_COLS = [
    "elo_diff",
    "rest_diff",
    "is_dome",
    "high_wind",
    "qb_epa_diff",
    "wr_yards_diff",
    "te_yards_diff",
    "rb_rush_diff",
    "rb_rec_diff",
    "roll_pass_epa_diff",
    "roll_rush_epa_diff",
    "roll_pass_yards_diff",
    "roll_rush_yards_diff",
    "roll_def_sacks_diff",
    "roll_def_ints_diff",
    "roll_def_tfl_diff",
    "roll_turnovers_diff",
    "roll_penalty_diff",
    "roll_first_downs_diff",
    "injury_score_diff",
    "qb_out_diff",
    "home_qb_out",
    "away_qb_out",
]


# ── Model helpers ─────────────────────────────────────────────────────────────
def _make_base() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.01,
        subsample=0.8, colsample_bytree=1.0,
        random_state=42, verbosity=0,
    )


def _stack(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    X_train = train[FEATURE_COLS].fillna(0)
    y_train = train["result"]
    X_test  = test[FEATURE_COLS].fillna(0)

    base = _make_base()
    base.fit(X_train, y_train)

    mid     = len(train) // 2
    train_a = train.iloc[:mid]
    train_b = train.iloc[mid:]

    base_a = _make_base()
    base_a.fit(train_a[FEATURE_COLS].fillna(0), train_a["result"])
    oof = base_a.predict(train_b[FEATURE_COLS].fillna(0))

    meta = Ridge(alpha=1.0)
    meta.fit(
        np.column_stack([oof,
                         train_b["vegas_spread"].fillna(0).values,
                         train_b["elo_diff"].fillna(0).values]),
        train_b["result"].values,
    )

    base_preds = base.predict(X_test)
    return meta.predict(np.column_stack([
        base_preds,
        test["vegas_spread"].fillna(0).values,
        test["elo_diff"].fillna(0).values,
    ]))


# ── Walk-forward backtest ─────────────────────────────────────────────────────
def _bootstrap_confidence(train: pd.DataFrame, test: pd.DataFrame,
                          n_boots: int = 10) -> np.ndarray:
    preds = []
    rng   = np.random.default_rng(42)
    for _ in range(n_boots):
        idx    = rng.integers(0, len(train), size=len(train))
        sample = train.iloc[idx]
        m      = _make_base()
        m.fit(sample[FEATURE_COLS].fillna(0), sample["result"])
        preds.append(m.predict(test[FEATURE_COLS].fillna(0)))
    return np.std(preds, axis=0)


def walk_forward_backtest(df: pd.DataFrame) -> pd.DataFrame:
    seasons = sorted(df["season"].unique())
    results = []

    for i, test_season in enumerate(seasons):
        train_seasons = seasons[:i]
        if len(train_seasons) < 2:
            continue

        train = df[df["season"].isin(train_seasons)]
        test  = df[df["season"] == test_season]
        preds      = _stack(train, test)
        confidence = _bootstrap_confidence(train, test)

        s = test[["game_id", "season", "week", "home_team", "away_team",
                   "result", "vegas_spread", "div_game"]].copy()
        s["model_spread"]  = preds.round(1)
        s["spread_gap"]    = (s["model_spread"] - s["vegas_spread"]).round(1)
        s["confidence"]    = confidence.round(2)
        s["model_winner"]  = np.where(s["model_spread"] > 0, s["home_team"], s["away_team"])
        s["vegas_winner"]  = np.where(s["vegas_spread"] > 0, s["home_team"], s["away_team"])
        s["actual_winner"] = np.where(s["result"] > 0,       s["home_team"], s["away_team"])
        results.append(s)

    return pd.concat(results, ignore_index=True)


# ── Edge calculation ──────────────────────────────────────────────────────────
def calculate_edges(df: pd.DataFrame,
                    threshold: float = EDGE_THRESHOLD,
                    max_std:   float = None) -> pd.DataFrame:
    df = df[df["season"] >= df["season"].min() + MIN_TRAIN_SEASONS].copy()
    df = df[~df["week"].isin([17, 18])]
    agrees    = (
        ((df["model_spread"] > 0) & (df["vegas_spread"] > 0)) |
        ((df["model_spread"] < 0) & (df["vegas_spread"] < 0))
    )
    large_gap = df["spread_gap"].abs() >= threshold
    confident = df["confidence"] <= max_std if max_std is not None else pd.Series(True, index=df.index)
    e = df[agrees & large_gap & confident].copy()
    e["bet_home"] = e["spread_gap"] > 0
    e["ats_win"]  = np.where(e["bet_home"],
                             e["result"] > e["vegas_spread"],
                             e["result"] < e["vegas_spread"])
    e["payout"]   = np.where(e["ats_win"], UNIT_BET * (100 / abs(ODDS)), -UNIT_BET)
    return e


def apply_sharp_filter(edges: pd.DataFrame) -> pd.DataFrame:
    """
    Post-filter: only keep bets where sharp money agrees with model direction.
    Games with no line movement data pass through unfiltered.
    """
    with get_connection() as conn:
        lm = pd.read_sql("""
            SELECT game_id, spread_move, sharp_home, sharp_away
            FROM line_movement
        """, conn)

    e = edges.merge(lm, on="game_id", how="left")
    no_data      = e["spread_move"].isna()
    sharp_agrees = (
        (e["bet_home"] & (e["sharp_home"] == 1)) |
        (~e["bet_home"] & (e["sharp_away"] == 1))
    )
    return e[no_data | sharp_agrees].copy()


# ── Reporting ─────────────────────────────────────────────────────────────────
def print_report(df: pd.DataFrame) -> None:
    seasons = sorted(df["season"].unique())
    print(f"\n{'='*55}")
    print(f"{'Season':<8} {'Games':>6} {'MAE':>6} {'Winner%':>8} {'vs Vegas%':>10}")
    print(f"{'='*55}")
    total_correct = total_vegas = total_games = 0
    for season in seasons:
        s          = df[df["season"] == season]
        mae        = mean_absolute_error(s["result"], s["model_spread"])
        winner_pct = (s["model_winner"] == s["actual_winner"]).mean() * 100
        vegas_pct  = (s["vegas_winner"] == s["actual_winner"]).mean() * 100
        total_correct += (s["model_winner"] == s["actual_winner"]).sum()
        total_vegas   += (s["vegas_winner"] == s["actual_winner"]).sum()
        total_games   += len(s)
        print(f"{season:<8} {len(s):>6} {mae:>6.1f} {winner_pct:>7.1f}%  {winner_pct-vegas_pct:>+8.1f}%")
    print(f"{'='*55}")
    print(f"{'All-time':<8} {total_games:>6}        "
          f"{total_correct/total_games*100:>7.1f}%  "
          f"{(total_correct-total_vegas)/total_games*100:>+8.1f}%")


def print_edge_report(df_edges: pd.DataFrame, label: str = "Edges") -> None:
    if df_edges.empty:
        print("No edges found.")
        return
    seasons = sorted(df_edges["season"].unique())
    print(f"\n{'='*70}")
    print(f"{label}")
    print(f"{'Season':<8} {'Bets':>6} {'ATS Win%':>10} {'ROI($)':>10} {'ROI%':>8}")
    print(f"{'='*70}")
    total_bets = total_wins = total_roi = 0
    for season in seasons:
        s       = df_edges[df_edges["season"] == season]
        bets    = len(s)
        wins    = s["ats_win"].sum()
        roi     = s["payout"].sum()
        win_pct = wins / bets * 100 if bets else 0
        roi_pct = roi / (bets * UNIT_BET) * 100 if bets else 0
        print(f"{season:<8} {bets:>6} {win_pct:>10.1f}% {roi:>10.0f} {roi_pct:>7.1f}%")
        total_bets += bets
        total_wins += wins
        total_roi  += roi
    print(f"{'='*70}")
    tw  = total_wins / total_bets * 100 if total_bets else 0
    print(f"{'All-time':<8} {total_bets:>6} {tw:>10.1f}% "
          f"{total_roi:>10.0f} {total_roi/(total_bets*UNIT_BET)*100:>7.1f}%")
    print(f"\nBreak-even ATS%: 52.4%  |  Model: {tw:.1f}%  |  "
          f"Edge: {tw-52.4:+.1f}%  |  {'PROFITABLE' if tw > 52.4 else 'NOT PROFITABLE'}")


def print_stacked_filter_report(results: pd.DataFrame) -> None:
    print(f"\n{'='*75}")
    print("Stacked filter sensitivity")
    print(f"{'Thresh':>7} {'Sharp':>6} {'No Div':>7} {'Bets':>6} {'ATS%':>8} {'ROI($)':>10}")
    print(f"{'='*75}")

    for threshold in [2.0, 2.5, 3.0, 3.5, 4.0]:
        for use_sharp in [False, True]:
            for no_div in [False, True]:
                edges = calculate_edges(results, threshold=threshold)
                if use_sharp:
                    edges = apply_sharp_filter(edges)
                if no_div and "div_game" in edges.columns:
                    edges = edges[edges["div_game"] == 0]
                if len(edges) < 3:
                    continue
                ats = edges["ats_win"].mean() * 100
                roi = edges["payout"].sum()
                print(f"{threshold:>7.1f} {'Y' if use_sharp else 'N':>6} "
                      f"{'Y' if no_div else 'N':>7} {len(edges):>6} "
                      f"{ats:>7.1f}% {roi:>10.0f}")


# ── Train & save final model ──────────────────────────────────────────────────
def train_final_model(df: pd.DataFrame) -> tuple:
    X, y   = df[FEATURE_COLS].fillna(0), df["result"]
    base   = _make_base()
    base.fit(X, y)

    mid    = len(df) // 2
    base_a = _make_base()
    base_a.fit(df.iloc[:mid][FEATURE_COLS].fillna(0), df.iloc[:mid]["result"])
    oof    = base_a.predict(df.iloc[mid:][FEATURE_COLS].fillna(0))

    meta   = Ridge(alpha=1.0)
    meta.fit(
        np.column_stack([oof,
                         df.iloc[mid:]["vegas_spread"].fillna(0).values,
                         df.iloc[mid:]["elo_diff"].fillna(0).values]),
        df.iloc[mid:]["result"].values,
    )

    model_dir = Path(__file__).resolve().parent
    with open(model_dir / "trained_base_model.pkl", "wb") as f:
        pickle.dump(base, f)
    with open(model_dir / "trained_meta_model.pkl", "wb") as f:
        pickle.dump(meta, f)
    print(f"Models saved to {model_dir}")
    return base, meta


def load_trained_model() -> tuple:
    model_dir = Path(__file__).resolve().parent
    with open(model_dir / "trained_base_model.pkl", "rb") as f:
        base = pickle.load(f)
    with open(model_dir / "trained_meta_model.pkl", "rb") as f:
        meta = pickle.load(f)
    return base, meta


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading training data...")
    df = load_training_data()
    print(f"Loaded {len(df)} games ({df['season'].min()}–{df['season'].max()})")
    print(f"Features: {len(FEATURE_COLS)}")

    print("\nTraining final model on all data...")
    train_final_model(df)

    print("\nRunning walk-forward backtest...")
    results = walk_forward_backtest(df)
    print_report(results)

    print("\nBase edges (no filters):")
    df_edges = calculate_edges(results)
    print_edge_report(df_edges, label="Base edges (threshold=2.5, no sharp filter)")

    print("\nSharp-filtered edges:")
    df_sharp = apply_sharp_filter(df_edges)
    print_edge_report(df_sharp, label="Sharp-filtered edges (threshold=2.5 + sharp agrees)")

    print("\nStacked filter analysis:")
    print_stacked_filter_report(results)

    print("\nSaving predictions to model_predictions table...")
    save(results, "model_predictions")
    print("Done.")
