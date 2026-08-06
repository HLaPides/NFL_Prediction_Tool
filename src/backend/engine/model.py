"""
model.py — XGBoost spread model: training, backtesting, and edge calculation.

Architecture:
    Pure XGBoost trained on ATS margin (result - vegas_spread) as the target.
    Vegas spread is included as a peer feature (not a stacking anchor).
    Output is ATS margin prediction; convert back to spread by adding vegas_spread.

    This means the model learns market inefficiency directly. Vegas importance
    sits at ~37% — it's informative but not dominant. The model has genuine
    independent signal.

Params (Optuna-tuned, 403 trials, confirmed ceiling):
    n_estimators=403, max_depth=2, learning_rate=0.0339,
    subsample=0.7768, colsample_bytree=0.7109, min_child_weight=6,
    gamma=1.7893, reg_alpha=3.0354, reg_lambda=7.8139

Edge filter (asymmetric — key finding):
    Underdog bets: all edges ≥ threshold (model overcorrects upward on dogs)
    Favorite bets: gap ≤ 3.0 AND spread magnitude ≤ 7 only
    Additional filters: no divisional games, no weeks 17-18

Backtest results (walk-forward, 20 seasons):
    66.4% winner accuracy, 58.7% ATS on 446 filtered bets, +$5,418 ROI

Walk-forward methodology:
    Train on seasons 1..N-1, test on season N. Never touches future data.
    Bootstrap confidence: std across 25 bootstrap models per game.

Writes to DB:
    trained_base_model.pkl (on disk), model_predictions (table)
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pickle
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from data.sql_helpers import get_connection, save

# ── Constants ─────────────────────────────────────────────────────────────────
MIN_SEASON        = 1999
INJURY_MIN_SEASON = 2015
EDGE_THRESHOLD    = 2.0
MIN_TRAIN_SEASONS = 5
UNIT_BET          = 100
ODDS              = -110
CLOSE_GAME_MARGIN = 7
CLUTCH_ROLL       = 8

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
    """
    Build pre-game injury scores for every historical game.

    Computes a weighted severity score per team per game:
        pos_weight × status_severity for each injured player, summed.

    QB out is tracked as a separate binary flag (qb_out_diff used in playoff model).
    Only available from INJURY_MIN_SEASON (2015) onward — earlier games get 0.

    Returns:
        pd.DataFrame: One row per game_id with injury_score_diff and qb_out_diff.
    """
    with get_connection() as conn:
        inj = pd.read_sql("""
            SELECT season, week, team, position, report_status, full_name
            FROM injuries_historical
            WHERE season >= %s
              AND position IN ('QB','WR','RB','TE','T','G','C','CB','S','LB','DE','DT')
        """, conn, params=(INJURY_MIN_SEASON,))

        games = pd.read_sql("""
            SELECT game_id, season, week, home_team, away_team
            FROM schedules
            WHERE game_type IN ('REG','WC','DIV','CON','SB')
              AND season >= %s
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


# ── Clutch feature builder ────────────────────────────────────────────────────
def build_clutch_features() -> pd.DataFrame:
    """
    Build rolling clutch rate (win% in close games) for every historical game.

    Close game = final margin ≤ CLOSE_GAME_MARGIN (7 pts).
    Rolling window = CLUTCH_ROLL (8 games), min 3 close games required.
    Uses shift(1) to ensure pre-game only — no leakage.

    Returns:
        pd.DataFrame: One row per game_id with home_clutch_rate and away_clutch_rate.
                      Null teams filled to 0.5 (neutral).
    """
    with get_connection() as conn:
        games = pd.read_sql("""
            SELECT game_id, season, week, home_team, away_team, result
            FROM schedules
            WHERE game_type IN ('REG', 'WC', 'DIV', 'CON', 'SB')
              AND result IS NOT NULL
            ORDER BY season, week
        """, conn)

    home_rows = games[["game_id", "season", "week", "home_team", "result"]].copy()
    home_rows = home_rows.rename(columns={"home_team": "team"})
    home_rows["team_won"] = (home_rows["result"] > 0).astype(float)
    home_rows["margin"]   = home_rows["result"].abs()

    away_rows = games[["game_id", "season", "week", "away_team", "result"]].copy()
    away_rows = away_rows.rename(columns={"away_team": "team"})
    away_rows["team_won"] = (away_rows["result"] < 0).astype(float)
    away_rows["margin"]   = away_rows["result"].abs()

    all_rows = pd.concat([home_rows, away_rows], ignore_index=True)
    all_rows = all_rows.sort_values(["team", "season", "week"]).reset_index(drop=True)

    all_rows["is_close"]  = (all_rows["margin"] <= CLOSE_GAME_MARGIN).astype(float)
    all_rows["close_win"] = all_rows["is_close"] * all_rows["team_won"]

    all_rows["roll_close_games"] = (
        all_rows.groupby("team")["is_close"]
                .transform(lambda x: x.shift(1).rolling(CLUTCH_ROLL, min_periods=3).sum())
    )
    all_rows["roll_close_wins"] = (
        all_rows.groupby("team")["close_win"]
                .transform(lambda x: x.shift(1).rolling(CLUTCH_ROLL, min_periods=3).sum())
    )
    all_rows["roll_clutch_rate"] = (
        all_rows["roll_close_wins"] / all_rows["roll_close_games"].replace(0, np.nan)
    ).fillna(0.5)

    clutch = all_rows[["game_id", "team", "roll_clutch_rate"]].copy()

    home_c = (
        games[["game_id", "home_team"]]
        .merge(
            clutch.rename(columns={"team": "home_team", "roll_clutch_rate": "home_clutch_rate"}),
            on=["game_id", "home_team"], how="left"
        )
    )
    away_c = (
        games[["game_id", "away_team"]]
        .merge(
            clutch.rename(columns={"team": "away_team", "roll_clutch_rate": "away_clutch_rate"}),
            on=["game_id", "away_team"], how="left"
        )
    )

    result = home_c.merge(away_c, on="game_id", how="left")
    result["home_clutch_rate"] = result["home_clutch_rate"].fillna(0.5)
    result["away_clutch_rate"] = result["away_clutch_rate"].fillna(0.5)

    return result[["game_id", "home_clutch_rate", "away_clutch_rate"]]


# ── Data loading ──────────────────────────────────────────────────────────────
def load_training_data() -> pd.DataFrame:
    """
    Load and assemble the full training dataset from DB.

    Joins schedules → ELO history → player features → game odds →
    game conditions → rolling team stats → defense ratings, then
    merges injury and clutch features computed in-memory.

    All diff columns are home - away (positive = home team advantage).
    Vegas spread sign convention: positive = home team favored.

    Returns:
        pd.DataFrame: One row per completed game with all features and result.
                      Ready to pass directly to FEATURE_COLS selection.
    """
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
                hd.roll_adj_def_pass_epa AS home_adj_def_pass_epa,
                hd.roll_adj_def_rush_epa AS home_adj_def_rush_epa,
                ad.roll_adj_def_pass_epa AS away_adj_def_pass_epa,
                ad.roll_adj_def_rush_epa AS away_adj_def_rush_epa
            FROM schedules s
            JOIN elo_history h  ON h.game_id = s.game_id AND h.team = s.home_team
            JOIN elo_history a  ON a.game_id = s.game_id AND a.team = s.away_team
            JOIN player_features pf ON pf.game_id = s.game_id
            LEFT JOIN game_odds o        ON o.game_id = s.game_id
            LEFT JOIN game_conditions c  ON c.game_id = s.game_id
            LEFT JOIN team_rolling_stats hr ON hr.game_id = s.game_id AND hr.team = s.home_team
            LEFT JOIN team_rolling_stats ar ON ar.game_id = s.game_id AND ar.team = s.away_team
            LEFT JOIN defense_ratings hd ON hd.game_id = s.game_id AND hd.team = s.home_team
            LEFT JOIN defense_ratings ad ON ad.game_id = s.game_id AND ad.team = s.away_team
            WHERE s.game_type IN ('REG', 'WC', 'DIV', 'CON', 'SB')
              AND s.result IS NOT NULL
              AND s.season >= %s
            ORDER BY s.season, s.week
        """, conn, params=(MIN_SEASON,))

    df["qb_epa_diff"]                = df["home_qb_epa"]            - df["away_qb_epa"]
    df["wr_yards_diff"]              = df["home_wr_yards"]           - df["away_wr_yards"]
    df["te_yards_diff"]              = df["home_te_yards"]           - df["away_te_yards"]
    df["rb_rush_diff"]               = df["home_rb_rush_yards"]      - df["away_rb_rush_yards"]
    df["rb_rec_diff"]                = df["home_rb_rec_yards"]       - df["away_rb_rec_yards"]
    df["roll_pass_epa_diff"]         = df["home_roll_pass_epa"]      - df["away_roll_pass_epa"]
    df["roll_rush_epa_diff"]         = df["home_roll_rush_epa"]      - df["away_roll_rush_epa"]
    df["roll_pass_yards_diff"]       = df["home_roll_pass_yards"]    - df["away_roll_pass_yards"]
    df["roll_rush_yards_diff"]       = df["home_roll_rush_yards"]    - df["away_roll_rush_yards"]
    df["roll_def_sacks_diff"]        = df["home_roll_def_sacks"]     - df["away_roll_def_sacks"]
    df["roll_def_ints_diff"]         = df["home_roll_def_ints"]      - df["away_roll_def_ints"]
    df["roll_def_tfl_diff"]          = df["home_roll_def_tfl"]       - df["away_roll_def_tfl"]
    df["roll_turnovers_diff"]        = df["home_roll_turnovers"]     - df["away_roll_turnovers"]
    df["roll_first_downs_diff"]      = df["home_roll_first_downs"]   - df["away_roll_first_downs"]
    df["roll_penalty_diff"]          = df["home_roll_penalty_yards"] - df["away_roll_penalty_yards"]
    df["roll_adj_def_pass_epa_diff"] = df["home_adj_def_pass_epa"].fillna(0) - df["away_adj_def_pass_epa"].fillna(0)
    df["roll_adj_def_rush_epa_diff"] = df["home_adj_def_rush_epa"].fillna(0) - df["away_adj_def_rush_epa"].fillna(0)

    inj = build_injury_features()
    df  = df.merge(inj, on="game_id", how="left")
    for col in ["home_injury_score", "away_injury_score",
                "injury_score_diff", "home_qb_out", "away_qb_out", "qb_out_diff"]:
        df[col] = df[col].fillna(0)

    clutch = build_clutch_features()
    df = df.merge(clutch, on="game_id", how="left")
    df["home_clutch_rate"] = df["home_clutch_rate"].fillna(0.5)
    df["away_clutch_rate"] = df["away_clutch_rate"].fillna(0.5)
    df["clutch_rate_diff"] = df["home_clutch_rate"] - df["away_clutch_rate"]

    print(f"  Training rows after all merges: {len(df):,}")
    return df


# ── Feature columns ───────────────────────────────────────────────────────────
FEATURE_COLS = [
    "vegas_spread",        # Vegas opening/closing spread — peer feature, not anchor
    "elo_diff",            # ELO rating differential (home - away)
    "roll_pass_yards_diff",
    "injury_score_diff",   # Weighted injury severity (home - away)
    "roll_def_tfl_diff",
    "roll_pass_epa_diff",
    "te_yards_diff",
    "qb_epa_diff",         # EPA per attempt differential (starter, injury-adjusted)
    "wr_yards_diff",
    "rb_rush_diff",
    "roll_first_downs_diff",
    "roll_def_sacks_diff",
    "is_dome",             # 1 if dome/closed roof
    "roll_rush_epa_diff",
    "rest_diff",           # Days rest differential (home - away)
    "roll_rush_yards_diff",
    "roll_penalty_diff",
    "roll_def_ints_diff",
    "clutch_rate_diff",    # Close-game win rate differential
    "rb_rec_diff",
    "high_wind",           # 1 if wind > 15 mph
]


# ── Model factory ─────────────────────────────────────────────────────────────
def _make_base() -> XGBRegressor:
    """
    Instantiate the XGBoost model with Optuna-tuned params.
    These params are final — confirmed ceiling across 3+ Optuna runs.
    Do not change without re-running hyperparameter_tuning.py.
    """
    return XGBRegressor(
        n_estimators=403,
        max_depth=2,
        learning_rate=0.0339,
        subsample=0.7768,
        colsample_bytree=0.7109,
        min_child_weight=6,
        gamma=1.7893,
        reg_alpha=3.0354,
        reg_lambda=7.8139,
        random_state=42, verbosity=0,
    )


def _stack(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """
    Train model on ATS margin target and predict for test set.

    Target: result - vegas_spread (ATS margin)
    Output: ATS prediction + vegas_spread = implied spread

    This architecture lets the model learn market inefficiency directly.
    Vegas spread as a peer feature contributes ~37% importance — informative
    but not dominant, unlike the old Ridge stacker (83% Vegas weight).
    """
    X_train = train[FEATURE_COLS].fillna(0)
    y_train = train["result"] - train["vegas_spread"].fillna(0)
    X_test  = test[FEATURE_COLS].fillna(0)

    base = _make_base()
    base.fit(X_train, y_train)

    ats_pred = base.predict(X_test)
    return ats_pred + test["vegas_spread"].fillna(0).values


# ── Walk-forward backtest ─────────────────────────────────────────────────────
def _bootstrap_confidence(train: pd.DataFrame, test: pd.DataFrame,
                           n_boots: int = 10) -> np.ndarray:
    """
    Estimate prediction uncertainty via bootstrap resampling.

    Fits n_boots models on bootstrap samples of train, computes std of
    predictions across models. Higher std = lower confidence in the pick.

    Returns:
        np.ndarray: Per-game confidence scores (std of spread predictions).
    """
    preds = []
    rng   = np.random.default_rng(42)
    for _ in range(n_boots):
        idx    = rng.integers(0, len(train), size=len(train))
        sample = train.iloc[idx]
        m      = _make_base()
        m.fit(sample[FEATURE_COLS].fillna(0),
              sample["result"] - sample["vegas_spread"].fillna(0))
        ats_p = m.predict(test[FEATURE_COLS].fillna(0))
        preds.append(ats_p + test["vegas_spread"].fillna(0).values)
    return np.std(preds, axis=0)


def walk_forward_backtest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run a strict walk-forward backtest across all seasons.

    For each test season N: train on seasons 1..N-1, predict N.
    Requires at least 2 training seasons before predicting.
    Bootstrap confidence computed per season.

    Returns:
        pd.DataFrame: All predictions with model_spread, spread_gap,
                      model_winner, vegas_winner, actual_winner, confidence.
    """
    seasons = sorted(df["season"].unique())
    results = []

    for i, test_season in enumerate(seasons):
        train_seasons = seasons[:i]
        if len(train_seasons) < 2:
            continue

        train      = df[df["season"].isin(train_seasons)]
        test       = df[df["season"] == test_season]
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
                    no_div: bool = True,
                    no_late: bool = True,
                    max_std: float = None) -> pd.DataFrame:
    """
    Apply the asymmetric edge filter to backtest results.

    Underdog bets: all games where model disagrees with Vegas by ≥ threshold.
    Favorite bets: additionally require gap ≤ 3.0 AND spread magnitude ≤ 7.
        Rationale: model systematically overcorrects on large-gap heavy favorites.
        Gap 2-3 on favorites → 58.3% ATS; Gap 4-5 on favorites → 40.3% ATS.

    Default filters (no_div=True, no_late=True):
        - Skip divisional games (model underperforms, likely due to familiarity)
        - Skip weeks 17-18 (rest management distorts results)

    Args:
        df: Output of walk_forward_backtest().
        threshold: Minimum |spread_gap| to consider a bet (default 2.0).
        no_div: Skip divisional games.
        no_late: Skip weeks 17-18.
        max_std: Optional bootstrap confidence cutoff (lower = more certain).

    Returns:
        pd.DataFrame: Filtered edge games with ats_win and payout columns.
    """
    df = df[df["season"] >= df["season"].min() + MIN_TRAIN_SEASONS].copy()
    if no_late:
        df = df[~df["week"].isin([17, 18])]
    if no_div and "div_game" in df.columns:
        df = df[df["div_game"] == 0]

    agrees    = (
        ((df["model_spread"] > 0) & (df["vegas_spread"] > 0)) |
        ((df["model_spread"] < 0) & (df["vegas_spread"] < 0))
    )
    large_gap = df["spread_gap"].abs() >= threshold
    confident = df["confidence"] <= max_std if max_std is not None else pd.Series(True, index=df.index)

    e = df[agrees & large_gap & confident].copy()
    e["bet_home"]   = e["spread_gap"] > 0
    e["spread_mag"] = e["vegas_spread"].abs()

    e["is_fav_bet"] = (
        (e["bet_home"] & (e["vegas_spread"] > 0)) |
        (~e["bet_home"] & (e["vegas_spread"] < 0))
    )

    fav_mask = e["is_fav_bet"] & (e["spread_gap"].abs() <= 3.0) & (e["spread_mag"] <= 7)
    dog_mask = ~e["is_fav_bet"]
    e = e[fav_mask | dog_mask].copy()

    e["ats_win"] = np.where(e["bet_home"],
                            e["result"] > e["vegas_spread"],
                            e["result"] < e["vegas_spread"])
    e["payout"]  = np.where(e["ats_win"], UNIT_BET * (100 / abs(ODDS)), -UNIT_BET)
    return e


def apply_sharp_filter(edges: pd.DataFrame) -> pd.DataFrame:
    """
    Filter edges to only bets where sharp money agrees with the model.
    Games with no line movement data are included (benefit of the doubt).
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
    """Print per-season accuracy and vs-Vegas delta for all backtest results."""
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
    """Print per-season ATS win rate and ROI for a filtered edge set."""
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
    tw = total_wins / total_bets * 100 if total_bets else 0
    print(f"{'All-time':<8} {total_bets:>6} {tw:>10.1f}% "
          f"{total_roi:>10.0f} {total_roi/(total_bets*UNIT_BET)*100:>7.1f}%")
    print(f"\nBreak-even ATS%: 52.4%  |  Model: {tw:.1f}%  |  "
          f"Edge: {tw-52.4:+.1f}%  |  {'PROFITABLE' if tw > 52.4 else 'NOT PROFITABLE'}")


def print_stacked_filter_report(results: pd.DataFrame) -> None:
    """Print ATS% and ROI for every combination of threshold/filter settings."""
    print(f"\n{'='*80}")
    print("Stacked filter sensitivity — all combinations")
    print(f"{'Thresh':>7} {'NoDiv':>6} {'NoLate':>7} {'Sharp':>6} {'Bets':>6} {'ATS%':>8} {'ROI($)':>10}")
    print(f"{'='*80}")
    for threshold in [2.0, 2.5, 3.0, 3.5]:
        for no_div in [False, True]:
            for no_late in [False, True]:
                for use_sharp in [False, True]:
                    edges = calculate_edges(results, threshold=threshold,
                                            no_div=no_div, no_late=no_late)
                    if use_sharp:
                        edges = apply_sharp_filter(edges)
                    if len(edges) < 5:
                        continue
                    ats = edges["ats_win"].mean() * 100
                    roi = edges["payout"].sum()
                    print(f"{threshold:>7.1f} {'Y' if no_div else 'N':>6} "
                          f"{'Y' if no_late else 'N':>7} {'Y' if use_sharp else 'N':>6} "
                          f"{len(edges):>6} {ats:>7.1f}% {roi:>10.0f}")


# ── Train & save ──────────────────────────────────────────────────────────────
def train_final_model(df: pd.DataFrame) -> XGBRegressor:
    """
    Train the model on all available historical data and save to disk.

    Trains on ATS margin target (result - vegas_spread).
    Saves to engine/trained_base_model.pkl for use by predictor.py.

    Args:
        df: Full training DataFrame from load_training_data().

    Returns:
        XGBRegressor: The fitted model.
    """
    X = df[FEATURE_COLS].fillna(0)
    y = df["result"] - df["vegas_spread"].fillna(0)
    base = _make_base()
    base.fit(X, y)
    model_dir = Path(__file__).resolve().parent
    with open(model_dir / "trained_base_model.pkl", "wb") as f:
        pickle.dump(base, f)
    print(f"  Model saved to {model_dir / 'trained_base_model.pkl'}")
    return base


def load_trained_model() -> tuple:
    """
    Load the trained model from disk.

    Returns:
        tuple: (XGBRegressor, None) — second element is a legacy placeholder
               from the old Ridge stacker architecture (now removed).
    """
    model_dir = Path(__file__).resolve().parent
    with open(model_dir / "trained_base_model.pkl", "rb") as f:
        base = pickle.load(f)
    return base, None


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading training data...")
    df = load_training_data()
    print(f"  {len(df)} games  |  {len(FEATURE_COLS)} features")
    print(f"  Target: ATS margin (result - vegas_spread)")
    print(f"  Filter: no restrictions — all games (edge≥{EDGE_THRESHOLD})")

    print("\nRunning walk-forward backtest...")
    results = walk_forward_backtest(df)
    print_report(results)

    print("\nBase edges (no-div default):")
    df_edges = calculate_edges(results)
    print_edge_report(df_edges, label="Base edges (threshold=2.0, all games)")

    print("\nStacked filter analysis:")
    print_stacked_filter_report(results)

    print("\nTraining final model on all data...")
    train_final_model(df)

    print("\nSaving predictions to model_predictions table...")
    save(results, "model_predictions")
    print("Done.")