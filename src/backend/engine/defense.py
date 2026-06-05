"""
defense.py — Opponent-adjusted defensive EPA ratings (DVOA-style).

Builds rolling pre-game defensive quality ratings for every team and game,
adjusting for opponent offensive strength. A defense that holds an elite
offense to average results looks better than one that holds a bad offense
to the same results. This is the key insight from Football Outsiders' DVOA.

Pipeline (build_defense_ratings):
    1. Load play-by-play from nflverse (run/pass plays, garbage time filtered)
    2. Build offensive EPA/play baselines per team (rolling, shift(1) for leakage)
    3. Build position-specific offensive baselines (WR/TE/RB targets via roster join)
    4. Calculate raw defensive EPA allowed per game
    5. Opponent-adjust: raw_def - opponent_off_baseline
    6. Roll the adjusted values over ROLL_WINDOW (4) games
    7. Apply regression to mean for early-season small samples
    8. Add season-level defensive ranks across all 32 teams

Key design decisions:
    - Garbage time filter: wp outside 5%-95% excluded (prevent/stat-padding distort EPA)
    - shift(1) everywhere: only pre-game information used, no leakage
    - Roll AFTER adjustment: adj(game1) + adj(game2) / 2, not adj(avg(game1, game2))
    - Regression weights: {1 game: 25%, 2: 45%, 3: 70%, 4+: 100%} toward league mean
    - EPA is zero-sum so league average = 0 is always the correct Bayesian prior

Output table: defense_ratings
    One row per (team, season, week) with roll_adj_def_pass_epa,
    roll_adj_def_rush_epa, roll_adj_def_wr/te/rb_epa, def_games_played,
    and pass/rush/wr/te/rb defense ranks.

Usage:
    python3.11 engine/defense.py                   # rebuild all seasons
    python3.11 engine/defense.py --current-only    # current season only
    python3.11 engine/defense.py --seasons 2023 2024 2025
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
import nflreadpy as nfl
from data.sql_helpers import get_connection, save

# ── Constants ─────────────────────────────────────────────────────────────────
MIN_SEASON   = 1999
ROLL_WINDOW  = 4      # 4-game rolling window — same as props models for consistency
MIN_PLAYS    = 10     # minimum plays for a game to count toward a team's rating
                      # filters out games where a team barely ran offense (garbage time blowouts)

# Garbage time filter — DVOA excludes plays where the game is already decided.
# wp < 0.05 means the offense is down huge and just throwing to run up stats.
# wp > 0.95 means the defense is sitting in prevent and yards don't mean anything.
# Both situations inflate raw yardage stats without representing real performance.
WP_MIN = 0.05
WP_MAX = 0.95

# Regression to mean weight for early season games.
# With only 1-2 games of data, a team's rating is extremely noisy.
# We blend toward zero (league average) when sample is small.
# 4+ games = full weight on observed data, fewer = partial blend toward 0.
REGRESSION_GAMES = {1: 0.25, 2: 0.45, 3: 0.70, 4: 1.0}

# Seasons where EPA model is less reliable — nflverse EPA goes back to 1999
# but the model was trained on modern data so early seasons have more noise.
# We still include them but note the limitation.
EPA_RELIABLE_FROM = 2006  # nflfastR EPA model validated from this season


# ── Load play-by-play data ────────────────────────────────────────────────────
def load_pbp(seasons: list = None) -> pd.DataFrame:
    """
    Load play-by-play from nflverse for all seasons.

    Why load_pbp and not player_stats?
    player_stats gives us game-level aggregates (yards, TDs per game).
    PBP gives us EPA per play, which accounts for down/distance/field position.
    A 5-yard gain on 3rd-and-4 (success, positive EPA) is completely different
    from a 5-yard gain on 3rd-and-15 (failure, negative EPA). Game-level stats
    cannot distinguish these situations.

    Why filter to play_type in ('run', 'pass')?
    Kickoffs, punts, spikes, kneels, and PATs all have EPA values but they
    don't measure offensive/defensive quality in the way we care about.
    Kneels in particular have massive negative EPA and would distort ratings.

    Why filter on wp?
    Win probability outside 5%-95% indicates garbage time. In garbage time:
    - Defenses play prevent coverage (inflate pass yards)
    - Offenses run to kill clock or throw deep to catch up (distort EPA)
    - The plays don't represent normal game situations
    DVOA uses a similar filter (excludes final minutes of blowouts).
    """
    if seasons is None:
        seasons = list(range(MIN_SEASON, 2026))

    print(f"Loading PBP for {len(seasons)} seasons...")
    pbp = nfl.load_pbp(seasons).to_pandas()

    # Filter to regular season only — postseason has different game dynamics
    # and sample sizes are too small to be meaningful for rolling averages
    pbp = pbp[pbp["season_type"] == "REG"].copy()

    # Filter to actual offensive plays
    pbp = pbp[pbp["play_type"].isin(["run", "pass"])].copy()

    # Must have EPA — some plays are missing EPA due to data issues
    pbp = pbp[pbp["epa"].notna()].copy()

    # Must have down — eliminates any remaining non-plays
    pbp = pbp[pbp["down"].notna()].copy()

    # Garbage time filter
    # We use notna() check first because wp can be null for early seasons
    wp_valid = pbp["wp"].notna()
    not_garbage = (pbp["wp"] >= WP_MIN) & (pbp["wp"] <= WP_MAX)
    pbp = pbp[~wp_valid | not_garbage].copy()

    print(f"  {len(pbp):,} qualifying plays loaded")
    # receiver_position is not in PBP — we use receiver_id and join to rosters
    # to get position. Keep receiver_id for the join.
    keep = ["season", "week", "game_id", "posteam", "defteam",
            "play_type", "epa", "wp", "down", "ydstogo",
            "yardline_100", "receiver_player_id",
            "rusher_player_id", "passer_player_id"]
    # receiver_player_id may also be named receiver_id depending on version
    for alt in ["receiver_id", "receiver_player_id"]:
        if alt in pbp.columns and alt not in keep:
            keep.append(alt)
    return pbp[[c for c in keep if c in pbp.columns]].copy()


# ── Calculate offensive baseline ──────────────────────────────────────────────
def build_offensive_baselines(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate each team's offensive EPA/play per game, then rolling average.

    Why do we need an offensive baseline?
    This is the core of opponent-adjustment. If the Bills allow 5 yards/carry
    against the Chiefs but the Chiefs average 6 yards/carry against everyone,
    that's actually a good defensive performance. Without knowing the opponent's
    offensive quality, we can't tell good from bad.

    Why EPA/play and not yards/play?
    EPA already accounts for down, distance, and field position. A 3-yard gain
    on 3rd-and-2 is worth more EPA than a 3-yard gain on 3rd-and-10. Yards
    treat both identically. EPA is a better measure of true offensive quality.

    Why shift(1)?
    We're building a pre-game rating. We can only use games that have already
    been played — never the current game. shift(1) ensures we use data from
    weeks 1 through (n-1) to predict week n, never using week n itself.
    This is identical to the leakage prevention in the props models.

    Why separate pass and rush baselines?
    A team's pass offense quality is the right baseline for evaluating pass
    defense. A team's rush offense quality is the right baseline for rush defense.
    Using overall EPA/play would mix the two and obscure the signal.
    """
    # Game-level offensive EPA/play by play type
    off_game = (
        pbp.groupby(["season", "week", "game_id", "posteam", "play_type"])["epa"]
           .agg(["mean", "count"])
           .reset_index()
           .rename(columns={"mean": "epa_per_play", "count": "n_plays", "posteam": "team"})
    )

    # Filter out games with too few plays — less than MIN_PLAYS plays of a given
    # type means the team barely used that strategy (e.g. team only ran 3 times)
    off_game = off_game[off_game["n_plays"] >= MIN_PLAYS].copy()

    # Pivot to get pass_epa and rush_epa per game per team
    off_pivot = off_game.pivot_table(
        index=["season", "week", "game_id", "team"],
        columns="play_type",
        values="epa_per_play"
    ).reset_index()
    off_pivot.columns.name = None
    off_pivot = off_pivot.rename(columns={"pass": "off_pass_epa", "run": "off_rush_epa"})
    off_pivot["off_pass_epa"] = off_pivot["off_pass_epa"].fillna(0)
    off_pivot["off_rush_epa"] = off_pivot["off_rush_epa"].fillna(0)

    # Rolling offensive baseline — shifted to avoid leakage
    off_pivot = off_pivot.sort_values(["team", "season", "week"]).reset_index(drop=True)
    for col in ["off_pass_epa", "off_rush_epa"]:
        off_pivot[f"roll_{col}"] = (
            off_pivot.groupby("team")[col]
                     .transform(lambda x: x.shift(1).rolling(ROLL_WINDOW, min_periods=1).mean())
        )

    # Fill nulls with league average (0.0 in EPA terms, since EPA is zero-sum)
    # Why 0.0? EPA is constructed so league average is always 0 by definition.
    off_pivot["roll_off_pass_epa"] = off_pivot["roll_off_pass_epa"].fillna(0.0)
    off_pivot["roll_off_rush_epa"] = off_pivot["roll_off_rush_epa"].fillna(0.0)

    return off_pivot[["season", "week", "team",
                       "roll_off_pass_epa", "roll_off_rush_epa"]]


# ── Calculate receiver-position pass defense ──────────────────────────────────
def build_receiver_position_baselines(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate each team's offensive EPA/play when targeting WR, TE, and RB.

    receiver_position is not directly in nflverse PBP. We join to rosters
    using receiver_player_id to get each receiver's position.
    """
    pass_plays = pbp[pbp["play_type"] == "pass"].copy()

    # Get receiver id column — may be receiver_player_id or receiver_id
    rec_id_col = "receiver_player_id" if "receiver_player_id" in pass_plays.columns                  else "receiver_id" if "receiver_id" in pass_plays.columns else None

    if rec_id_col is None or pass_plays[rec_id_col].isna().all():
        # No receiver ID available — return empty frame, caller handles fillna
        print("  Warning: no receiver_id column found, skipping position splits")
        return pd.DataFrame(columns=["season","week","team",
                                     "roll_off_wr_epa","roll_off_te_epa","roll_off_rb_rec_epa"])

    # Load rosters to get position for each receiver_id
    seasons = sorted(pass_plays["season"].unique().tolist())
    try:
        rosters = nfl.load_rosters(seasons).to_pandas()
        # Keep gsis_id (nflverse player id) and position
        id_col = "gsis_id" if "gsis_id" in rosters.columns else                  "player_id" if "player_id" in rosters.columns else None
        if id_col:
            pos_map = rosters.dropna(subset=[id_col]).set_index(id_col)["position"].to_dict()
            pass_plays["rec_pos_group"] = pass_plays[rec_id_col].map(pos_map).map(
                lambda p: "WR" if p == "WR"
                     else "TE" if p == "TE"
                     else "RB" if p in ("RB", "FB", "HB")
                     else "other"
            ).fillna("other")
        else:
            pass_plays["rec_pos_group"] = "other"
    except Exception as e:
        print(f"  Warning: roster join failed ({e}), skipping position splits")
        return pd.DataFrame(columns=["season","week","team",
                                     "roll_off_wr_epa","roll_off_te_epa","roll_off_rb_rec_epa"])

    pass_plays = pass_plays[pass_plays["rec_pos_group"] != "other"].copy()

    # Game-level offensive EPA by receiver position
    off_by_pos = (
        pass_plays.groupby(["season", "week", "game_id", "posteam", "rec_pos_group"])["epa"]
                  .agg(["mean", "count"])
                  .reset_index()
                  .rename(columns={"mean": "epa_per_play", "count": "n_plays",
                                   "posteam": "team"})
    )
    off_by_pos = off_by_pos[off_by_pos["n_plays"] >= 3].copy()  # min 3 targets

    # Pivot by position group
    off_pos_pivot = off_by_pos.pivot_table(
        index=["season", "week", "game_id", "team"],
        columns="rec_pos_group",
        values="epa_per_play"
    ).reset_index()
    off_pos_pivot.columns.name = None
    for col in ["WR", "TE", "RB"]:
        if col not in off_pos_pivot.columns:
            off_pos_pivot[col] = 0.0
        off_pos_pivot[col] = off_pos_pivot[col].fillna(0.0)
    off_pos_pivot = off_pos_pivot.rename(columns={
        "WR": "off_wr_epa", "TE": "off_te_epa", "RB": "off_rb_rec_epa"
    })

    # Rolling baselines
    off_pos_pivot = off_pos_pivot.sort_values(["team", "season", "week"]).reset_index(drop=True)
    for col in ["off_wr_epa", "off_te_epa", "off_rb_rec_epa"]:
        off_pos_pivot[f"roll_{col}"] = (
            off_pos_pivot.groupby("team")[col]
                         .transform(lambda x: x.shift(1).rolling(ROLL_WINDOW, min_periods=1).mean())
        ).fillna(0.0)

    return off_pos_pivot[["season", "week", "team",
                           "roll_off_wr_epa", "roll_off_te_epa", "roll_off_rb_rec_epa"]]


# ── Build raw defensive EPA per game ─────────────────────────────────────────
def build_raw_defense(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate raw defensive EPA/play allowed per game.

    Why use defteam's perspective?
    EPA is calculated from the offensive team's perspective (positive = good for offense).
    So when we group by defteam, higher EPA allowed = worse defense.
    We keep it in this orientation so the opponent adjustment math is intuitive:
    opponent_adj_def = raw_def_epa_allowed - opponent_off_baseline
    Positive = defense allowed more than expected (bad defense)
    Negative = defense held opponent below expected (good defense)

    This matches DVOA's convention where negative DVOA = better defense.
    """
    def_game = (
        pbp.groupby(["season", "week", "game_id", "defteam", "play_type"])["epa"]
           .agg(["mean", "count"])
           .reset_index()
           .rename(columns={"mean": "epa_allowed", "count": "n_plays",
                            "defteam": "team"})
    )
    def_game = def_game[def_game["n_plays"] >= MIN_PLAYS].copy()

    def_pivot = def_game.pivot_table(
        index=["season", "week", "game_id", "team"],
        columns="play_type",
        values="epa_allowed"
    ).reset_index()
    def_pivot.columns.name = None
    def_pivot = def_pivot.rename(columns={"pass": "def_pass_epa", "run": "def_rush_epa"})
    def_pivot["def_pass_epa"] = def_pivot["def_pass_epa"].fillna(0.0)
    def_pivot["def_rush_epa"] = def_pivot["def_rush_epa"].fillna(0.0)

    # Receiver-position defensive EPA
    pass_plays = pbp[pbp["play_type"] == "pass"].copy()
    rec_id_col = "receiver_player_id" if "receiver_player_id" in pass_plays.columns                  else "receiver_id" if "receiver_id" in pass_plays.columns else None

    if rec_id_col and not pass_plays[rec_id_col].isna().all():
        try:
            seasons = sorted(pass_plays["season"].unique().tolist())
            rosters = nfl.load_rosters(seasons).to_pandas()
            id_col  = "gsis_id" if "gsis_id" in rosters.columns else                       "player_id" if "player_id" in rosters.columns else None
            if id_col:
                pos_map = rosters.dropna(subset=[id_col]).set_index(id_col)["position"].to_dict()
                pass_plays["rec_pos_group"] = pass_plays[rec_id_col].map(pos_map).map(
                    lambda p: "WR" if p == "WR"
                         else "TE" if p == "TE"
                         else "RB" if p in ("RB","FB","HB")
                         else "other"
                ).fillna("other")
            else:
                pass_plays["rec_pos_group"] = "other"
        except Exception as e:
            print(f"  Warning: roster join failed ({e})")
            pass_plays["rec_pos_group"] = "other"
    else:
        pass_plays["rec_pos_group"] = "other"
    pass_plays = pass_plays[pass_plays["rec_pos_group"] != "other"].copy()

    def_by_pos = (
        pass_plays.groupby(["season", "week", "game_id", "defteam", "rec_pos_group"])["epa"]
                  .agg(["mean", "count"])
                  .reset_index()
                  .rename(columns={"mean": "epa_allowed", "count": "n_plays",
                                   "defteam": "team"})
    )
    def_by_pos = def_by_pos[def_by_pos["n_plays"] >= 3].copy()

    def_pos_pivot = def_by_pos.pivot_table(
        index=["season", "week", "game_id", "team"],
        columns="rec_pos_group",
        values="epa_allowed"
    ).reset_index()
    def_pos_pivot.columns.name = None
    for col in ["WR", "TE", "RB"]:
        if col not in def_pos_pivot.columns:
            def_pos_pivot[col] = 0.0
        def_pos_pivot[col] = def_pos_pivot[col].fillna(0.0)
    def_pos_pivot = def_pos_pivot.rename(columns={
        "WR": "def_wr_epa", "TE": "def_te_epa", "RB": "def_rb_rec_epa"
    })

    def_pivot = def_pivot.merge(
        def_pos_pivot[["season", "week", "game_id", "team",
                       "def_wr_epa", "def_te_epa", "def_rb_rec_epa"]],
        on=["season", "week", "game_id", "team"], how="left"
    )
    for col in ["def_wr_epa", "def_te_epa", "def_rb_rec_epa"]:
        def_pivot[col] = def_pivot[col].fillna(0.0)

    return def_pivot


# ── Opponent-adjust defense ───────────────────────────────────────────────────
def opponent_adjust(def_game: pd.DataFrame,
                    off_baseline: pd.DataFrame,
                    off_pos_baseline: pd.DataFrame) -> pd.DataFrame:
    """
    Subtract the opponent's offensive baseline from the raw defensive EPA.

    The math:
        adj_def_pass_epa = def_pass_epa_allowed - opponent_roll_off_pass_epa

    Example:
        Bills allow 0.15 EPA/pass play against the Chiefs.
        Chiefs' rolling offensive pass EPA baseline = 0.20 (elite offense).
        Adjusted Bills pass defense = 0.15 - 0.20 = -0.05 (good defense)

        Bills allow 0.15 EPA/pass play against the Patriots.
        Patriots' rolling offensive pass EPA baseline = -0.10 (bad offense).
        Adjusted Bills pass defense = 0.15 - (-0.10) = 0.25 (bad defense)

    Same raw number (0.15), completely different defensive quality.
    This is exactly what DVOA does — adjust for opponent quality.

    Why merge on opponent_team?
    Each defensive game row has a defteam. We need the OPPONENT's offensive
    baseline — the team that was running plays against that defense. We get
    this by merging off_baseline on (season, week, posteam=opponent).
    """
    # Get opponent team for each defensive game
    # The opponent is the posteam (the team running offense against this defense)
    def_game = def_game.copy()

    # Extract opponent from game_id — game_id format is YYYY_WW_away_home
    # We know defteam, so opponent = the other team in the game
    # Merge via schedule to get posteam
    with get_connection() as conn:
        schedule = pd.read_sql("""
            SELECT game_id, season, week, home_team, away_team
            FROM schedules WHERE game_type = 'REG'
        """, conn)

    # Map defteam -> posteam (opponent)
    home_def = schedule.rename(columns={"home_team": "team", "away_team": "opponent"})
    away_def = schedule.rename(columns={"away_team": "team", "home_team": "opponent"})
    team_opponents = pd.concat([
        home_def[["game_id", "season", "week", "team", "opponent"]],
        away_def[["game_id", "season", "week", "team", "opponent"]],
    ], ignore_index=True)

    def_game = def_game.merge(team_opponents, on=["game_id", "season", "week", "team"], how="left")

    # Merge opponent's offensive baseline
    # Note: off_baseline uses the opponent's ROLLING baseline (from previous games)
    # so this is truly pre-game information — no leakage
    opp_off = off_baseline.rename(columns={
        "team": "opponent",
        "roll_off_pass_epa": "opp_off_pass_epa",
        "roll_off_rush_epa": "opp_off_rush_epa",
    })
    def_game = def_game.merge(
        opp_off, on=["season", "week", "opponent"], how="left"
    )
    def_game["opp_off_pass_epa"] = def_game["opp_off_pass_epa"].fillna(0.0)
    def_game["opp_off_rush_epa"] = def_game["opp_off_rush_epa"].fillna(0.0)

    # Merge opponent's position-specific offensive baseline
    opp_pos = off_pos_baseline.rename(columns={
        "team": "opponent",
        "roll_off_wr_epa":     "opp_off_wr_epa",
        "roll_off_te_epa":     "opp_off_te_epa",
        "roll_off_rb_rec_epa": "opp_off_rb_rec_epa",
    })
    def_game = def_game.merge(
        opp_pos, on=["season", "week", "opponent"], how="left"
    )
    for col in ["opp_off_wr_epa", "opp_off_te_epa", "opp_off_rb_rec_epa"]:
        def_game[col] = def_game[col].fillna(0.0)

    # Opponent-adjusted defensive EPA
    # Lower = better defense (held opponent below their offensive baseline)
    def_game["adj_def_pass_epa"] = def_game["def_pass_epa"] - def_game["opp_off_pass_epa"]
    def_game["adj_def_rush_epa"] = def_game["def_rush_epa"] - def_game["opp_off_rush_epa"]
    def_game["adj_def_wr_epa"]   = def_game["def_wr_epa"]   - def_game["opp_off_wr_epa"]
    def_game["adj_def_te_epa"]   = def_game["def_te_epa"]   - def_game["opp_off_te_epa"]
    def_game["adj_def_rb_epa"]   = def_game["def_rb_rec_epa"] - def_game["opp_off_rb_rec_epa"]

    return def_game


# ── Rolling defensive ratings ─────────────────────────────────────────────────
def build_rolling_defense(def_adj: pd.DataFrame) -> pd.DataFrame:
    """
    Roll the opponent-adjusted defensive EPA over the last ROLL_WINDOW games.

    Why roll AFTER opponent adjustment?
    Rolling raw EPA and then adjusting would blend the adjustments incorrectly.
    We want: roll(adj_epa) not adj(roll(raw_epa)).
    Each game's adjusted value should be computed first, then averaged.

    Why apply regression to mean for early games?
    With 1 game of data, a defense that happened to face a hot offense looks
    amazing. With 2 games, still very noisy. DVOA uses a Bayesian prior that
    pulls early-season ratings toward league average. We implement a simplified
    version: blend toward 0.0 (league average EPA) based on games played.

    Why shift(1)?
    Same reason as offensive baseline — we need pre-game ratings only.
    The defensive rating for week N uses weeks 1 through N-1.

    Output columns:
        roll_adj_def_pass_epa  — rolling opponent-adjusted pass defense
        roll_adj_def_rush_epa  — rolling opponent-adjusted rush defense
        roll_adj_def_wr_epa    — rolling opponent-adjusted WR coverage
        roll_adj_def_te_epa    — rolling opponent-adjusted TE coverage
        roll_adj_def_rb_epa    — rolling opponent-adjusted RB receiving coverage
        def_games_played       — games in rolling window (for regression weight)
    """
    def_adj = def_adj.sort_values(["team", "season", "week"]).reset_index(drop=True)

    adj_cols = ["adj_def_pass_epa", "adj_def_rush_epa",
                "adj_def_wr_epa",   "adj_def_te_epa",   "adj_def_rb_epa"]

    for col in adj_cols:
        rolled = (
            def_adj.groupby("team")[col]
                   .transform(lambda x: x.shift(1).rolling(ROLL_WINDOW, min_periods=1).mean())
        )
        n_games = (
            def_adj.groupby("team")[col]
                   .transform(lambda x: x.shift(1).rolling(ROLL_WINDOW, min_periods=1).count())
        )

        # Regression to mean — blend toward 0 for small samples
        reg_weight = n_games.map(
            lambda n: REGRESSION_GAMES.get(int(n), 1.0) if pd.notna(n) else 0.25
        )
        def_adj[f"roll_{col}"] = (rolled * reg_weight).fillna(0.0)

    def_adj["def_games_played"] = (
        def_adj.groupby("team")["adj_def_pass_epa"]
               .transform(lambda x: x.shift(1).rolling(ROLL_WINDOW, min_periods=1).count())
    ).fillna(0)

    return def_adj


# ── Season defense rank ───────────────────────────────────────────────────────
def add_season_ranks(df: pd.DataFrame) -> pd.DataFrame:
    for col, rank_col in [
        ("roll_adj_def_pass_epa", "pass_def_rank"),
        ("roll_adj_def_rush_epa", "rush_def_rank"),
        ("roll_adj_def_wr_epa",   "wr_def_rank"),
        ("roll_adj_def_te_epa",   "te_def_rank"),
        ("roll_adj_def_rb_epa",   "rb_def_rank"),
    ]:
        # Get the last rolling value per team per season
        season_summary = (
            df.sort_values(["season", "team", "week"])
              .groupby(["season", "team"])[col]
              .last()
              .reset_index()
        )
        # Rank across the 32 teams within each season
        season_summary[rank_col] = (
            season_summary.groupby("season")[col]
                          .rank(ascending=True, method="min")
        )
        # Broadcast rank back to every row for that team+season
        df = df.merge(season_summary[["season", "team", rank_col]],
                      on=["season", "team"], how="left")
        df[rank_col] = df[rank_col].fillna(16)

    return df

# ── Main build function ───────────────────────────────────────────────────────
def build_defense_ratings(seasons: list = None) -> pd.DataFrame:
    """
    Full pipeline: PBP → offensive baselines → raw defense →
    opponent adjustment → rolling window → season ranks → save to DB.

    Output table: defense_ratings
    One row per (team, season, week) representing pre-game defensive quality.
    """
    pbp = load_pbp(seasons)

    print("Building offensive baselines...")
    off_baseline     = build_offensive_baselines(pbp)
    off_pos_baseline = build_receiver_position_baselines(pbp)

    print("Building raw defensive EPA per game...")
    def_raw = build_raw_defense(pbp)

    print("Opponent-adjusting defensive ratings...")
    def_adj = opponent_adjust(def_raw, off_baseline, off_pos_baseline)

    print("Building rolling defensive ratings...")
    def_rolling = build_rolling_defense(def_adj)

    print("Adding season ranks...")
    def_final = add_season_ranks(def_rolling)

    # Select final output columns
    output_cols = [
        "season", "week", "game_id", "team",
        "roll_adj_def_pass_epa", "roll_adj_def_rush_epa",
        "roll_adj_def_wr_epa",   "roll_adj_def_te_epa",
        "roll_adj_def_rb_epa",   "def_games_played",
        "pass_def_rank", "rush_def_rank",
        "wr_def_rank",   "te_def_rank",   "rb_def_rank",
    ]
    def_final = def_final[[c for c in output_cols if c in def_final.columns]]

    print(f"Built ratings for {len(def_final):,} team-game rows")
    print(f"Seasons: {sorted(def_final['season'].unique())}")

    return def_final


def load_defense_ratings() -> pd.DataFrame:
    """Load pre-computed defense ratings from DB."""
    with get_connection() as conn:
        return pd.read_sql("SELECT * FROM defense_ratings ORDER BY season, week", conn)


def get_current_defense_ratings() -> pd.DataFrame:
    """
    Get the most recent defensive rating per team.
    Used by predict_current_week() in each props model.
    Returns one row per team with their latest rolling ratings.
    """
    with get_connection() as conn:
        return pd.read_sql("""
            SELECT dr.*
            FROM defense_ratings dr
            INNER JOIN (
                SELECT team, MAX(season * 100 + week) AS max_key
                FROM defense_ratings GROUP BY team
            ) latest ON dr.team = latest.team
              AND (dr.season * 100 + dr.week) = latest.max_key
        """, conn)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", type=int,
                        help="Specific seasons to build (default: all)")
    parser.add_argument("--current-only", action="store_true",
                        help="Only rebuild current season")
    args = parser.parse_args()

    if args.current_only:
        from nflreadpy.utils_date import get_current_season
        seasons = [get_current_season()]
    else:
        seasons = args.seasons  # None = all seasons

    print("Building opponent-adjusted defense ratings...")
    df = build_defense_ratings(seasons)

    print("\nSample output:")
    sample = df[df["season"] == df["season"].max()].head(10)
    print(sample[["season", "week", "team",
                   "roll_adj_def_pass_epa", "roll_adj_def_rush_epa",
                   "pass_def_rank", "rush_def_rank"]].to_string(index=False))

    print("\nSaving to DB...")
    save(df, "defense_ratings")
    print("Done.")

    # Sanity checks
    print("\nSanity checks:")
    latest = df[df["season"] == df["season"].max()]
    print(f"  Teams with ratings: {latest['team'].nunique()}")
    print(f"  EPA range pass: {df['roll_adj_def_pass_epa'].min():.3f} to {df['roll_adj_def_pass_epa'].max():.3f}")
    print(f"  EPA range rush: {df['roll_adj_def_rush_epa'].min():.3f} to {df['roll_adj_def_rush_epa'].max():.3f}")
    print(f"  Null check: {df[['roll_adj_def_pass_epa','roll_adj_def_rush_epa']].isna().sum().sum()} nulls")