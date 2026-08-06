-- postgres_schema.sql — Jaginator database schema for Postgres.
--
-- Translated from the original SQLite jaginator.db, with two additions
-- SQLite couldn't give us: real PRIMARY KEY / FOREIGN KEY enforcement, and
-- indexes on every join/filter column identified from the actual API and
-- pipeline queries.
--
-- Design notes:
--   - REAL (SQLite, 8-byte float) -> DOUBLE PRECISION (not Postgres REAL,
--     which is single precision and would silently lose accuracy on
--     rounded values like ELO ratings and EPA figures).
--   - *_current tables (schedules_current, player_stats_current, etc.) are
--     replaced wholesale every week via DROP+CREATE (pandas to_sql
--     if_exists='replace'). Postgres blocks DROP TABLE while a foreign key
--     references it, so nothing in this schema has an FK pointing at a
--     *_current table, and those tables themselves have no PK — they're
--     read-heavy scratch tables, not relationally anchored data.
--   - Tables without a clean natural key (rosters_current,
--     depth_charts_current, injuries_current, injuries_historical,
--     line_movement, qb/rb/wr_prop_predictions) get a SERIAL surrogate id
--     instead of a composite PK, since source data doesn't guarantee
--     uniqueness on any real column combination.
--
-- Usage:
--   psql "$DATABASE_URL" -f postgres_schema.sql

BEGIN;

-- ── Historical: schedules and the tables split off it ─────────────────────────
CREATE TABLE IF NOT EXISTS schedules (
    game_id     TEXT PRIMARY KEY,
    season      INTEGER NOT NULL,
    game_type   TEXT,
    week        INTEGER,
    gameday     TEXT,
    weekday     TEXT,
    gametime    TEXT,
    away_team   TEXT,
    home_team   TEXT,
    away_score  INTEGER,
    home_score  INTEGER,
    location    TEXT,
    result      INTEGER,
    total       INTEGER,
    overtime    INTEGER,
    div_game    INTEGER,
    away_rest   INTEGER,
    home_rest   INTEGER
);
CREATE INDEX IF NOT EXISTS idx_schedules_season_week ON schedules(season, week);

CREATE TABLE IF NOT EXISTS game_odds (
    game_id           TEXT PRIMARY KEY REFERENCES schedules(game_id),
    away_moneyline    DOUBLE PRECISION,
    home_moneyline    DOUBLE PRECISION,
    spread_line       DOUBLE PRECISION,
    away_spread_odds  DOUBLE PRECISION,
    home_spread_odds  DOUBLE PRECISION,
    total_line        DOUBLE PRECISION,
    under_odds        DOUBLE PRECISION,
    over_odds         DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS game_conditions (
    game_id     TEXT PRIMARY KEY REFERENCES schedules(game_id),
    roof        TEXT,
    surface     TEXT,
    temp        DOUBLE PRECISION,
    wind        DOUBLE PRECISION,
    stadium_id  TEXT,
    stadium     TEXT
);

CREATE TABLE IF NOT EXISTS game_personnel (
    game_id       TEXT PRIMARY KEY REFERENCES schedules(game_id),
    away_qb_id    TEXT,
    home_qb_id    TEXT,
    away_qb_name  TEXT,
    home_qb_name  TEXT,
    away_coach    TEXT,
    home_coach    TEXT,
    referee       TEXT
);

-- ── Historical: season-level team stats ────────────────────────────────────────
CREATE TABLE IF NOT EXISTS team_stats (
    season      INTEGER NOT NULL,
    team        TEXT NOT NULL,
    season_type TEXT NOT NULL,
    games                       INTEGER,
    completions                INTEGER, attempts INTEGER,
    passing_yards               INTEGER, passing_tds INTEGER, passing_interceptions INTEGER,
    sacks_suffered              INTEGER, sack_yards_lost INTEGER,
    sack_fumbles                INTEGER, sack_fumbles_lost INTEGER,
    passing_air_yards           INTEGER, passing_yards_after_catch INTEGER,
    passing_first_downs         INTEGER, passing_epa DOUBLE PRECISION, passing_cpoe DOUBLE PRECISION,
    passing_2pt_conversions     INTEGER,
    carries                     INTEGER, rushing_yards INTEGER, rushing_tds INTEGER,
    rushing_fumbles             INTEGER, rushing_fumbles_lost INTEGER,
    rushing_first_downs         INTEGER, rushing_epa DOUBLE PRECISION, rushing_2pt_conversions INTEGER,
    receptions                  INTEGER, targets INTEGER,
    receiving_yards             INTEGER, receiving_tds INTEGER,
    receiving_fumbles           INTEGER, receiving_fumbles_lost INTEGER,
    receiving_air_yards         INTEGER, receiving_yards_after_catch INTEGER,
    receiving_first_downs       INTEGER, receiving_epa DOUBLE PRECISION, receiving_2pt_conversions INTEGER,
    special_teams_tds           INTEGER,
    def_tackles_solo            INTEGER, def_tackles_with_assist INTEGER, def_tackle_assists INTEGER,
    def_tackles_for_loss        INTEGER, def_tackles_for_loss_yards INTEGER,
    def_fumbles_forced          INTEGER, def_sacks DOUBLE PRECISION, def_sack_yards DOUBLE PRECISION,
    def_qb_hits                 INTEGER, def_interceptions INTEGER, def_interception_yards INTEGER,
    def_pass_defended           INTEGER, def_tds INTEGER, def_fumbles INTEGER, def_safeties INTEGER,
    misc_yards                  INTEGER,
    fumble_recovery_own         INTEGER, fumble_recovery_yards_own INTEGER,
    fumble_recovery_opp         INTEGER, fumble_recovery_yards_opp INTEGER, fumble_recovery_tds INTEGER,
    penalties                   INTEGER, penalty_yards INTEGER, timeouts INTEGER,
    punt_returns                INTEGER, punt_return_yards INTEGER,
    kickoff_returns              INTEGER, kickoff_return_yards INTEGER,
    fg_made INTEGER, fg_att INTEGER, fg_missed INTEGER, fg_blocked INTEGER,
    fg_long DOUBLE PRECISION, fg_pct DOUBLE PRECISION,
    fg_made_0_19 INTEGER, fg_made_20_29 INTEGER, fg_made_30_39 INTEGER,
    fg_made_40_49 INTEGER, fg_made_50_59 INTEGER, fg_made_60_ INTEGER,
    fg_missed_0_19 INTEGER, fg_missed_20_29 INTEGER, fg_missed_30_39 INTEGER,
    fg_missed_40_49 INTEGER, fg_missed_50_59 INTEGER, fg_missed_60_ INTEGER,
    fg_made_list TEXT, fg_missed_list TEXT, fg_blocked_list TEXT,
    fg_made_distance INTEGER, fg_missed_distance INTEGER, fg_blocked_distance INTEGER,
    pat_made INTEGER, pat_att INTEGER, pat_missed INTEGER, pat_blocked INTEGER, pat_pct DOUBLE PRECISION,
    gwfg_made INTEGER, gwfg_att INTEGER, gwfg_missed INTEGER, gwfg_blocked INTEGER,
    gwfg_distance_list TEXT,
    PRIMARY KEY (season, team, season_type)
);

-- ── Historical: per-player-game stats (large table — see indexes below) ───────
CREATE TABLE IF NOT EXISTS player_stats (
    player_id            TEXT NOT NULL,
    game_id              TEXT REFERENCES schedules(game_id),  -- sparse: only populated 2022+
    player_name          TEXT, player_display_name TEXT,
    position             TEXT, position_group TEXT, headshot_url TEXT,
    season               INTEGER NOT NULL, week INTEGER NOT NULL, season_type TEXT NOT NULL,
    team                 TEXT, opponent_team TEXT,
    completions          INTEGER, attempts INTEGER,
    passing_yards        INTEGER, passing_tds INTEGER, passing_interceptions INTEGER,
    sacks_suffered       INTEGER, sack_yards_lost INTEGER,
    sack_fumbles         INTEGER, sack_fumbles_lost INTEGER,
    passing_air_yards    INTEGER, passing_yards_after_catch INTEGER,
    passing_first_downs  INTEGER, passing_epa DOUBLE PRECISION, passing_cpoe DOUBLE PRECISION,
    passing_2pt_conversions INTEGER, pacr DOUBLE PRECISION,
    carries              INTEGER, rushing_yards INTEGER, rushing_tds INTEGER,
    rushing_fumbles      INTEGER, rushing_fumbles_lost INTEGER,
    rushing_first_downs  INTEGER, rushing_epa DOUBLE PRECISION, rushing_2pt_conversions INTEGER,
    receptions           INTEGER, targets INTEGER,
    receiving_yards      INTEGER, receiving_tds INTEGER,
    receiving_fumbles    INTEGER, receiving_fumbles_lost INTEGER,
    receiving_air_yards  INTEGER, receiving_yards_after_catch INTEGER,
    receiving_first_downs INTEGER, receiving_epa DOUBLE PRECISION, receiving_2pt_conversions INTEGER,
    racr DOUBLE PRECISION, target_share DOUBLE PRECISION, air_yards_share DOUBLE PRECISION, wopr DOUBLE PRECISION,
    special_teams_tds    INTEGER,
    def_tackles_solo     INTEGER, def_tackles_with_assist INTEGER, def_tackle_assists INTEGER,
    def_tackles_for_loss INTEGER, def_tackles_for_loss_yards INTEGER,
    def_fumbles_forced   INTEGER, def_sacks DOUBLE PRECISION, def_sack_yards DOUBLE PRECISION,
    def_qb_hits          INTEGER, def_interceptions INTEGER, def_interception_yards INTEGER,
    def_pass_defended    INTEGER, def_tds INTEGER, def_fumbles INTEGER, def_safeties INTEGER,
    misc_yards           INTEGER,
    fumble_recovery_own  INTEGER, fumble_recovery_yards_own INTEGER,
    fumble_recovery_opp  INTEGER, fumble_recovery_yards_opp INTEGER, fumble_recovery_tds INTEGER,
    penalties            INTEGER, penalty_yards INTEGER,
    punt_returns         INTEGER, punt_return_yards INTEGER,
    kickoff_returns      INTEGER, kickoff_return_yards INTEGER,
    fg_made INTEGER, fg_att INTEGER, fg_missed INTEGER, fg_blocked INTEGER,
    fg_long DOUBLE PRECISION, fg_pct DOUBLE PRECISION,
    fg_made_0_19 INTEGER, fg_made_20_29 INTEGER, fg_made_30_39 INTEGER,
    fg_made_40_49 INTEGER, fg_made_50_59 INTEGER, fg_made_60_ INTEGER,
    fg_missed_0_19 INTEGER, fg_missed_20_29 INTEGER, fg_missed_30_39 INTEGER,
    fg_missed_40_49 INTEGER, fg_missed_50_59 INTEGER, fg_missed_60_ INTEGER,
    fg_made_list TEXT, fg_missed_list TEXT, fg_blocked_list TEXT,
    fg_made_distance INTEGER, fg_missed_distance INTEGER, fg_blocked_distance INTEGER,
    pat_made INTEGER, pat_att INTEGER, pat_missed INTEGER, pat_blocked INTEGER, pat_pct DOUBLE PRECISION,
    gwfg_made INTEGER, gwfg_att INTEGER, gwfg_missed INTEGER, gwfg_blocked INTEGER, gwfg_distance INTEGER,
    fantasy_points DOUBLE PRECISION, fantasy_points_ppr DOUBLE PRECISION,
    PRIMARY KEY (player_id, season, week, season_type)
);
CREATE INDEX IF NOT EXISTS idx_player_stats_player_season ON player_stats(player_id, season, week);
CREATE INDEX IF NOT EXISTS idx_player_stats_game ON player_stats(game_id);

-- ── Historical: injuries ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS injuries_historical (
    id                          SERIAL PRIMARY KEY,
    season                      DOUBLE PRECISION,
    game_type                   TEXT,
    team                        TEXT,
    week                        DOUBLE PRECISION,
    gsis_id                     TEXT,
    position                    TEXT,
    full_name                   TEXT, first_name TEXT, last_name TEXT,
    report_primary_injury       TEXT, report_secondary_injury TEXT, report_status TEXT,
    practice_primary_injury     TEXT, practice_secondary_injury TEXT, practice_status TEXT,
    date_modified               TIMESTAMP,
    season_type                 TEXT
);
CREATE INDEX IF NOT EXISTS idx_injuries_historical_season_week_team
    ON injuries_historical(season, week, team);

-- ── Historical: ELO, rolling stats, features, defense ──────────────────────────
CREATE TABLE IF NOT EXISTS elo_history (
    game_id     TEXT NOT NULL REFERENCES schedules(game_id),
    season      INTEGER NOT NULL,
    week        INTEGER,
    team        TEXT NOT NULL,
    opponent    TEXT,
    elo_before  DOUBLE PRECISION,
    elo_after   DOUBLE PRECISION,
    PRIMARY KEY (game_id, team)
);
CREATE INDEX IF NOT EXISTS idx_elo_history_team_season_week ON elo_history(team, season, week);

CREATE TABLE IF NOT EXISTS team_rolling_stats (
    game_id                 TEXT NOT NULL REFERENCES schedules(game_id),
    team                    TEXT NOT NULL,
    roll_passing_epa        DOUBLE PRECISION, roll_rushing_epa DOUBLE PRECISION,
    roll_passing_yards      DOUBLE PRECISION, roll_rushing_yards DOUBLE PRECISION,
    roll_passing_tds        DOUBLE PRECISION, roll_interceptions DOUBLE PRECISION,
    roll_def_sacks          DOUBLE PRECISION, roll_def_ints DOUBLE PRECISION,
    roll_def_fumbles        DOUBLE PRECISION, roll_def_tfl DOUBLE PRECISION,
    roll_def_qb_hits        DOUBLE PRECISION, roll_def_pass_def DOUBLE PRECISION,
    roll_turnovers          DOUBLE PRECISION, roll_first_downs DOUBLE PRECISION,
    roll_penalty_yards      DOUBLE PRECISION, roll_avg_margin DOUBLE PRECISION,
    roll_margin_trend       DOUBLE PRECISION, roll_blowout_wins DOUBLE PRECISION,
    roll_blowout_losses     DOUBLE PRECISION, roll_streak INTEGER,
    roll_coach_win_rate     DOUBLE PRECISION, roll_coach_games INTEGER,
    PRIMARY KEY (game_id, team)
);

CREATE TABLE IF NOT EXISTS player_features (
    game_id  TEXT PRIMARY KEY REFERENCES schedules(game_id),
    season   INTEGER, week INTEGER,
    home_qb_epa DOUBLE PRECISION, home_qb_games INTEGER, home_qb_injured INTEGER,
    home_wr_yards DOUBLE PRECISION, home_wr_target_share DOUBLE PRECISION, home_wr_injured INTEGER,
    home_te_yards DOUBLE PRECISION, home_te_target_share DOUBLE PRECISION, home_te_injured INTEGER,
    home_rb_rush_yards DOUBLE PRECISION, home_rb_rec_yards DOUBLE PRECISION,
    home_rb_carry_share DOUBLE PRECISION, home_rb_injured INTEGER,
    away_qb_epa DOUBLE PRECISION, away_qb_games INTEGER, away_qb_injured INTEGER,
    away_wr_yards DOUBLE PRECISION, away_wr_target_share DOUBLE PRECISION, away_wr_injured INTEGER,
    away_te_yards DOUBLE PRECISION, away_te_target_share DOUBLE PRECISION, away_te_injured INTEGER,
    away_rb_rush_yards DOUBLE PRECISION, away_rb_rec_yards DOUBLE PRECISION,
    away_rb_carry_share DOUBLE PRECISION, away_rb_injured INTEGER
);

CREATE TABLE IF NOT EXISTS defense_ratings (
    season INTEGER, week INTEGER,
    game_id TEXT NOT NULL REFERENCES schedules(game_id),
    team    TEXT NOT NULL,
    roll_adj_def_pass_epa DOUBLE PRECISION, roll_adj_def_rush_epa DOUBLE PRECISION,
    roll_adj_def_wr_epa   DOUBLE PRECISION, roll_adj_def_te_epa   DOUBLE PRECISION,
    roll_adj_def_rb_epa   DOUBLE PRECISION,
    def_games_played DOUBLE PRECISION,
    pass_def_rank DOUBLE PRECISION, rush_def_rank DOUBLE PRECISION,
    wr_def_rank   DOUBLE PRECISION, te_def_rank   DOUBLE PRECISION, rb_def_rank DOUBLE PRECISION,
    PRIMARY KEY (game_id, team)
);

-- ── Historical: moneyline / line movement (matched to schedules post-hoc) ─────
CREATE TABLE IF NOT EXISTS line_movement (
    id SERIAL PRIMARY KEY,
    away_team TEXT, home_team TEXT,
    away_score INTEGER, home_score INTEGER,
    away_ml INTEGER, home_ml INTEGER,
    open_spread DOUBLE PRECISION, close_spread DOUBLE PRECISION,
    open_total DOUBLE PRECISION, close_total DOUBLE PRECISION,
    spread_move DOUBLE PRECISION, total_move DOUBLE PRECISION,
    sharp_home INTEGER, sharp_away INTEGER, line_moved INTEGER,
    season INTEGER, sbro_idx INTEGER,
    game_id TEXT REFERENCES schedules(game_id),
    week INTEGER
);

-- ── Predictions ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS model_predictions (
    game_id      TEXT PRIMARY KEY REFERENCES schedules(game_id),
    season       INTEGER, week INTEGER,
    home_team    TEXT, away_team TEXT,
    result       INTEGER,
    vegas_spread DOUBLE PRECISION, div_game INTEGER,
    model_spread DOUBLE PRECISION, spread_gap DOUBLE PRECISION, confidence DOUBLE PRECISION,
    model_winner TEXT, vegas_winner TEXT, actual_winner TEXT
);
CREATE INDEX IF NOT EXISTS idx_model_predictions_season_week ON model_predictions(season, week);

CREATE TABLE IF NOT EXISTS playoff_predictions (
    game_id             TEXT PRIMARY KEY REFERENCES schedules(game_id),
    season              INTEGER, week INTEGER,
    home_team           TEXT, away_team TEXT,
    home_win_prob       DOUBLE PRECISION, away_win_prob DOUBLE PRECISION,
    playoff_pick        TEXT, playoff_confidence DOUBLE PRECISION,
    features_json       TEXT
);

-- weekly_predictions/prop predictions reference the *current* (upcoming,
-- unplayed) week's games via schedules_current, which is replaced wholesale
-- every run — so no FK here (see docstring note in sql_helpers.py).
CREATE TABLE IF NOT EXISTS weekly_predictions (
    game_id           TEXT PRIMARY KEY,
    season            INTEGER, week INTEGER,
    gameday           TEXT, gametime TEXT,
    home_team         TEXT, away_team TEXT,
    model_spread      DOUBLE PRECISION, vegas_spread DOUBLE PRECISION, spread_gap DOUBLE PRECISION,
    home_win_prob     DOUBLE PRECISION, away_win_prob DOUBLE PRECISION,
    predicted_winner  TEXT,
    generated_at      TEXT,
    confidence        DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS qb_prop_predictions (
    id SERIAL PRIMARY KEY,
    player_name TEXT, team TEXT, opponent TEXT, week INTEGER, is_home INTEGER,
    pred_passing_yards DOUBLE PRECISION, pred_passing_tds DOUBLE PRECISION, pred_attempts DOUBLE PRECISION,
    implied_team_total DOUBLE PRECISION, spread_line DOUBLE PRECISION,
    game_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_qb_prop_predictions_week ON qb_prop_predictions(week);

CREATE TABLE IF NOT EXISTS rb_prop_predictions (
    id SERIAL PRIMARY KEY,
    player_name TEXT, team TEXT, opponent TEXT, week INTEGER, is_home INTEGER,
    pred_rushing_yards DOUBLE PRECISION, pred_rushing_tds DOUBLE PRECISION,
    pred_carries DOUBLE PRECISION, pred_receptions DOUBLE PRECISION, pred_receiving_yards DOUBLE PRECISION,
    implied_team_total DOUBLE PRECISION, spread_line DOUBLE PRECISION,
    game_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_rb_prop_predictions_week ON rb_prop_predictions(week);

CREATE TABLE IF NOT EXISTS wr_prop_predictions (
    id SERIAL PRIMARY KEY,
    player_name TEXT, team TEXT, opponent TEXT, week INTEGER, is_home INTEGER,
    pred_receiving_yards DOUBLE PRECISION, pred_receiving_tds DOUBLE PRECISION,
    pred_receptions DOUBLE PRECISION, pred_targets DOUBLE PRECISION,
    implied_team_total DOUBLE PRECISION, spread_line DOUBLE PRECISION,
    game_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_wr_prop_predictions_week ON wr_prop_predictions(week);

-- ── Current (replaced wholesale weekly — no PK/FK, see docstring note) ─────────
CREATE TABLE IF NOT EXISTS schedules_current (
    game_id TEXT, season INTEGER, game_type TEXT, week INTEGER,
    gameday TEXT, weekday TEXT, gametime TEXT,
    away_team TEXT, away_score INTEGER, home_team TEXT, home_score INTEGER,
    location TEXT, result INTEGER, total INTEGER, overtime INTEGER,
    old_game_id TEXT, gsis INTEGER, nfl_detail_id TEXT, pfr TEXT, pff DOUBLE PRECISION,
    espn TEXT, ftn DOUBLE PRECISION,
    away_rest INTEGER, home_rest INTEGER,
    away_moneyline INTEGER, home_moneyline INTEGER,
    spread_line DOUBLE PRECISION, away_spread_odds INTEGER, home_spread_odds INTEGER,
    total_line DOUBLE PRECISION, under_odds INTEGER, over_odds INTEGER,
    div_game INTEGER, roof TEXT, surface TEXT, temp DOUBLE PRECISION, wind DOUBLE PRECISION,
    away_qb_id TEXT, home_qb_id TEXT, away_qb_name TEXT, home_qb_name TEXT,
    away_coach TEXT, home_coach TEXT, referee TEXT,
    stadium_id TEXT, stadium TEXT
);
CREATE INDEX IF NOT EXISTS idx_schedules_current_game_id ON schedules_current(game_id);

CREATE TABLE IF NOT EXISTS player_stats_current (
    player_id TEXT, game_id TEXT,
    player_name TEXT, player_display_name TEXT,
    position TEXT, position_group TEXT, headshot_url TEXT,
    season INTEGER, week INTEGER, season_type TEXT,
    team TEXT, opponent_team TEXT,
    completions INTEGER, attempts INTEGER,
    passing_yards INTEGER, passing_tds INTEGER, passing_interceptions INTEGER,
    sacks_suffered INTEGER, sack_yards_lost INTEGER,
    sack_fumbles INTEGER, sack_fumbles_lost INTEGER,
    passing_air_yards INTEGER, passing_yards_after_catch INTEGER,
    passing_first_downs INTEGER, passing_epa DOUBLE PRECISION, passing_cpoe DOUBLE PRECISION,
    passing_2pt_conversions INTEGER, pacr DOUBLE PRECISION,
    carries INTEGER, rushing_yards INTEGER, rushing_tds INTEGER,
    rushing_fumbles INTEGER, rushing_fumbles_lost INTEGER,
    rushing_first_downs INTEGER, rushing_epa DOUBLE PRECISION, rushing_2pt_conversions INTEGER,
    receptions INTEGER, targets INTEGER,
    receiving_yards INTEGER, receiving_tds INTEGER,
    receiving_fumbles INTEGER, receiving_fumbles_lost INTEGER,
    receiving_air_yards INTEGER, receiving_yards_after_catch INTEGER,
    receiving_first_downs INTEGER, receiving_epa DOUBLE PRECISION, receiving_2pt_conversions INTEGER,
    racr DOUBLE PRECISION, target_share DOUBLE PRECISION, air_yards_share DOUBLE PRECISION, wopr DOUBLE PRECISION,
    special_teams_tds INTEGER,
    def_tackles_solo INTEGER, def_tackles_with_assist INTEGER, def_tackle_assists INTEGER,
    def_tackles_for_loss INTEGER, def_tackles_for_loss_yards INTEGER,
    def_fumbles_forced INTEGER, def_sacks DOUBLE PRECISION, def_sack_yards DOUBLE PRECISION,
    def_qb_hits INTEGER, def_interceptions INTEGER, def_interception_yards INTEGER,
    def_pass_defended INTEGER, def_tds INTEGER, def_fumbles INTEGER, def_safeties INTEGER,
    misc_yards INTEGER,
    fumble_recovery_own INTEGER, fumble_recovery_yards_own INTEGER,
    fumble_recovery_opp INTEGER, fumble_recovery_yards_opp INTEGER, fumble_recovery_tds INTEGER,
    penalties INTEGER, penalty_yards INTEGER,
    punt_returns INTEGER, punt_return_yards INTEGER,
    kickoff_returns INTEGER, kickoff_return_yards INTEGER,
    fg_made INTEGER, fg_att INTEGER, fg_missed INTEGER, fg_blocked INTEGER,
    fg_long DOUBLE PRECISION, fg_pct DOUBLE PRECISION,
    fg_made_0_19 INTEGER, fg_made_20_29 INTEGER, fg_made_30_39 INTEGER,
    fg_made_40_49 INTEGER, fg_made_50_59 INTEGER, fg_made_60_ INTEGER,
    fg_missed_0_19 INTEGER, fg_missed_20_29 INTEGER, fg_missed_30_39 INTEGER,
    fg_missed_40_49 INTEGER, fg_missed_50_59 INTEGER, fg_missed_60_ INTEGER,
    fg_made_list TEXT, fg_missed_list TEXT, fg_blocked_list TEXT,
    fg_made_distance INTEGER, fg_missed_distance INTEGER, fg_blocked_distance INTEGER,
    pat_made INTEGER, pat_att INTEGER, pat_missed INTEGER, pat_blocked INTEGER, pat_pct DOUBLE PRECISION,
    gwfg_made INTEGER, gwfg_att INTEGER, gwfg_missed INTEGER, gwfg_blocked INTEGER, gwfg_distance INTEGER,
    fantasy_points DOUBLE PRECISION, fantasy_points_ppr DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS idx_player_stats_current_game ON player_stats_current(game_id);

CREATE TABLE IF NOT EXISTS team_stats_current (
    season INTEGER, week INTEGER, team TEXT, season_type TEXT,
    game_id TEXT, opponent_team TEXT,
    completions INTEGER, attempts INTEGER,
    passing_yards INTEGER, passing_tds INTEGER, passing_interceptions INTEGER,
    sacks_suffered INTEGER, sack_yards_lost INTEGER,
    sack_fumbles INTEGER, sack_fumbles_lost INTEGER,
    passing_air_yards INTEGER, passing_yards_after_catch INTEGER,
    passing_first_downs INTEGER, passing_epa DOUBLE PRECISION, passing_cpoe DOUBLE PRECISION,
    passing_2pt_conversions INTEGER,
    carries INTEGER, rushing_yards INTEGER, rushing_tds INTEGER,
    rushing_fumbles INTEGER, rushing_fumbles_lost INTEGER,
    rushing_first_downs INTEGER, rushing_epa DOUBLE PRECISION, rushing_2pt_conversions INTEGER,
    receptions INTEGER, targets INTEGER,
    receiving_yards INTEGER, receiving_tds INTEGER,
    receiving_fumbles INTEGER, receiving_fumbles_lost INTEGER,
    receiving_air_yards INTEGER, receiving_yards_after_catch INTEGER,
    receiving_first_downs INTEGER, receiving_epa DOUBLE PRECISION, receiving_2pt_conversions INTEGER,
    special_teams_tds INTEGER,
    def_tackles_solo INTEGER, def_tackles_with_assist INTEGER, def_tackle_assists INTEGER,
    def_tackles_for_loss INTEGER, def_tackles_for_loss_yards INTEGER,
    def_fumbles_forced INTEGER, def_sacks DOUBLE PRECISION, def_sack_yards DOUBLE PRECISION,
    def_qb_hits INTEGER, def_interceptions INTEGER, def_interception_yards INTEGER,
    def_pass_defended INTEGER, def_tds INTEGER, def_fumbles INTEGER, def_safeties INTEGER,
    misc_yards INTEGER,
    fumble_recovery_own INTEGER, fumble_recovery_yards_own INTEGER,
    fumble_recovery_opp INTEGER, fumble_recovery_yards_opp INTEGER, fumble_recovery_tds INTEGER,
    penalties INTEGER, penalty_yards INTEGER, timeouts INTEGER,
    punt_returns INTEGER, punt_return_yards INTEGER,
    kickoff_returns INTEGER, kickoff_return_yards INTEGER,
    fg_made INTEGER, fg_att INTEGER, fg_missed INTEGER, fg_blocked INTEGER,
    fg_long DOUBLE PRECISION, fg_pct DOUBLE PRECISION,
    fg_made_0_19 INTEGER, fg_made_20_29 INTEGER, fg_made_30_39 INTEGER,
    fg_made_40_49 INTEGER, fg_made_50_59 INTEGER, fg_made_60_ INTEGER,
    fg_missed_0_19 INTEGER, fg_missed_20_29 INTEGER, fg_missed_30_39 INTEGER,
    fg_missed_40_49 INTEGER, fg_missed_50_59 INTEGER, fg_missed_60_ INTEGER,
    fg_made_list TEXT, fg_missed_list TEXT, fg_blocked_list TEXT,
    fg_made_distance INTEGER, fg_missed_distance INTEGER, fg_blocked_distance INTEGER,
    pat_made INTEGER, pat_att INTEGER, pat_missed INTEGER, pat_blocked INTEGER, pat_pct DOUBLE PRECISION,
    gwfg_made INTEGER, gwfg_att INTEGER, gwfg_missed INTEGER, gwfg_blocked INTEGER, gwfg_distance INTEGER
);
CREATE INDEX IF NOT EXISTS idx_team_stats_current_game_team ON team_stats_current(game_id, team);

CREATE TABLE IF NOT EXISTS team_stats_weekly (
    season INTEGER, week INTEGER, team TEXT, season_type TEXT, opponent_team TEXT,
    completions INTEGER, attempts INTEGER,
    passing_yards INTEGER, passing_tds INTEGER, passing_interceptions INTEGER,
    sacks_suffered INTEGER, sack_yards_lost INTEGER,
    sack_fumbles INTEGER, sack_fumbles_lost INTEGER,
    passing_air_yards INTEGER, passing_yards_after_catch INTEGER,
    passing_first_downs INTEGER, passing_epa DOUBLE PRECISION, passing_cpoe DOUBLE PRECISION,
    passing_2pt_conversions INTEGER,
    carries INTEGER, rushing_yards INTEGER, rushing_tds INTEGER,
    rushing_fumbles INTEGER, rushing_fumbles_lost INTEGER,
    rushing_first_downs INTEGER, rushing_epa DOUBLE PRECISION, rushing_2pt_conversions INTEGER,
    receptions INTEGER, targets INTEGER,
    receiving_yards INTEGER, receiving_tds INTEGER,
    receiving_fumbles INTEGER, receiving_fumbles_lost INTEGER,
    receiving_air_yards INTEGER, receiving_yards_after_catch INTEGER,
    receiving_first_downs INTEGER, receiving_epa DOUBLE PRECISION, receiving_2pt_conversions INTEGER,
    special_teams_tds INTEGER,
    def_tackles_solo INTEGER, def_tackles_with_assist INTEGER, def_tackle_assists INTEGER,
    def_tackles_for_loss INTEGER, def_tackles_for_loss_yards INTEGER,
    def_fumbles_forced INTEGER, def_sacks DOUBLE PRECISION, def_sack_yards DOUBLE PRECISION,
    def_qb_hits INTEGER, def_interceptions INTEGER, def_interception_yards INTEGER,
    def_pass_defended INTEGER, def_tds INTEGER, def_fumbles INTEGER, def_safeties INTEGER,
    misc_yards INTEGER,
    fumble_recovery_own INTEGER, fumble_recovery_yards_own INTEGER,
    fumble_recovery_opp INTEGER, fumble_recovery_yards_opp INTEGER, fumble_recovery_tds INTEGER,
    penalties INTEGER, penalty_yards INTEGER, timeouts INTEGER,
    punt_returns INTEGER, punt_return_yards INTEGER,
    kickoff_returns INTEGER, kickoff_return_yards INTEGER,
    fg_made INTEGER, fg_att INTEGER, fg_missed INTEGER, fg_blocked INTEGER,
    fg_long DOUBLE PRECISION, fg_pct DOUBLE PRECISION,
    fg_made_0_19 INTEGER, fg_made_20_29 INTEGER, fg_made_30_39 INTEGER,
    fg_made_40_49 INTEGER, fg_made_50_59 INTEGER, fg_made_60_ INTEGER,
    fg_missed_0_19 INTEGER, fg_missed_20_29 INTEGER, fg_missed_30_39 INTEGER,
    fg_missed_40_49 INTEGER, fg_missed_50_59 INTEGER, fg_missed_60_ INTEGER,
    fg_made_list TEXT, fg_missed_list TEXT, fg_blocked_list TEXT,
    fg_made_distance INTEGER, fg_missed_distance INTEGER, fg_blocked_distance INTEGER,
    pat_made INTEGER, pat_att INTEGER, pat_missed INTEGER, pat_blocked INTEGER, pat_pct DOUBLE PRECISION,
    gwfg_made INTEGER, gwfg_att INTEGER, gwfg_missed INTEGER, gwfg_blocked INTEGER, gwfg_distance INTEGER,
    game_id TEXT REFERENCES schedules(game_id)
);
CREATE INDEX IF NOT EXISTS idx_team_stats_weekly_season_week ON team_stats_weekly(season, week, team);

CREATE TABLE IF NOT EXISTS rosters_current (
    id SERIAL PRIMARY KEY,
    season INTEGER, team TEXT, position TEXT, depth_chart_position TEXT,
    jersey_number DOUBLE PRECISION, status TEXT,
    full_name TEXT, first_name TEXT, last_name TEXT, birth_date TIMESTAMP,
    height DOUBLE PRECISION, weight INTEGER, college TEXT,
    gsis_id TEXT, espn_id TEXT, sportradar_id TEXT, yahoo_id TEXT, rotowire_id TEXT,
    pff_id TEXT, pfr_id TEXT, fantasy_data_id TEXT, sleeper_id TEXT,
    years_exp INTEGER, headshot_url TEXT, ngs_position TEXT,
    week INTEGER, game_type TEXT, status_description_abbr TEXT, football_name TEXT,
    esb_id TEXT, gsis_it_id TEXT, smart_id TEXT,
    entry_year INTEGER, rookie_year INTEGER, draft_club TEXT, draft_number DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS idx_rosters_current_team ON rosters_current(team);

CREATE TABLE IF NOT EXISTS depth_charts_current (
    id SERIAL PRIMARY KEY,
    team TEXT, position TEXT, depth_order INTEGER,
    player_id TEXT, player_name TEXT, fetched_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_depth_charts_current_team ON depth_charts_current(team);

CREATE TABLE IF NOT EXISTS injuries_current (
    id SERIAL PRIMARY KEY,
    team_display TEXT, player_name TEXT, position TEXT,
    status TEXT, comment TEXT, date TEXT
);

COMMIT;