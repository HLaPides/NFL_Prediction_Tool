import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import numpy as np
from data.sql_helpers import get_connection

TEAM_MAP = {
    # Current teams
    'Arizona':'ARI',      'Atlanta':'ATL',       'Baltimore':'BAL',
    'Buffalo':'BUF',      'Carolina':'CAR',      'Chicago':'CHI',
    'Cincinnati':'CIN',   'Cleveland':'CLE',     'Dallas':'DAL',
    'Denver':'DEN',       'Detroit':'DET',       'GreenBay':'GB',
    'Houston':'HOU',      'Indianapolis':'IND',  'Jacksonville':'JAX',
    'KansasCity':'KC',    'Miami':'MIA',         'Minnesota':'MIN',
    'NewEngland':'NE',    'NewOrleans':'NO',     'NYGiants':'NYG',
    'NYJets':'NYJ',       'Philadelphia':'PHI',  'Pittsburgh':'PIT',
    'SanFrancisco':'SF',  'Seattle':'SEA',       'TampaBay':'TB',
    'Tennessee':'TEN',    'Washington':'WAS',
    # Historical — must match nflverse historical abbreviations
    'Oakland':'OAK',      'SanDiego':'SD',       'StLouis':'STL',
    'St.Louis':'STL',
    # Current relocated teams
    'LasVegas':'LV',      'LAChargers':'LAC',    'LARams':'LA',
    'LosAngeles':'LA',    'LVRaiders':'LV',
    # Alternate spellings
    'HoustonTexans':'HOU','BuffaloBills':'BUF',  'KCChiefs':'KC',
    'NewYork':'NYG',      'Tampa':'TB',          'Washingtom':'WAS',
    'Kansas':'KC',
}


def parse_sbro_file(filepath: str, season: int) -> pd.DataFrame:
    raw = pd.read_html(filepath, flavor='lxml')[0]

    # Row 0 is a header row embedded in the data
    raw = raw.iloc[1:].copy().reset_index(drop=True)
    raw.columns = ['date','rot','vh','team','q1','q2','q3','q4',
                   'final','open','close','ml','2h']
    raw = raw[raw['vh'].isin(['V','H'])].copy().reset_index(drop=True)

    visitors = raw[raw['vh']=='V'].copy().reset_index(drop=True)
    homes    = raw[raw['vh']=='H'].copy().reset_index(drop=True)

    # Track rows where visitor opened at pick-em
    v_was_pk = visitors['open'] == 'pk'

    for col in ['open','close']:
        visitors[col] = visitors[col].replace('pk', '0')
        homes[col]    = homes[col].replace('pk', '0')

    v_open  = pd.to_numeric(visitors['open'],  errors='coerce').fillna(99)
    v_close = pd.to_numeric(visitors['close'], errors='coerce').fillna(99)
    h_open  = pd.to_numeric(homes['open'],     errors='coerce').fillna(99)
    h_close = pd.to_numeric(homes['close'],    errors='coerce').fillna(99)

    # Determine which row contains the spread vs total by magnitude
    # Spread is between -30 and 30, total is between 30 and 70
    v_is_spread = v_open.abs() <= 30

    open_spread  = np.where(v_is_spread, v_open,  h_open)
    close_spread = np.where(v_is_spread, v_close, h_close)
    open_total   = np.where(v_is_spread, h_open,  v_open)
    close_total  = np.where(v_is_spread, h_close, v_close)

    # Edge case: when V opened at pk, close column has the total not the spread
    open_spread  = np.where(v_was_pk, 0,       open_spread)
    close_spread = np.where(v_was_pk, h_close, close_spread)
    open_total   = np.where(v_was_pk, h_open,  open_total)
    close_total  = np.where(v_was_pk, v_close, close_total)

    # Convert to home-team perspective (negative = home favored)
    open_spread  = np.where(v_is_spread | v_was_pk, -open_spread,  open_spread)
    close_spread = np.where(v_is_spread | v_was_pk, -close_spread, close_spread)

    games = pd.DataFrame()
    games['away_team']    = visitors['team'].map(TEAM_MAP).fillna(visitors['team'])
    games['home_team']    = homes['team'].map(TEAM_MAP).fillna(homes['team'])
    games['away_score']   = pd.to_numeric(visitors['final'], errors='coerce')
    games['home_score']   = pd.to_numeric(homes['final'],    errors='coerce')
    games['away_ml']      = pd.to_numeric(visitors['ml'],    errors='coerce')
    games['home_ml']      = pd.to_numeric(homes['ml'],       errors='coerce')
    games['open_spread']  = open_spread.astype(float)
    games['close_spread'] = close_spread.astype(float)
    games['open_total']   = open_total.astype(float)
    games['close_total']  = close_total.astype(float)
    games['spread_move']  = games['close_spread'] - games['open_spread']
    games['total_move']   = games['close_total']  - games['open_total']
    games['sharp_home']   = (games['spread_move'] > 0).astype(int)
    games['sharp_away']   = (games['spread_move'] < 0).astype(int)
    games['line_moved']   = (games['spread_move'].abs() >= 0.5).astype(int)

    games = games.dropna(subset=['open_spread','close_spread','away_score','home_score'])

    # Filter out rows where spread_move is unrealistically large
    # (indicates total line bleeding into spread column)
    games = games[games['spread_move'].abs() <= 10]

    games['season'] = int(season)
    return games


def load_all_sbro(data_dir: str = None) -> pd.DataFrame:
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent

    files = sorted(Path(data_dir).glob('nfl_*.htm'))
    if not files:
        print(f"No nfl_*.htm files found in {data_dir}")
        print("Save SBRO files as: nfl_2008.htm, nfl_2009.htm, etc.")
        return pd.DataFrame()

    frames = []
    for f in files:
        try:
            # SBRO names files by end year (2008 = 2007-08 season)
            # nflverse uses start year, so subtract 1
            season = int(f.stem.split('_')[1]) - 1
        except (IndexError, ValueError):
            print(f"Skipping {f.name} — can't parse season from filename")
            continue

        print(f"  Parsing {f.name} (season {season})...")
        try:
            df = parse_sbro_file(str(f), season)
            print(f"    {len(df)} games parsed")
            frames.append(df)
        except Exception as e:
            print(f"    Error: {e}")

    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    result['season'] = result['season'].astype(int)
    return result


def match_to_schedules(df: pd.DataFrame) -> pd.DataFrame:
    with get_connection() as conn:
        schedules = pd.read_sql("""
            SELECT game_id, season, week, home_team, away_team
            FROM schedules
            WHERE game_type IN ('REG','WC','DIV','CON','SB')
        """, conn)

    schedules['season'] = schedules['season'].astype(int)
    df = df.copy()
    df['season'] = df['season'].astype(int)
    df['sbro_idx'] = range(len(df))

    merged = df.merge(schedules, on=['season','home_team','away_team'], how='inner')
    merged = merged.sort_values(['sbro_idx', 'week'])
    merged = merged.drop_duplicates(subset='sbro_idx', keep='first')
    merged = merged.drop_duplicates(subset='game_id', keep='first')

    matched, total = len(merged), len(df)
    print(f"Matched {matched}/{total} games ({matched/total*100:.1f}%)")

    if matched < total * 0.8:
        merged_left = df.merge(schedules, on=['season','home_team','away_team'], how='left')
        unmatched = merged_left[merged_left['game_id'].isna()]
        print(f"\nUnmatched sample:")
        print(unmatched[['season','home_team','away_team']].head(20).to_string())
        print(f"\nUnmatched by season:")
        print(unmatched.groupby('season').size())

    return merged


def save_line_movement(df: pd.DataFrame) -> None:
    with get_connection() as conn:
        df.to_sql('line_movement', conn, if_exists='replace', index=False)
    print(f"Saved {len(df)} rows to line_movement table")


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent
    print(f"Looking for nfl_YYYY.htm files in {data_dir}\n")

    df = load_all_sbro(str(data_dir))
    if df.empty:
        sys.exit(1)

    print(f"\nTotal games parsed: {len(df)}")
    print(f"Seasons: {sorted(df['season'].unique())}")

    df_test = df.copy()
    df_test['result']  = df_test['home_score'] - df_test['away_score']
    df_test['covered'] = (df_test['result'] > df_test['close_spread']).astype(int)
    print(f"Home cover rate: {df_test['covered'].mean():.3f} (expect ~0.50)")
    print(f"Bad spreads (>30): {(df['open_spread'].abs() > 30).sum()}")

    print(f"\nSample:")
    print(df[['season','home_team','away_team','open_spread','close_spread',
              'spread_move','open_total','close_total']].head(10).to_string())

    print("\nMatching to nflverse schedules...")
    matched = match_to_schedules(df)

    print("\nSaving to DB...")
    save_line_movement(matched)

    print(f"\nLine movement summary:")
    print(f"  Games with move >= 0.5pt: {matched['line_moved'].sum()} / {len(matched)}")
    print(f"  Sharp home signals:       {matched['sharp_home'].sum()}")
    print(f"  Sharp away signals:       {matched['sharp_away'].sum()}")
    print(f"\nSpread movement distribution:")
    print(matched['spread_move'].value_counts().sort_index().to_string())