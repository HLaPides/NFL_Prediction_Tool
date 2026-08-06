"""
odds_getter.py — Fetch live NFL spread lines from The Odds API.

Supplements nflverse schedule data with current spread lines for upcoming
games. Called during weekly_update.py after the nflverse fetch to overwrite
stale spread lines with the most current consensus line.

The Odds API returns all upcoming NFL games in a single request, so this
costs exactly 1 API call per weekly update — well within the 500/month
free tier even across multiple users.

Team name mapping:
    The Odds API uses full names ("Kansas City Chiefs") while the rest of
    the pipeline uses abbreviations ("KC"). TEAM_NAME_MAP handles the
    conversion. Washington is listed as "Washington Commanders" in The Odds
    API but nflverse uses "WAS" — both are handled.

Usage:
    From weekly_update.py:
        from data.odds_getter import fetch_live_spreads, update_odds_in_db

    Standalone (prints current lines):
        python3.11 data/odds_getter.py

Requires:
    ODDS_API_KEY in src/backend/.env
    pip install python-dotenv
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import os
import requests
import pandas as pd
from dotenv import load_dotenv
from data.sql_helpers import get_connection

# Load .env from backend root
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"

# Preferred book order — use first available per game
BOOK_PRIORITY = ["draftkings", "fanduel", "betmgm", "bovada", "williamhill_us"]

# The Odds API full name → nflverse abbreviation
TEAM_NAME_MAP = {
    "Arizona Cardinals":     "ARI",
    "Atlanta Falcons":       "ATL",
    "Baltimore Ravens":      "BAL",
    "Buffalo Bills":         "BUF",
    "Carolina Panthers":     "CAR",
    "Chicago Bears":         "CHI",
    "Cincinnati Bengals":    "CIN",
    "Cleveland Browns":      "CLE",
    "Dallas Cowboys":        "DAL",
    "Denver Broncos":        "DEN",
    "Detroit Lions":         "DET",
    "Green Bay Packers":     "GB",
    "Houston Texans":        "HOU",
    "Indianapolis Colts":    "IND",
    "Jacksonville Jaguars":  "JAX",
    "Kansas City Chiefs":    "KC",
    "Los Angeles Rams":      "LA",
    "Los Angeles Chargers":  "LAC",
    "Las Vegas Raiders":     "LV",
    "Miami Dolphins":        "MIA",
    "Minnesota Vikings":     "MIN",
    "New England Patriots":  "NE",
    "New Orleans Saints":    "NO",
    "New York Giants":       "NYG",
    "New York Jets":         "NYJ",
    "Philadelphia Eagles":   "PHI",
    "Pittsburgh Steelers":   "PIT",
    "Seattle Seahawks":      "SEA",
    "San Francisco 49ers":   "SF",
    "Tampa Bay Buccaneers":  "TB",
    "Tennessee Titans":      "TEN",
    "Washington Commanders": "WAS",
}


def fetch_live_spreads() -> pd.DataFrame:
    """
    Fetch current NFL spread lines from The Odds API.

    Hits the spreads market for all upcoming NFL games. Uses the first
    available book from BOOK_PRIORITY per game — DraftKings preferred,
    falls back down the list if not available. Spread is always expressed
    from the home team's perspective (negative = home team favored),
    matching the nflverse convention.

    Returns:
        pd.DataFrame: Columns: home_team, away_team, spread_line, book, fetched_at.
                      One row per game. Empty DataFrame if API call fails
                      or no upcoming games found.

    Raises:
        EnvironmentError: If ODDS_API_KEY is not set in .env.
    """
    if not ODDS_API_KEY:
        raise EnvironmentError(
            "ODDS_API_KEY not set. Add it to src/backend/.env"
        )

    try:
        resp = requests.get(ODDS_API_URL, params={
            "apiKey":     ODDS_API_KEY,
            "regions":    "us",
            "markets":    "spreads",
            "oddsFormat": "american",
        }, timeout=10)
        resp.raise_for_status()
    except requests.HTTPError as e:
        print(f"  Odds API HTTP error: {e}")
        return pd.DataFrame()
    except requests.RequestException as e:
        print(f"  Odds API connection error: {e}")
        return pd.DataFrame()

    games    = resp.json()
    requests_used      = resp.headers.get("x-requests-used", "?")
    requests_remaining = resp.headers.get("x-requests-remaining", "?")
    print(f"  Odds API: {requests_used} used, {requests_remaining} remaining this month")

    rows = []
    for game in games:
        home_full = game.get("home_team", "")
        away_full = game.get("away_team", "")
        home_abbr = TEAM_NAME_MAP.get(home_full)
        away_abbr = TEAM_NAME_MAP.get(away_full)

        if not home_abbr or not away_abbr:
            print(f"  Unknown team name: '{home_full}' or '{away_full}' — skipping")
            continue

        # Find best available bookmaker
        bookmakers = {b["key"]: b for b in game.get("bookmakers", [])}
        book       = next((b for b in BOOK_PRIORITY if b in bookmakers), None)
        if not book:
            # Fall back to any available book
            book = next(iter(bookmakers), None)
        if not book:
            continue

        markets = {m["key"]: m for m in bookmakers[book].get("markets", [])}
        if "spreads" not in markets:
            continue

        outcomes     = markets["spreads"]["outcomes"]
        home_outcome = next((o for o in outcomes if o["name"] == home_full), None)
        if not home_outcome:
            continue

        rows.append({
            "home_team":   home_abbr,
            "away_team":   away_abbr,
            "spread_line": float(home_outcome["point"]),
            "book":        book,
            "fetched_at":  pd.Timestamp.now().isoformat(),
        })

    df = pd.DataFrame(rows)
    print(f"  Fetched {len(df)} live spread lines")
    return df


def update_odds_in_db(live_odds: pd.DataFrame) -> int:
    """
    Overwrite spread_line in game_odds for upcoming games with live lines.

    Matches games by home_team + away_team. Only updates games that exist
    in schedules_current with no score yet (i.e. upcoming games). Already
    completed games are never touched.

    Args:
        live_odds: Output of fetch_live_spreads().

    Returns:
        int: Number of game_odds rows updated.
    """
    if live_odds.empty:
        return 0

    with get_connection() as conn:
        upcoming = pd.read_sql("""
            SELECT game_id, home_team, away_team
            FROM schedules_current
            WHERE home_score IS NULL
        """, conn)

    if upcoming.empty:
        return 0

    # Join live odds to upcoming game_ids by home + away team
    merged = upcoming.merge(live_odds, on=["home_team", "away_team"], how="inner")
    if merged.empty:
        print("  No live odds matched upcoming games")
        return 0

    updated = 0
    with get_connection() as conn:
        for _, row in merged.iterrows():
            conn.exec_driver_sql("""
                UPDATE game_odds
                SET spread_line = %s
                WHERE game_id = %s
            """, (row["spread_line"], row["game_id"]))
            updated += 1
        conn.commit()

    print(f"  Updated {updated} spread lines with live odds")
    return updated


def get_live_spreads_for_display() -> pd.DataFrame:
    """
    Fetch live odds and return as a display-ready DataFrame.
    Used by api.py if you want a /odds endpoint for the frontend.

    Returns:
        pd.DataFrame with home_team, away_team, spread_line, book, fetched_at.
    """
    return fetch_live_spreads()


if __name__ == "__main__":
    print("Fetching live NFL spreads...")
    df = fetch_live_spreads()

    if df.empty:
        print("No live odds available.")
    else:
        print(f"\n{'Home':<6} {'Away':<6} {'Spread':>8} {'Book'}")
        print("-" * 35)
        for _, row in df.iterrows():
            spread = f"{row['spread_line']:+.1f}"
            print(f"{row['home_team']:<6} {row['away_team']:<6} {spread:>8}  {row['book']}")