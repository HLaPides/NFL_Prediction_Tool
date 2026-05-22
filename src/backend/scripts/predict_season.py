import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import math
import random
import pandas as pd
from collections import defaultdict
from data.sql_helpers import get_connection

# ── Config ────────────────────────────────────────────────────────────────────
N_SIMULATIONS      = 10_000
HOME_ADVANTAGE     = 48.0
SCHEDULE_SEASON    = 2025   # Use 2025 matchup structure for 2026 projection

# QB EPA adjustment — how much to shift ELO per unit of QB EPA above/below average
QB_EPA_ELO_SCALE   = 20.0

# Strength of schedule adjustment
SOS_ELO_SCALE      = 0.15

# ELO penalty per position when starter is confirmed out/doubtful
# Half penalty if backup is available, full penalty if backup also out
INJURY_ELO_IMPACT  = {
    "QB": 40.0,
    "RB": 10.0,
    "WR":  8.0,
    "TE":  6.0,
}

UNLIKELY_STATUSES  = {"Out", "Doubtful", "IR", "PUP", "Suspended"}

TEAM_DISPLAY_NAMES = {
    "Arizona Cardinals":    "ARI", "Atlanta Falcons":       "ATL", "Baltimore Ravens":    "BAL",
    "Buffalo Bills":        "BUF", "Carolina Panthers":     "CAR", "Chicago Bears":       "CHI",
    "Cincinnati Bengals":   "CIN", "Cleveland Browns":      "CLE", "Dallas Cowboys":      "DAL",
    "Denver Broncos":       "DEN", "Detroit Lions":         "DET", "Green Bay Packers":   "GB",
    "Houston Texans":       "HOU", "Indianapolis Colts":    "IND", "Jacksonville Jaguars":"JAX",
    "Kansas City Chiefs":   "KC",  "Los Angeles Rams":      "LA",  "Los Angeles Chargers":"LAC",
    "Las Vegas Raiders":    "LV",  "Miami Dolphins":        "MIA", "Minnesota Vikings":   "MIN",
    "New England Patriots": "NE",  "New Orleans Saints":    "NO",  "New York Giants":     "NYG",
    "New York Jets":        "NYJ", "Philadelphia Eagles":   "PHI", "Pittsburgh Steelers": "PIT",
    "Seattle Seahawks":     "SEA", "San Francisco 49ers":   "SF",  "Tampa Bay Buccaneers":"TB",
    "Tennessee Titans":     "TEN", "Washington Commanders": "WAS",
}

AFC_TEAMS = {"BAL","BUF","CIN","CLE","DEN","HOU","IND","JAX","KC","LAC","LV","MIA","NE","NYJ","PIT","TEN"}
NFC_TEAMS = {"ARI","ATL","CAR","CHI","DAL","DET","GB","LA","MIN","NO","NYG","PHI","SEA","SF","TB","WAS"}


# ── Core ELO helpers ──────────────────────────────────────────────────────────
def win_prob(home_elo: float, away_elo: float) -> float:
    adjusted = home_elo + HOME_ADVANTAGE
    return 1.0 / (1 + math.pow(10, (away_elo - adjusted) / 400))


def simulate_game(home_elo: float, away_elo: float, k: float = 20.0) -> tuple:
    prob_home    = win_prob(home_elo, away_elo)
    home_wins    = random.random() < prob_home
    elo_diff     = home_elo + HOME_ADVANTAGE - away_elo
    est_margin   = elo_diff / 25.0
    mov_mult     = min(math.log(abs(est_margin) + 1), 2.5)
    k_adj        = k * mov_mult
    outcome_home = 1.0 if home_wins else 0.0
    delta        = k_adj * (outcome_home - prob_home)
    return (
        "home" if home_wins else "away",
        round(home_elo + delta, 1),
        round(away_elo - delta, 1),
    )


def get_conference(team: str) -> str:
    return "AFC" if team in AFC_TEAMS else "NFC"


def seed_teams(standings: dict, conference: str) -> list:
    conf_teams = [(t, s) for t, s in standings.items() if get_conference(t) == conference]
    conf_teams.sort(key=lambda x: (x[1]["wins"], x[1]["elo"]), reverse=True)
    return [t for t, _ in conf_teams[:7]]


def simulate_playoffs(seeds_afc: list, seeds_nfc: list, elos: dict) -> str:
    def playoff_game(home: str, away: str) -> str:
        prob = win_prob(elos[home] + HOME_ADVANTAGE * 0.5, elos[away])
        return home if random.random() < prob else away

    def neutral_game(a: str, b: str) -> str:
        prob = win_prob(elos[a], elos[b])
        return a if random.random() < prob else b

    def run_conference(seeds: list) -> str:
        wc1 = playoff_game(seeds[1], seeds[6])
        wc2 = playoff_game(seeds[2], seeds[5])
        wc3 = playoff_game(seeds[3], seeds[4])
        remaining = sorted([wc1, wc2, wc3],
                           key=lambda t: seeds.index(t) if t in seeds else 9)
        div1 = playoff_game(seeds[0], remaining[-1])
        div2 = playoff_game(remaining[0], remaining[1])
        if seeds.index(div1) < seeds.index(div2):
            return playoff_game(div1, div2)
        return playoff_game(div2, div1)

    return neutral_game(run_conference(seeds_afc), run_conference(seeds_nfc))


# ── Adjustments ───────────────────────────────────────────────────────────────
def apply_qb_epa_adjustment(starting_elos: dict) -> dict:
    """Adjust ELO based on last season QB EPA relative to league average."""
    try:
        with get_connection() as conn:
            qb_epa = pd.read_sql("""
                SELECT team, AVG(passing_epa) AS avg_qb_epa
                FROM player_stats
                WHERE position = 'QB'
                  AND season_type = 'REG'
                  AND season = ?
                  AND attempts >= 10
                GROUP BY team
            """, conn, params=(SCHEDULE_SEASON,))

        if qb_epa.empty:
            print("  QB EPA: no data, skipping")
            return starting_elos

        league_avg = qb_epa["avg_qb_epa"].mean()
        adjusted   = dict(starting_elos)
        for _, row in qb_epa.iterrows():
            team = row["team"]
            if team in adjusted:
                adjusted[team] = round(
                    adjusted[team] + (row["avg_qb_epa"] - league_avg) * QB_EPA_ELO_SCALE, 1
                )
        print(f"  QB EPA applied (league avg EPA: {league_avg:.3f})")
        return adjusted

    except Exception as e:
        print(f"  QB EPA failed: {e}")
        return starting_elos


def apply_sos_adjustment(starting_elos: dict, schedule: pd.DataFrame) -> dict:
    """Adjust ELO based on average opponent strength in the schedule."""
    if schedule.empty:
        return starting_elos

    opp_elos = defaultdict(list)
    for _, game in schedule.iterrows():
        home, away = game["home_team"], game["away_team"]
        if home in starting_elos and away in starting_elos:
            opp_elos[home].append(starting_elos[away])
            opp_elos[away].append(starting_elos[home])

    if not opp_elos:
        return starting_elos

    avg_opp        = {t: sum(o) / len(o) for t, o in opp_elos.items()}
    league_avg_opp = sum(avg_opp.values()) / len(avg_opp)
    adjusted       = dict(starting_elos)
    for team, avg in avg_opp.items():
        adjusted[team] = round(adjusted[team] + (avg - league_avg_opp) * SOS_ELO_SCALE, 1)

    print("  SOS adjustment applied")
    return adjusted


def build_injury_penalties() -> dict:
    """
    Cross-reference depth_charts_current with injuries_current.
    Returns { team_abbr: total_elo_penalty }.
    Starter out + backup available = 50% penalty.
    Starter out + no backup        = 100% penalty.
    Starter questionable           = 25% penalty.
    """
    try:
        with get_connection() as conn:
            depth = pd.read_sql("""
                SELECT team, position, depth_order, player_name
                FROM depth_charts_current
                WHERE position IN ('QB','RB','WR','TE')
                ORDER BY team, position, depth_order
            """, conn)

            inj = pd.read_sql("""
                SELECT team_display, player_name, position, status
                FROM injuries_current
            """, conn)

    except Exception as e:
        print(f"  Injuries: could not load data ({e}), skipping")
        return {}

    if depth.empty:
        print("  Injuries: no depth chart data, skipping")
        return {}

    inj["team"] = inj["team_display"].map(TEAM_DISPLAY_NAMES)
    inj = inj[inj["team"].notna()].copy()

    out_names          = set(inj[inj["status"].isin(UNLIKELY_STATUSES)]["player_name"].str.lower())
    questionable_names = set(inj[inj["status"] == "Questionable"]["player_name"].str.lower())

    penalties = defaultdict(float)
    reported  = []

    for team in depth["team"].unique():
        team_depth = depth[depth["team"] == team]

        for position, base_penalty in INJURY_ELO_IMPACT.items():
            pos_players = team_depth[team_depth["position"] == position].sort_values("depth_order")
            if pos_players.empty:
                continue

            starter       = pos_players.iloc[0]["player_name"]
            starter_lower = starter.lower() if starter else ""
            backup        = pos_players.iloc[1]["player_name"] if len(pos_players) > 1 else None
            backup_lower  = backup.lower() if backup else ""

            if starter_lower in out_names:
                if backup_lower and backup_lower not in out_names:
                    penalty = base_penalty * 0.5
                else:
                    penalty = base_penalty
                penalties[team] += penalty
                reported.append(f"    {team} {position}1 OUT ({starter}) -{penalty:.0f} ELO")

            elif starter_lower in questionable_names:
                penalty = base_penalty * 0.25
                penalties[team] += penalty
                reported.append(f"    {team} {position}1 QUESTIONABLE ({starter}) -{penalty:.0f} ELO")

    if reported:
        print(f"  Injury penalties ({len(reported)} starters affected):")
        for r in reported:
            print(r)
    else:
        print("  No significant starter injuries found")

    return dict(penalties)


def apply_injury_penalties(starting_elos: dict, penalties: dict) -> dict:
    adjusted = dict(starting_elos)
    for team, penalty in penalties.items():
        if team in adjusted:
            adjusted[team] = round(adjusted[team] - penalty, 1)
    return adjusted


# ── Main simulation ───────────────────────────────────────────────────────────
def simulate_season(schedule: pd.DataFrame, starting_elos: dict) -> str:
    elos      = dict(starting_elos)
    standings = {team: {"wins": 0, "losses": 0, "elo": elos.get(team, 1500)}
                 for team in elos}

    for _, game in schedule.sort_values("week").iterrows():
        home = game["home_team"]
        away = game["away_team"]
        if home not in elos or away not in elos:
            continue

        result, new_home, new_away = simulate_game(elos[home], elos[away])
        elos[home] = new_home
        elos[away] = new_away

        if result == "home":
            standings[home]["wins"]   += 1
            standings[away]["losses"] += 1
        else:
            standings[away]["wins"]   += 1
            standings[home]["losses"] += 1

        standings[home]["elo"] = new_home
        standings[away]["elo"] = new_away

    return simulate_playoffs(
        seed_teams(standings, "AFC"),
        seed_teams(standings, "NFC"),
        elos,
    )


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Super Bowl LXI simulator")
    print(f"Schedule: {SCHEDULE_SEASON} structure  |  ELO: current ratings")
    print(f"Simulations: {N_SIMULATIONS:,}\n")

    print("Loading schedule from DB...")
    with get_connection() as conn:
        reg_only = pd.read_sql("""
            SELECT game_id, season, week, home_team, away_team, game_type
            FROM schedules
            WHERE season = ? AND game_type = 'REG'
            ORDER BY week
        """, conn, params=(SCHEDULE_SEASON,))
    print(f"  {len(reg_only)} regular season games")

    print("\nLoading ELO ratings...")
    with get_connection() as conn:
        elo_df = pd.read_sql("""
            SELECT h.team, h.elo_after AS elo
            FROM elo_history h
            INNER JOIN (
                SELECT team, MAX(season * 100 + week) AS max_key
                FROM elo_history GROUP BY team
            ) latest ON h.team = latest.team
              AND (h.season * 100 + h.week) = latest.max_key
            WHERE h.team NOT IN ('OAK', 'SD', 'STL')
        """, conn)
    starting_elos = dict(zip(elo_df["team"], elo_df["elo"]))
    print(f"  {len(starting_elos)} teams loaded")

    print("\nApplying preseason adjustments...")
    starting_elos = apply_qb_epa_adjustment(starting_elos)
    starting_elos = apply_sos_adjustment(starting_elos, reg_only)
    penalties     = build_injury_penalties()
    starting_elos = apply_injury_penalties(starting_elos, penalties)

    print(f"\nFinal starting ELO (all teams):")
    for team, elo in sorted(starting_elos.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int((elo - 1400) / 10)
        print(f"  {team:<5} {elo:>6.0f}  {bar}")

    print(f"\nRunning {N_SIMULATIONS:,} simulations...")
    sb_wins   = defaultdict(int)
    conf_wins = defaultdict(int)

    for i in range(N_SIMULATIONS):
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1:,} / {N_SIMULATIONS:,} complete...")
        winner = simulate_season(reg_only, starting_elos)
        sb_wins[winner] += 1
        conf_wins[get_conference(winner)] += 1

    # Ensure all teams appear even if 0 wins
    for team in starting_elos:
        if team not in sb_wins:
            sb_wins[team] = 0

    print(f"\n{'='*55}")
    print(f"SUPER BOWL LXI PREDICTIONS")
    print(f"Based on {N_SIMULATIONS:,} simulations")
    print(f"{'='*55}")
    print(f"{'Rank':<5} {'Team':<6} {'SB Win%':>8} {'Implied Odds':>14}")
    print(f"{'-'*35}")

    sorted_teams = sorted(sb_wins.items(), key=lambda x: x[1], reverse=True)
    for rank, (team, wins) in enumerate(sorted_teams, 1):
        pct = wins / N_SIMULATIONS * 100
        if pct >= 50:
            odds = f"-{int((pct / (100 - pct)) * 100)}"
        elif pct > 0:
            odds = f"+{int(((100 - pct) / pct) * 100)}"
        else:
            odds = "N/A"
        print(f"{rank:<5} {team:<6} {pct:>7.1f}%  {odds:>13}")

    print(f"\nConference breakdown:")
    for conf in ["AFC", "NFC"]:
        pct = conf_wins[conf] / N_SIMULATIONS * 100
        print(f"  {conf}: {pct:.1f}%")