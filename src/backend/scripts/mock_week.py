import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from data.sql_helpers import get_connection

# ── Mock QB props ─────────────────────────────────────────────────────────────
mock_qbs = [
    {"player_name":"P.Mahomes",  "team":"KC",  "opponent":"BUF","week":15,"is_home":1,"pred_passing_yards":287.3,"pred_passing_tds":2.1,"pred_attempts":34.2,"implied_team_total":27.5,"spread_line":-3.5},
    {"player_name":"J.Allen",    "team":"BUF", "opponent":"KC", "week":15,"is_home":0,"pred_passing_yards":264.8,"pred_passing_tds":1.9,"pred_attempts":36.1,"implied_team_total":24.5,"spread_line": 3.5},
    {"player_name":"L.Jackson",  "team":"BAL", "opponent":"PIT","week":15,"is_home":1,"pred_passing_yards":231.4,"pred_passing_tds":1.7,"pred_attempts":28.6,"implied_team_total":26.0,"spread_line":-7.0},
    {"player_name":"J.Burrow",   "team":"CIN", "opponent":"CLE","week":15,"is_home":1,"pred_passing_yards":278.9,"pred_passing_tds":2.2,"pred_attempts":37.4,"implied_team_total":25.5,"spread_line":-5.5},
    {"player_name":"J.Goff",     "team":"DET", "opponent":"MIN","week":15,"is_home":0,"pred_passing_yards":252.1,"pred_passing_tds":1.8,"pred_attempts":33.8,"implied_team_total":23.0,"spread_line": 2.5},
    {"player_name":"J.Hurts",    "team":"PHI", "opponent":"DAL","week":15,"is_home":1,"pred_passing_yards":219.6,"pred_passing_tds":1.6,"pred_attempts":29.2,"implied_team_total":27.0,"spread_line":-6.0},
    {"player_name":"B.Purdy",    "team":"SF",  "opponent":"LAR","week":15,"is_home":1,"pred_passing_yards":244.3,"pred_passing_tds":1.9,"pred_attempts":31.5,"implied_team_total":24.5,"spread_line":-4.5},
    {"player_name":"C.Stroud",   "team":"HOU", "opponent":"TEN","week":15,"is_home":1,"pred_passing_yards":261.7,"pred_passing_tds":1.8,"pred_attempts":35.0,"implied_team_total":25.0,"spread_line":-8.5},
    {"player_name":"D.Prescott", "team":"DAL", "opponent":"PHI","week":15,"is_home":0,"pred_passing_yards":247.2,"pred_passing_tds":1.7,"pred_attempts":34.6,"implied_team_total":21.0,"spread_line": 6.0},
    {"player_name":"J.Love",     "team":"GB",  "opponent":"SEA","week":15,"is_home":1,"pred_passing_yards":256.4,"pred_passing_tds":1.9,"pred_attempts":33.1,"implied_team_total":24.0,"spread_line":-3.0},
]

# ── Mock RB props ─────────────────────────────────────────────────────────────
mock_rbs = [
    {"player_name":"D.Henry",    "team":"TEN", "opponent":"HOU","week":15,"is_home":0,"pred_rushing_yards":88.4, "pred_rushing_tds":0.7,"pred_carries":21.2,"pred_receptions":2.1,"pred_receiving_yards":14.3,"implied_team_total":20.5,"spread_line": 8.5},
    {"player_name":"J.Taylor",   "team":"IND", "opponent":"JAX","week":15,"is_home":1,"pred_rushing_yards":76.2, "pred_rushing_tds":0.6,"pred_carries":18.4,"pred_receptions":3.2,"pred_receiving_yards":22.1,"implied_team_total":22.0,"spread_line":-1.5},
    {"player_name":"C.McCaffrey","team":"SF",  "opponent":"LAR","week":15,"is_home":1,"pred_rushing_yards":64.8, "pred_rushing_tds":0.5,"pred_carries":14.1,"pred_receptions":6.8,"pred_receiving_yards":52.4,"implied_team_total":24.5,"spread_line":-4.5},
    {"player_name":"B.Robinson", "team":"ATL", "opponent":"NO", "week":15,"is_home":1,"pred_rushing_yards":82.1, "pred_rushing_tds":0.6,"pred_carries":19.7,"pred_receptions":2.4,"pred_receiving_yards":16.8,"implied_team_total":21.5,"spread_line":-3.0},
    {"player_name":"K.Walker",   "team":"SEA", "opponent":"GB", "week":15,"is_home":0,"pred_rushing_yards":71.3, "pred_rushing_tds":0.5,"pred_carries":17.2,"pred_receptions":2.8,"pred_receiving_yards":19.2,"implied_team_total":21.5,"spread_line": 3.0},
    {"player_name":"T.Pollard",  "team":"TEN", "opponent":"HOU","week":15,"is_home":0,"pred_rushing_yards":48.6, "pred_rushing_tds":0.3,"pred_carries":12.1,"pred_receptions":3.5,"pred_receiving_yards":26.4,"implied_team_total":20.5,"spread_line": 8.5},
    {"player_name":"J.Mixon",    "team":"HOU", "opponent":"TEN","week":15,"is_home":1,"pred_rushing_yards":79.4, "pred_rushing_tds":0.7,"pred_carries":18.6,"pred_receptions":2.9,"pred_receiving_yards":20.1,"implied_team_total":25.0,"spread_line":-8.5},
    {"player_name":"A.Jones",    "team":"GB",  "opponent":"SEA","week":15,"is_home":1,"pred_rushing_yards":66.7, "pred_rushing_tds":0.5,"pred_carries":15.8,"pred_receptions":3.1,"pred_receiving_yards":21.3,"implied_team_total":24.0,"spread_line":-3.0},
]

# ── Mock WR props ─────────────────────────────────────────────────────────────
mock_wrs = [
    {"player_name":"T.Hill",     "team":"MIA", "opponent":"NE", "week":15,"is_home":1,"pred_receiving_yards":98.4, "pred_receiving_tds":0.6,"pred_receptions":6.8,"pred_targets":9.2,"implied_team_total":24.0,"spread_line":-3.5},
    {"player_name":"S.Diggs",    "team":"BUF", "opponent":"KC", "week":15,"is_home":0,"pred_receiving_yards":76.2, "pred_receiving_tds":0.5,"pred_receptions":5.4,"pred_targets":7.8,"implied_team_total":24.5,"spread_line": 3.5},
    {"player_name":"J.Chase",    "team":"CIN", "opponent":"CLE","week":15,"is_home":1,"pred_receiving_yards":84.7, "pred_receiving_tds":0.7,"pred_receptions":5.9,"pred_targets":8.4,"implied_team_total":25.5,"spread_line":-5.5},
    {"player_name":"D.Adams",    "team":"LV",  "opponent":"LAC","week":15,"is_home":0,"pred_receiving_yards":71.3, "pred_receiving_tds":0.4,"pred_receptions":5.2,"pred_targets":7.6,"implied_team_total":19.5,"spread_line": 4.5},
    {"player_name":"C.Lamb",     "team":"DAL", "opponent":"PHI","week":15,"is_home":0,"pred_receiving_yards":89.1, "pred_receiving_tds":0.6,"pred_receptions":6.4,"pred_targets":9.0,"implied_team_total":21.0,"spread_line": 6.0},
    {"player_name":"S.Diggs",    "team":"HOU", "opponent":"TEN","week":15,"is_home":1,"pred_receiving_yards":68.4, "pred_receiving_tds":0.4,"pred_receptions":4.9,"pred_targets":7.1,"implied_team_total":25.0,"spread_line":-8.5},
    {"player_name":"A.Cooper",   "team":"CLE", "opponent":"CIN","week":15,"is_home":0,"pred_receiving_yards":62.8, "pred_receiving_tds":0.3,"pred_receptions":4.6,"pred_targets":6.8,"implied_team_total":18.5,"spread_line": 5.5},
    {"player_name":"D.Samuel",   "team":"SF",  "opponent":"LAR","week":15,"is_home":1,"pred_receiving_yards":74.6, "pred_receiving_tds":0.5,"pred_receptions":5.6,"pred_targets":7.4,"implied_team_total":24.5,"spread_line":-4.5},
    {"player_name":"J.Jefferson","team":"MIN", "opponent":"DET","week":15,"is_home":0,"pred_receiving_yards":91.2, "pred_receiving_tds":0.6,"pred_receptions":6.2,"pred_targets":8.8,"implied_team_total":22.5,"spread_line": 2.5},
    {"player_name":"T.McLaurin", "team":"WAS", "opponent":"NYG","week":15,"is_home":1,"pred_receiving_yards":67.3, "pred_receiving_tds":0.4,"pred_receptions":4.8,"pred_targets":7.0,"implied_team_total":22.0,"spread_line":-4.0},
]

def insert_mock_data():
    qb_df = pd.DataFrame(mock_qbs)
    rb_df = pd.DataFrame(mock_rbs)
    wr_df = pd.DataFrame(mock_wrs)

    qb_df["game_id"] = qb_df.apply(lambda r: f"2025_15_{r['team']}_{r['opponent']}", axis=1)
    rb_df["game_id"] = rb_df.apply(lambda r: f"2025_15_{r['team']}_{r['opponent']}", axis=1)
    wr_df["game_id"] = wr_df.apply(lambda r: f"2025_15_{r['team']}_{r['opponent']}", axis=1)

    with get_connection() as conn:
        qb_df.to_sql("qb_prop_predictions", conn, if_exists="replace", index=False)
        rb_df.to_sql("rb_prop_predictions", conn, if_exists="replace", index=False)
        wr_df.to_sql("wr_prop_predictions", conn, if_exists="replace", index=False)

    print(f"Inserted {len(qb_df)} QB, {len(rb_df)} RB, {len(wr_df)} WR mock predictions")

if __name__ == "__main__":
    insert_mock_data()