import sys
sys.path.insert(0, '.')
from data.sql_helpers import get_connection
import pandas as pd

# Spreads based on actual 2025 season strength:
# Elite: NE, DEN, PHI, SEA, JAX, HOU, PIT
# Good:  BUF, LAC, MIN, GB, TB, ATL, SF
# Mid:   CIN, BAL, DAL, WAS, LA, CHI, NYJ
# Weak:  KC, LV, MIA, CLE, IND, TEN, CAR, NYG, ARI, NO

mock_games = [
    # Thursday night
    {"game_id":"2025_TEST_NE_KC",   "season":2025,"week":1,"gameday":"2025-09-04","gametime":"8:20PM", "home_team":"NE",  "away_team":"KC",  "model_spread":  8.1, "vegas_spread": 7.5, "spread_gap": 0.6, "home_win_prob":0.73,"away_win_prob":0.27,"predicted_winner":"NE",  "generated_at":"2025-09-03T12:00:00"},
    # Sunday 1pm
    {"game_id":"2025_TEST_PHI_DAL", "season":2025,"week":1,"gameday":"2025-09-07","gametime":"1:00PM", "home_team":"PHI", "away_team":"DAL", "model_spread":  4.2, "vegas_spread": 4.0, "spread_gap": 0.2, "home_win_prob":0.61,"away_win_prob":0.39,"predicted_winner":"PHI", "generated_at":"2025-09-03T12:00:00"},
    {"game_id":"2025_TEST_PIT_CLE", "season":2025,"week":1,"gameday":"2025-09-07","gametime":"1:00PM", "home_team":"PIT", "away_team":"CLE", "model_spread":  6.3, "vegas_spread": 5.5, "spread_gap": 0.8, "home_win_prob":0.66,"away_win_prob":0.34,"predicted_winner":"PIT", "generated_at":"2025-09-03T12:00:00"},
    {"game_id":"2025_TEST_JAX_TEN", "season":2025,"week":1,"gameday":"2025-09-07","gametime":"1:00PM", "home_team":"JAX", "away_team":"TEN", "model_spread":  9.4, "vegas_spread": 8.5, "spread_gap": 0.9, "home_win_prob":0.75,"away_win_prob":0.25,"predicted_winner":"JAX", "generated_at":"2025-09-03T12:00:00"},
    {"game_id":"2025_TEST_HOU_IND", "season":2025,"week":1,"gameday":"2025-09-07","gametime":"1:00PM", "home_team":"HOU", "away_team":"IND", "model_spread":  7.2, "vegas_spread": 6.5, "spread_gap": 0.7, "home_win_prob":0.70,"away_win_prob":0.30,"predicted_winner":"HOU", "generated_at":"2025-09-03T12:00:00"},
    {"game_id":"2025_TEST_MIN_CHI", "season":2025,"week":1,"gameday":"2025-09-07","gametime":"1:00PM", "home_team":"MIN", "away_team":"CHI", "model_spread": -1.8, "vegas_spread":-2.5, "spread_gap": 0.7, "home_win_prob":0.45,"away_win_prob":0.55,"predicted_winner":"CHI", "generated_at":"2025-09-03T12:00:00"},
    {"game_id":"2025_TEST_BUF_MIA", "season":2025,"week":1,"gameday":"2025-09-07","gametime":"1:00PM", "home_team":"BUF", "away_team":"MIA", "model_spread":  5.8, "vegas_spread": 5.5, "spread_gap": 0.3, "home_win_prob":0.65,"away_win_prob":0.35,"predicted_winner":"BUF", "generated_at":"2025-09-03T12:00:00"},
    {"game_id":"2025_TEST_ATL_NO",  "season":2025,"week":1,"gameday":"2025-09-07","gametime":"1:00PM", "home_team":"ATL", "away_team":"NO",  "model_spread":  3.1, "vegas_spread": 3.0, "spread_gap": 0.1, "home_win_prob":0.58,"away_win_prob":0.42,"predicted_winner":"ATL", "generated_at":"2025-09-03T12:00:00"},
    {"game_id":"2025_TEST_GB_NYG",  "season":2025,"week":1,"gameday":"2025-09-07","gametime":"1:00PM", "home_team":"GB",  "away_team":"NYG", "model_spread":  7.9, "vegas_spread": 6.0, "spread_gap": 1.9, "home_win_prob":0.72,"away_win_prob":0.28,"predicted_winner":"GB",  "generated_at":"2025-09-03T12:00:00"},
    # Sunday 4pm
    {"game_id":"2025_TEST_DEN_LV",  "season":2025,"week":1,"gameday":"2025-09-07","gametime":"4:05PM", "home_team":"DEN", "away_team":"LV",  "model_spread": 10.2, "vegas_spread": 9.5, "spread_gap": 0.7, "home_win_prob":0.77,"away_win_prob":0.23,"predicted_winner":"DEN", "generated_at":"2025-09-03T12:00:00"},
    {"game_id":"2025_TEST_LAC_ARI", "season":2025,"week":1,"gameday":"2025-09-07","gametime":"4:05PM", "home_team":"LAC", "away_team":"ARI", "model_spread":  4.8, "vegas_spread": 4.5, "spread_gap": 0.3, "home_win_prob":0.62,"away_win_prob":0.38,"predicted_winner":"LAC", "generated_at":"2025-09-03T12:00:00"},
    {"game_id":"2025_TEST_SF_CAR",  "season":2025,"week":1,"gameday":"2025-09-07","gametime":"4:25PM", "home_team":"SF",  "away_team":"CAR", "model_spread":  6.1, "vegas_spread": 7.0, "spread_gap":-0.9, "home_win_prob":0.66,"away_win_prob":0.34,"predicted_winner":"SF",  "generated_at":"2025-09-03T12:00:00"},
    {"game_id":"2025_TEST_SEA_LA",  "season":2025,"week":1,"gameday":"2025-09-07","gametime":"4:25PM", "home_team":"SEA", "away_team":"LA",  "model_spread":  5.5, "vegas_spread": 3.0, "spread_gap": 2.5, "home_win_prob":0.64,"away_win_prob":0.36,"predicted_winner":"SEA", "generated_at":"2025-09-03T12:00:00"},
    {"game_id":"2025_TEST_TB_WAS",  "season":2025,"week":1,"gameday":"2025-09-07","gametime":"4:25PM", "home_team":"TB",  "away_team":"WAS", "model_spread":  1.2, "vegas_spread": 1.5, "spread_gap":-0.3, "home_win_prob":0.53,"away_win_prob":0.47,"predicted_winner":"TB",  "generated_at":"2025-09-03T12:00:00"},
    # Sunday night
    {"game_id":"2025_TEST_BAL_NYJ", "season":2025,"week":1,"gameday":"2025-09-07","gametime":"8:20PM", "home_team":"BAL", "away_team":"NYJ", "model_spread":  2.1, "vegas_spread": 5.5, "spread_gap":-3.4, "home_win_prob":0.55,"away_win_prob":0.45,"predicted_winner":"BAL", "generated_at":"2025-09-03T12:00:00"},
    # Monday night
    {"game_id":"2025_TEST_CIN_NYJ2","season":2025,"week":1,"gameday":"2025-09-08","gametime":"8:15PM", "home_team":"CIN", "away_team":"GB",  "model_spread": -1.4, "vegas_spread":-1.0, "spread_gap":-0.4, "home_win_prob":0.46,"away_win_prob":0.54,"predicted_winner":"GB",  "generated_at":"2025-09-03T12:00:00"},
]

df = pd.DataFrame(mock_games)

with get_connection() as conn:
    df.to_sql("weekly_predictions", conn, if_exists="replace", index=False)
    print(f"Inserted {len(df)} mock games into weekly_predictions")
    print("Open the app and check the PICKS tab")

# To remove:
#with get_connection() as conn:
#     conn.execute("DELETE FROM weekly_predictions WHERE game_id LIKE '2025_TEST_%'")
#     conn.commit()