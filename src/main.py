from backend.data.sql_helpers import get_connection
import pandas as pd

with get_connection() as conn:
    df = pd.read_sql("""
        SELECT mp.model_spread, mp.actual_winner, mp.vegas_winner,
               mp.home_team, mp.away_team, mp.result,
               o.home_moneyline, o.away_moneyline
        FROM model_predictions mp
        JOIN game_odds o ON o.game_id = mp.game_id
        WHERE o.home_moneyline IS NOT NULL
    """, conn)

# Flag games where model disagrees with Vegas on winner
df["model_winner"] = df.apply(lambda r: r["home_team"] if r["model_spread"] > 0 else r["away_team"], axis=1)
df["disagrees"]    = df["model_winner"] != df["vegas_winner"]
df["correct"]      = df["model_winner"] == df["actual_winner"]

print(f"Games model disagrees with Vegas: {df['disagrees'].sum()}")
print(f"Model correct on those: {df[df['disagrees']]['correct'].mean()*100:.1f}%")
print(f"\nModel agrees with Vegas:")
print(f"Model correct: {df[~df['disagrees']]['correct'].mean()*100:.1f}%")