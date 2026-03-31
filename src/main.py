from backend.data.sql_helpers import get_connection
import pandas as pd

with get_connection() as conn:
    df = pd.read_sql("""
        SELECT h.season, h.week, h.team, h.opponent, h.elo_before, h.elo_after
        FROM elo_history h
        WHERE h.team = 'KC' AND h.season = 2024
        ORDER BY h.week
    """, conn)
print(df.head(10).to_string())