from backend.data.sql_helpers import get_connection
import pandas as pd

with get_connection() as conn:
    df = pd.read_sql("""
        SELECT passing_interceptions, attempts,
               rushing_fumbles, receiving_fumbles,
               passing_first_downs, rushing_first_downs,
               receiving_first_downs, penalties, penalty_yards
        FROM team_stats_weekly
        WHERE season = 2024 AND week = 1
        LIMIT 3
    """, conn)
print(df.to_string())