import sys
sys.path.insert(0, '.')
from backend.data.sql_helpers import get_connection
import pandas as pd

with get_connection() as conn:
    df = pd.read_sql("""
        SELECT season, COUNT(DISTINCT player_id) as players,
               COUNT(*) as rows
        FROM player_stats
        WHERE position = 'QB'
        GROUP BY season
        ORDER BY season
    """, conn)
print(df.to_string())