import sys
sys.path.insert(0, '.')
from backend.data.sql_helpers import get_connection
import pandas as pd

with get_connection() as conn:
    rb = pd.read_sql("""
        SELECT season, COUNT(DISTINCT player_id) as players, COUNT(*) as rows
        FROM player_stats
        WHERE position = 'RB' AND season_type = 'REG' AND carries >= 5
        GROUP BY season ORDER BY season
    """, conn)
    wr = pd.read_sql("""
        SELECT season, COUNT(DISTINCT player_id) as players, COUNT(*) as rows
        FROM player_stats
        WHERE position = 'WR' AND season_type = 'REG' AND targets >= 3
        GROUP BY season ORDER BY season
    """, conn)

print("RB:")
print(rb.to_string())
print("\nWR:")
print(wr.to_string())