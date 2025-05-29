from datetime import datetime, timezone
import pandas as pd
from meteostat import Point, Hourly
import sqlite3, pathlib

POINT = Point(53.06, 6.66)  # Zeegse
DB = pathlib.Path(__file__).parent.parent / "data" / "weather.sqlite"

def fetch_and_store():
    # 1. figure out last timestamp we already have
    conn = sqlite3.connect(DB)
    try:
        last = pd.read_sql("SELECT MAX(time) AS t FROM hourly", conn)["t"][0]
        # Make start offset-aware (UTC)
        start = pd.to_datetime(last, utc=True) + pd.Timedelta(hours=1)
    except Exception:
        # Make start offset-aware (UTC)
        start = datetime(1970, 1, 1, tzinfo=timezone.utc)
    end = datetime.now(timezone.utc)

    # 2. grab data
    if start >= end:
        return
    df = Hourly(POINT, start, end).fetch()

    # 3. persist
    # Ensure the DataFrame index is timezone-aware (UTC) before saving if it's not already
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')
        
    df.to_sql("hourly", conn, if_exists="append", index_label="time")
    conn.close()

if __name__ == "__main__":
    fetch_and_store() 