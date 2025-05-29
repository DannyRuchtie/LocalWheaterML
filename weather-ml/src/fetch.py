from datetime import datetime, timezone
import pandas as pd
from meteostat import Point, Hourly
import sqlite3, pathlib

POINT = Point(53.06, 6.66)  # Zeegse
DB = pathlib.Path(__file__).parent.parent / "weather.sqlite"

def fetch_and_store():
    # 1. figure out last timestamp we already have
    conn = sqlite3.connect(DB)
    try:
        last = pd.read_sql("SELECT MAX(time) AS t FROM hourly", conn)["t"][0]
        start = pd.to_datetime(last, utc=True) + pd.Timedelta(hours=1)
    except Exception:
        start = datetime(1970, 1, 1, tzinfo=timezone.utc)
    end = datetime.now(timezone.utc)

    # 2. grab data
    if start >= end:
        print("Data is up to date. No new data to fetch.")
        return

    # For the Hourly call, provide naive UTC datetimes as the library seems to mix naive/aware internally
    # The actual start and end times remain timezone-aware UTC for our logic.
    start_for_hourly = start.replace(tzinfo=None)
    end_for_hourly = end.replace(tzinfo=None)

    print(f"Fetching data from {start.strftime('%Y-%m-%d %H:%M:%S %Z')} to {end.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    df = Hourly(POINT, start_for_hourly, end_for_hourly).fetch()

    if df.empty:
        print("Fetched data is empty. Nothing to persist.")
        conn.close()
        return

    # 3. persist
    # Ensure the DataFrame index is timezone-aware (UTC) before saving if it's not already
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC', ambiguous='infer') # be careful with ambiguous times if any
    else:
        df.index = df.index.tz_convert('UTC')
        
    df.to_sql("hourly", conn, if_exists="append", index_label="time")
    print(f"Appended {len(df)} new rows to the database.")
    conn.close()

if __name__ == "__main__":
    fetch_and_store() 