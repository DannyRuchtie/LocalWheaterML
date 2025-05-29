import sqlite3
import pandas as pd
import pathlib

DB_PATH = pathlib.Path(__file__).parent.parent / 'weather.sqlite'

def minimal_db_read_test():
    print(f"Attempting to connect to DB: {DB_PATH.resolve()}")
    if not DB_PATH.exists():
        print("Database file does not exist at the specified path.")
        return
    
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        print("Successfully connected to SQLite database.")
        
        # Attempt the exact type of read that fails in predict.py (simplified query)
        query = "SELECT time, temp FROM hourly ORDER BY time DESC LIMIT 5"
        print(f"Executing query: {query}")
        df = pd.read_sql(query, conn, index_col="time")
        print("Successfully executed pd.read_sql.")
        print("Sample data:")
        print(df.head())
        
    except sqlite3.Error as e:
        print(f"SQLite error during minimal_db_read_test: {e}")
    except pd.errors.DatabaseError as e_pd:
        print(f"Pandas DatabaseError during minimal_db_read_test: {e_pd}")        
    except Exception as e:
        print(f"An unexpected error occurred in minimal_db_read_test: {e}")
    finally:
        if conn:
            print("Closing connection.")
            conn.close()

if __name__ == "__main__":
    minimal_db_read_test() 