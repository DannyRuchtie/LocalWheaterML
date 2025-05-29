from fetch import DB
from features import make_features
import pandas as pd, sqlite3, joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

def train():
    df = pd.read_sql("SELECT * FROM hourly", sqlite3.connect(DB), parse_dates=["time"], index_col="time")
    print(f"Shape of df before make_features: {df.shape}") # DEBUG
    df = make_features(df)
    print(f"Shape of df after make_features: {df.shape}") # DEBUG

    # Remove rows where the target variable 'temp' is NaN
    initial_rows = df.shape[0]
    df.dropna(subset=['temp'], inplace=True)
    print(f"Shape of df after dropping NaNs in 'temp': {df.shape}. Rows removed: {initial_rows - df.shape[0]}") # DEBUG

    # Add a check to prevent further execution if df is empty
    if df.empty:
        print("DataFrame is empty after dropping NaNs in 'temp'. Skipping training.")
        return

    X = df.drop(columns=["temp"])  # target is hourly temperature
    y = df["temp"]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = HistGradientBoostingRegressor(loss="squared_error")
    model.fit(X_train, y_train)
    print("val R²:", model.score(X_val, y_val))
    joblib.dump(model, "model_temp.joblib")

if __name__ == "__main__":
    train() 