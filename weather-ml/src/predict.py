import joblib, pandas as pd, json, sqlite3, pathlib
from features import make_features
from fetch import DB
import os

# Must be consistent with TARGET_VARIABLES in train.py
TARGET_VARIABLES = ['temp', 'rhum', 'prcp', 'wspd', 'wdir_sin', 'wdir_cos']

DB_PATH = pathlib.Path(__file__).parent.parent / 'weather.sqlite'
MODEL_DIR = pathlib.Path(__file__).parent / "models"
FORECAST_OUTPUT_PATH = pathlib.Path(__file__).parent / "forecast_24h.json"

def make_predictions():
    conn = sqlite3.connect(DB_PATH)
    # Fetch more data for robust feature engineering, e.g., last 72 hours for up to 24h lags/rolling
    # Order by time ascending to easily get the latest records with .tail()
    query = "SELECT * FROM hourly ORDER BY time DESC LIMIT 72" # Select all columns
    print(f"Connecting to database at: {DB_PATH.resolve()}") # Print resolved path
    # Load data, explicitly parse time column later for robustness
    df = pd.read_sql(query, conn, index_col="time") # Keep index_col="time" for now
    conn.close()

    if df.empty:
        print("No data fetched from database for prediction.")
        # Create an empty JSON file or a file with an error message
        with open(FORECAST_OUTPUT_PATH, 'w') as f:
            json.dump({"message": "No data available for prediction."}, f)
        return

    # Robustly parse the index (which is the 'time' column)
    df.index = pd.to_datetime(df.index, utc=True, errors='coerce')
    # df.index.name = 'time' # Explicitly set/restore index name
    # df.dropna(subset=[df.index.name], inplace=True) # Drop rows where time index became NaT
    df = df[df.index.notna()] # More direct way to drop rows with NaT in index

    # Sort by time ascending as feature engineering expects this usually for lags
    df.sort_index(ascending=True, inplace=True)

    if df.empty:
        print("No valid data after timestamp parsing and NaT drop for prediction.")
        with open(FORECAST_OUTPUT_PATH, 'w') as f:
            json.dump({"message": "No valid data available after parsing for prediction."}, f)
        return

    # print(f"Raw data for prediction (last 5 rows after NaT filter and sort):\n{df.tail()}")

    # Engineer features using the same function as in training
    df_features_full = make_features(df)
    
    if df_features_full.shape[0] < 24:
        if not df_features_full.empty: # Only print warning if some data exists
            print(f"Warning: Not enough data rows ({df_features_full.shape[0]}) after feature engineering to make 24h forecast. Forecasting for available rows.")
        X_raw_for_forecast = df_features_full
    else:
        X_raw_for_forecast = df_features_full.iloc[-24:]

    X_raw_for_forecast = X_raw_for_forecast[pd.notna(X_raw_for_forecast.index)]
    
    if X_raw_for_forecast.empty:
        print("No valid recent data available for prediction. Skipping forecast generation.")
        output_path = pathlib.Path(__file__).parent / "forecast_24h.json"
        output_path.write_text(json.dumps({}, indent=2))
        return

    all_forecasts = {}

    for target_var in TARGET_VARIABLES:
        model_filename = MODEL_DIR / f"model_{target_var}.joblib"
        if not model_filename.exists():
            print(f"Model for {target_var} not found at {model_filename}. Skipping prediction for this target.")
            for timestamp_obj in X_raw_for_forecast.index:
                timestamp_str = timestamp_obj.strftime("%Y-%m-%d %H:%M")
                if timestamp_str not in all_forecasts:
                    all_forecasts[timestamp_str] = {}
                all_forecasts[timestamp_str][target_var] = None
            continue

        model = joblib.load(model_filename)
        
        # Select only the features the model was trained on
        # Ensure all feature_names_in_ are present in X_raw_for_forecast
        missing_features = [f for f in model.feature_names_in_ if f not in X_raw_for_forecast.columns]
        if missing_features:
            print(f"Error: Missing features for {target_var} model: {missing_features}.")
            for timestamp_obj in X_raw_for_forecast.index:
                timestamp_str = timestamp_obj.strftime("%Y-%m-%d %H:%M")
                if timestamp_str not in all_forecasts:
                    all_forecasts[timestamp_str] = {}
                all_forecasts[timestamp_str][target_var] = None
            continue
            
        X_predict = X_raw_for_forecast[model.feature_names_in_]
        preds = model.predict(X_predict)

        # Populate the all_forecasts dictionary
        for i, timestamp_obj in enumerate(X_raw_for_forecast.index):
            timestamp_str = timestamp_obj.strftime("%Y-%m-%d %H:%M")
            if timestamp_str not in all_forecasts:
                all_forecasts[timestamp_str] = {}
            
            # Rounding: temp, rhum, wspd to 1 decimal; prcp to 2; wdir_sin/cos to 3-4 for precision
            if target_var in ['temp', 'rhum', 'wspd']:
                all_forecasts[timestamp_str][target_var] = round(preds[i], 1)
            elif target_var == 'prcp':
                all_forecasts[timestamp_str][target_var] = round(preds[i], 2)
            elif target_var in ['wdir_sin', 'wdir_cos']:
                all_forecasts[timestamp_str][target_var] = round(preds[i], 4) # More precision for sin/cos
            else:
                all_forecasts[timestamp_str][target_var] = preds[i] # Default if no specific rounding

    # Save the combined forecast to JSON
    # The forecast_24h.json will be created in the same directory as predict.py (i.e. src/)
    output_path = pathlib.Path(__file__).parent / "forecast_24h.json"
    output_path.write_text(json.dumps(all_forecasts, indent=2))
    print(f"Saved multi-target forecast to {output_path}")

if __name__ == "__main__":
    make_predictions() 