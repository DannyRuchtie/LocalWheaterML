import joblib, pandas as pd, json, sqlite3, pathlib
from features import make_features
from fetch import DB

def make_prediction():
    conn = sqlite3.connect(DB)
    df = pd.read_sql("SELECT * FROM hourly ORDER BY time DESC LIMIT 48", conn,
                     parse_dates=["time"], index_col="time").sort_index()
    X_raw_features = make_features(df).iloc[-24:]       # last 24 rows, includes 'temp' and other raw columns
    model = joblib.load("model_temp.joblib")

    # Select only the features the model was trained on
    X_predict = X_raw_features[model.feature_names_in_]

    preds = model.predict(X_predict)
    # The forecast part uses X.index, so we should use X_raw_features.index
    forecast = dict(zip(X_raw_features.index.strftime("%Y-%m-%d %H:%M"), preds.round(1)))
    pathlib.Path("forecast_24h.json").write_text(json.dumps(forecast, indent=2))

if __name__ == "__main__":
    make_prediction() 