from fetch import DB
from features import make_features
import pandas as pd, sqlite3, joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
import os
import pathlib

# Define all variables that will be predicted. 
# These will also be excluded from features X when training for any specific target.
TARGET_VARIABLES = ['temp', 'rhum', 'prcp', 'wspd', 'wdir_sin', 'wdir_cos']
RAW_WIND_DIR_COL = 'wdir' # Raw wind direction, also to be excluded from features X

def train():
    # Create the directory for models if it doesn't exist
    model_dir = pathlib.Path(__file__).parent / "models"
    os.makedirs(model_dir, exist_ok=True)

    df_full = pd.read_sql("SELECT * FROM hourly", sqlite3.connect(DB), parse_dates=["time"], index_col="time")
    print(f"Shape of df_full before make_features: {df_full.shape}")
    
    df_features = make_features(df_full)
    print(f"Shape of df_features after make_features: {df_features.shape}")

    # Columns to exclude from the feature set X for any model.
    # This includes all target variables and the original wind direction column.
    cols_to_drop_for_X = TARGET_VARIABLES + [RAW_WIND_DIR_COL]
    # Ensure we only try to drop columns that actually exist in df_features
    cols_to_drop_for_X = [col for col in cols_to_drop_for_X if col in df_features.columns]

    for target_var in TARGET_VARIABLES:
        print(f"\n--- Training model for {target_var} ---")
        df_target = df_features.copy() # Use a copy for each target

        # Check if the target variable itself exists in the dataframe
        if target_var not in df_target.columns:
            print(f"Target variable {target_var} not found in DataFrame. Skipping training for this target.")
            continue
            
        # Remove rows where the current target variable is NaN
        initial_rows = df_target.shape[0]
        df_target.dropna(subset=[target_var], inplace=True)
        rows_removed = initial_rows - df_target.shape[0]
        print(f"Shape of df_target for {target_var} after dropping its NaNs: {df_target.shape}. Rows removed: {rows_removed}")

        if df_target.empty:
            print(f"DataFrame is empty for {target_var} after dropping its NaNs. Skipping training.")
            continue

        X = df_target.drop(columns=cols_to_drop_for_X, errors='ignore') # errors='ignore' is belt-and-suspenders
        y = df_target[target_var]
        
        # Check if X is empty (e.g. if all columns were target columns or dropped)
        if X.empty:
            print(f"Feature set X is empty for {target_var}. Skipping training.")
            continue
            
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

        if X_train.empty or X_val.empty:
            print(f"Training or validation set is empty for {target_var} after split. Min samples for split might not be met. Skipping.")
            continue

        model = HistGradientBoostingRegressor(loss="squared_error")
        model.fit(X_train, y_train)
        
        score = model.score(X_val, y_val)
        print(f"Validation R² for {target_var}: {score}")
        
        model_filename = model_dir / f"model_{target_var}.joblib"
        joblib.dump(model, model_filename)
        print(f"Saved model for {target_var} to {model_filename}")

if __name__ == "__main__":
    train() 