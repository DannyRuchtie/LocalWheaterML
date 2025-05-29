import pandas as pd
import numpy as np

LAGS = [1, 6, 24]  # h
ROLLS = [(6, "mean"), (24, "mean")]
# Variables for which to create lag/roll features directly
SIMPLE_FEATURE_VARS = ['temp', 'rhum', 'prcp', 'wspd']
# Cyclical encoding for time features remains
CYCLICAL_TIME = ["hour", "dayofyear"]

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Create lag and rolling features for simple variables
    for var in SIMPLE_FEATURE_VARS:
        if var not in out.columns:
            # If a variable is missing (e.g., all NaNs and dropped earlier, or not in API response)
            # Create NaN columns for it to maintain consistent feature set, model will handle NaNs
            out[var] = np.nan 
        for h in LAGS:
            out[f"{var}_lag{h}"] = out[var].shift(h)
        for win, agg in ROLLS:
            # Make sure window is not larger than available data for the variable
            # However, rolling().mean() handles this by returning NaNs if window is too large for available points
            out[f"{var}_{agg}{win}"] = out[var].rolling(window=win, min_periods=1).mean()

    # Feature engineering for wind direction (wdir)
    if 'wdir' in out.columns:
        out['wdir_sin'] = np.sin(2 * np.pi * out['wdir'] / 360.0)
        out['wdir_cos'] = np.cos(2 * np.pi * out['wdir'] / 360.0)

        # Now create lag/roll features for wdir_sin and wdir_cos
        for var_comp in ['wdir_sin', 'wdir_cos']:
            for h in LAGS:
                out[f"{var_comp}_lag{h}"] = out[var_comp].shift(h)
            for win, agg in ROLLS:
                out[f"{var_comp}_{agg}{win}"] = out[var_comp].rolling(window=win, min_periods=1).mean()
    else:
        # If wdir is not present, create NaN columns for its derived features
        for comp in ['sin', 'cos']:
            for h in LAGS:
                out[f"wdir_{comp}_lag{h}"] = np.nan
            for win, agg in ROLLS:
                out[f"wdir_{comp}_{agg}{win}"] = np.nan
        out['wdir_sin'] = np.nan # Base sin/cos features also NaN
        out['wdir_cos'] = np.nan

    # Cyclical time encodings (hour, dayofyear) - these use the DataFrame index
    idx = out.index
    out["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24.0)
    out["doy_sin"]  = np.sin(2 * np.pi * idx.dayofyear / 365.25)
    out["doy_cos"]  = np.cos(2 * np.pi * idx.dayofyear / 365.25)
    
    # The .dropna() was removed previously; HistGradientBoostingRegressor handles NaNs in features.
    # However, target variables will need NaNs removed before training each specific model.
    return out 