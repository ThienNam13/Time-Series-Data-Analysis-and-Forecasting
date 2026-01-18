# preprocessing/feature_engineering.py

import pandas as pd


# =====================================================
# LAG FEATURES (FOR ML & DL)
# =====================================================
def create_lag_features(
    data: pd.DataFrame,
    variables,
    max_lag=24
):
    """
    Create lag features for given variables

    Example:
        load_lag_1 ... load_lag_24
        temperature_lag_1 ... temperature_lag_24
    """

    df = data.copy()

    lag_features = []

    for var in variables:
        for l in range(1, max_lag + 1):
            col_name = f"{var}_lag_{l}"
            df[col_name] = df[var].shift(l)
            lag_features.append(col_name)

    df.dropna(inplace=True)

    return df, lag_features


# =====================================================
# ROLLING FEATURES (MAINLY FOR XGBOOST)
# =====================================================
def create_rolling_features(data: pd.DataFrame):
    """
    Create rolling statistics on LOAD variable
    """

    df = data.copy()

    df["load_roll_mean_6"] = df["load"].rolling(window=6).mean()
    df["load_roll_std_24"] = df["load"].rolling(window=24).std()
    df["load_trend_24"] = df["load"] - df["load"].shift(24)

    df.dropna(inplace=True)

    return df


# =====================================================
# FULL FEATURE ENGINEERING PIPELINE (FOR XGBOOST)
# =====================================================
def build_ml_features(
    scaled_data: pd.DataFrame,
    variables,
    max_lag=24
):
    """
    Apply:
        - lag features
        - rolling features

    Input must be SCALED data
    """

    df_lag, lag_features = create_lag_features(
        scaled_data,
        variables=variables,
        max_lag=max_lag
    )

    df_full = create_rolling_features(df_lag)

    return df_full, lag_features
