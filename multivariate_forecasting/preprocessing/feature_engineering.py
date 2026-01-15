import pandas as pd
import os

def create_lag_features(
    df,
    variables,
    lags=24,
    log_file=None
):
    """
    Tạo lag features cho danh sách biến
    """
    df_lag = df.copy()

    for var in variables:
        for lag in range(1, lags + 1):
            df_lag[f"{var}_lag_{lag}"] = df_lag[var].shift(lag)

    rows_before = len(df_lag)
    df_lag = df_lag.dropna()
    rows_after = len(df_lag)

    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write("=== LAG FEATURES ===\n")
            f.write(f"Lag window (L): {lags}\n")
            f.write(f"Số biến gốc: {len(variables)}\n")
            f.write(f"Tổng số lag features: {lags * len(variables)}\n")
            f.write(f"Số dòng bị drop: {rows_before - rows_after}\n\n")

    return df_lag


def create_rolling_features(
    df,
    log_file=None
):
    """
    Rolling features (CHỈ dùng cho XGBoost)
    """
    df_roll = df.copy()

    df_roll["load_roll_mean_6"] = df_roll["load"].rolling(window=6).mean()
    df_roll["load_roll_std_24"] = df_roll["load"].rolling(window=24).std()
    df_roll["load_trend_24"] = df_roll["load"] - df_roll["load"].shift(24)

    rows_before = len(df_roll)
    df_roll = df_roll.dropna()
    rows_after = len(df_roll)

    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write("=== ROLLING FEATURES (XGBOOST ONLY) ===\n")
            f.write("Rolling mean: 6 giờ\n")
            f.write("Rolling std: 24 giờ\n")
            f.write("Load trend: load(t) - load(t-24)\n")
            f.write(f"Số dòng bị drop: {rows_before - rows_after}\n\n")

    return df_roll