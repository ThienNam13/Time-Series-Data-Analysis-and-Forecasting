import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt

from data_loader import load_and_prepare_multivariate_data

# ===================== CONFIG =====================
LAG_WINDOW = 24
ROLLING_MEAN_WINDOW = 6
ROLLING_STD_WINDOW = 24

FEATURE_COLS = ["load", "temperature", "humidity", "wind_speed"]

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
DATA_DIR = os.path.join(BASE_DIR, "data")

os.makedirs(LOG_DIR, exist_ok=True)

# ===================== LOGGING =====================
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "feature_engineering.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ===================== FUNCTIONS =====================
def create_lag_features(df, cols, lag_window):
    df_lag = df.copy()
    for col in cols:
        for lag in range(1, lag_window + 1):
            df_lag[f"{col}_lag_{lag}"] = df_lag[col].shift(lag)
    return df_lag


def create_rolling_features(df):
    """
    Rolling features chỉ dùng cho XGBoost
    """
    df_feat = df.copy()

    df_feat["load_roll_mean_6h"] = df_feat["load"].rolling(ROLLING_MEAN_WINDOW).mean()
    df_feat["load_roll_std_24h"] = df_feat["load"].rolling(ROLLING_STD_WINDOW).std()
    df_feat["load_trend_24h"] = df_feat["load"] - df_feat["load"].shift(24)

    return df_feat


def plot_sample_features(df):
    plt.figure(figsize=(10, 5))
    plt.plot(df["load"], label="Load")
    plt.plot(df["load_roll_mean_6h"], label="Rolling Mean 6h")
    plt.legend()
    plt.title("Load vs Rolling Mean Feature")
    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, "rolling_feature_example.png"))
    plt.close()


# ===================== MAIN =====================
def main():
    logging.info("START FEATURE ENGINEERING")

    # ===== LOAD & PREPARE MULTIVARIATE DATA =====
    df = load_and_prepare_multivariate_data(
        load_path=os.path.join(DATA_DIR, "LD2011_2014.txt"),
        weather_path=os.path.join(DATA_DIR, "weather.csv"),
        freq="H"
    )

    original_rows = df.shape[0]

    # ===== LAG FEATURES (ML & DL) =====
    df_lagged = create_lag_features(df, FEATURE_COLS, LAG_WINDOW)

    # ===== ROLLING FEATURES (ONLY XGBOOST) =====
    df_features = create_rolling_features(df_lagged)

    # ===== DROP NA =====
    df_features.dropna(inplace=True)
    dropped_rows = original_rows - df_features.shape[0]

    # ===== LOGGING =====
    logging.info(f"Lag window (L): {LAG_WINDOW}")
    logging.info(f"Total features after FE: {df_features.shape[1]}")
    logging.info(f"Rows dropped due to lag/rolling: {dropped_rows}")

    # ===== SAVE OUTPUT =====
    df_features.to_csv(os.path.join(LOG_DIR, "feature_engineered_data.csv"))

    # ===== SAVE PLOT =====
    plot_sample_features(df_features)

    logging.info("FEATURE ENGINEERING COMPLETED")


if __name__ == "__main__":
    main()
