# =========================================================
# RANDOM FOREST FOR TIME SERIES FORECASTING
# =========================================================

import os
import logging
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# =========================================================
# 1. PATH CONFIGURATION
# =========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "machine_learning_model", "data")
LOG_DIR = os.path.join(
    BASE_DIR,
    "machine_learning_model",
    "logs",
    "RandomForest_model",
    "AEPhourly"
)

os.makedirs(LOG_DIR, exist_ok=True)

RUN_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(LOG_DIR, f"run_{RUN_TIME}")
os.makedirs(RUN_DIR, exist_ok=True)

DATA_PATH = os.path.join(DATA_DIR, "AEP_hourly.csv")

LOG_FILE = os.path.join(RUN_DIR, "random_forest.log")
MODEL_INFO_TXT = os.path.join(RUN_DIR, "model_info.txt")
METRICS_TXT = os.path.join(RUN_DIR, "evaluation_metrics.txt")

# =========================================================
# 2. LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ],
    force=True
)

logger = logging.getLogger(__name__)

# =========================================================
# 3. FUNCTIONS
# =========================================================

def load_data(path):
    """Load dataset"""
    df = pd.read_csv(path)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df.set_index("Datetime", inplace=True)
    df.sort_index(inplace=True)

    logger.info(f"Loaded data shape: {df.shape}")
    return df


def create_lag_features(series, n_lags=24):
    """
    Convert time series into supervised learning format
    """
    df = pd.DataFrame({"y": series})

    for lag in range(1, n_lags + 1):
        df[f"y_lag_{lag}"] = series.shift(lag)

    df.dropna(inplace=True)

    logger.info(f"Lag features created: {n_lags}")
    logger.info(f"Supervised data shape: {df.shape}")
    return df


def time_series_train_test_split(df, train_ratio=0.8):
    """
    Split data into train/test without shuffling
    """
    split_idx = int(len(df) * train_ratio)

    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    X_train = train.drop(columns="y")
    y_train = train["y"]
    X_test = test.drop(columns="y")
    y_test = test["y"]

    logger.info(f"Train samples: {len(train)}")
    logger.info(f"Test samples: {len(test)}")

    return X_train, X_test, y_train, y_test


def evaluate_model(y_true, y_pred):
    """Compute MAE, RMSE, MAPE"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    mask = y_true != 0
    mape = np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])) * 100

    return mae, rmse, mape

# =========================================================
# MAIN PIPELINE
# =========================================================

def main():
    logger.info("========== START RANDOM FOREST PIPELINE ==========")

    # -----------------------------------------------------
    # STEP 1: LOAD DATA
    # -----------------------------------------------------
    df = load_data(DATA_PATH)

    # -----------------------------------------------------
    # STEP 2: SUPERVISED LEARNING (LAG FEATURES)
    # -----------------------------------------------------
    supervised_df = create_lag_features(df.iloc[:, 0], n_lags=24)

    # -----------------------------------------------------
    # STEP 3: TRAIN / TEST SPLIT (TIME ORDER)
    # -----------------------------------------------------
    X_train, X_test, y_train, y_test = time_series_train_test_split(supervised_df)

    # -----------------------------------------------------
    # STEP 4: TRAIN RANDOM FOREST
    # -----------------------------------------------------
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    logger.info("Random Forest model trained")

    # -----------------------------------------------------
    # STEP 5: SAVE MODEL INFO
    # -----------------------------------------------------
    with open(MODEL_INFO_TXT, "w", encoding="utf-8") as f:
        f.write("RANDOM FOREST REGRESSOR\n")
        f.write("=" * 50 + "\n\n")
        f.write("Model purpose:\n")
        f.write("- Time series forecasting using lag features\n\n")

        f.write("Hyperparameters:\n")
        f.write(f"- n_estimators: {model.n_estimators}\n")
        f.write(f"- max_depth  : {model.max_depth}\n")
        f.write(f"- random_state: {model.random_state}\n\n")

        f.write("Input features:\n")
        f.write("- Lagged values: y(t-1) ... y(t-24)\n")

    logger.info(f"Model info saved → {MODEL_INFO_TXT}")

    # -----------------------------------------------------
    # STEP 6: PREDICTION & EVALUATION
    # -----------------------------------------------------
    y_pred = model.predict(X_test)

    mae, rmse, mape = evaluate_model(y_test, y_pred)

    with open(METRICS_TXT, "w", encoding="utf-8") as f:
        f.write("RANDOM FOREST EVALUATION METRICS\n")
        f.write("=" * 50 + "\n")
        f.write(f"MAE : {mae:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"MAPE: {mape:.2f}%\n")

    logger.info(f"MAE  = {mae:.4f}")
    logger.info(f"RMSE = {rmse:.4f}")
    logger.info(f"MAPE = {mape:.2f}%")
    logger.info(f"Metrics saved → {METRICS_TXT}")

    logger.info("=========== END RANDOM FOREST PIPELINE ===========")

if __name__ == "__main__":
    main()
