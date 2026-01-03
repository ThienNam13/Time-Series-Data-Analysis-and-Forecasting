import os
import logging
from datetime import datetime

import pandas as pd
import numpy as np
from xgboost import XGBRegressor

# =========================================================
# 1. PATH CONFIGURATION
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR = os.path.join(BASE_DIR, "log")

os.makedirs(LOG_DIR, exist_ok=True)

RUN_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")

DATA_PATH = os.path.join(DATA_DIR, "AEP_hourly.csv")
LOG_FILE = os.path.join(LOG_DIR, f"xgboost_pipeline_{RUN_TIME}.log")
INFO_TXT = os.path.join(LOG_DIR, f"data_check_{RUN_TIME}.txt")
FORECAST_TXT = os.path.join(LOG_DIR, f"multi_step_forecast_{RUN_TIME}.txt")

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

def load_data(path: str) -> pd.DataFrame:
    logger.info(f"Loading dataset from: {path}")
    df = pd.read_csv(path)
    logger.info(f"Dataset shape: {df.shape}")
    return df


def preprocess_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df.set_index("Datetime", inplace=True)
    df.sort_index(inplace=True)
    logger.info("Datetime parsed & set as index")
    return df


def check_data_quality(df: pd.DataFrame):
    missing = df.isna().sum().sum()
    freq = pd.infer_freq(df.index)

    with open(INFO_TXT, "w", encoding="utf-8") as f:
        f.write("DATA QUALITY CHECK – AEP_hourly\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Missing values: {missing}\n")
        f.write(f"Inferred frequency: {freq}\n")

        if missing == 0:
            f.write("→ No missing values detected.\n")
        else:
            f.write("→ Dataset contains missing values.\n")

        if freq == "H":
            f.write("→ Hourly frequency is consistent.\n")
        else:
            f.write("→ WARNING: Hourly frequency is broken.\n")

    logger.info("Data quality check saved")


def create_lag_features(series: pd.Series, n_lags: int = 24) -> pd.DataFrame:
    """
    Convert time series to supervised learning format
    """
    df = pd.DataFrame({"y": series})

    for lag in range(1, n_lags + 1):
        df[f"y_lag_{lag}"] = series.shift(lag)

    before = len(df)
    df.dropna(inplace=True)
    after = len(df)

    logger.info(f"Number of lags: {n_lags}")
    logger.info(f"Rows dropped due to lagging: {before - after}")
    logger.info(f"Feature columns: {list(df.columns)}")

    return df


def multi_step_forecast(model, last_window, n_steps=24):
    """
    Recursive multi-step forecast
    """
    forecasts = []
    current_window = last_window.copy()

    for step in range(n_steps):
        pred = model.predict(current_window.reshape(1, -1))[0]
        forecasts.append(pred)

        current_window = np.roll(current_window, -1)
        current_window[-1] = pred

    return forecasts

# =========================================================
# 4. MAIN
# =========================================================

def main():
    logger.info("========== START XGBOOST PIPELINE ==========")

    # Load & preprocess
    df = load_data(DATA_PATH)
    df = preprocess_datetime(df)

    # Data quality check
    check_data_quality(df)

    # Supervised learning
    supervised_df = create_lag_features(df.iloc[:, 0], n_lags=24)

    X = supervised_df.drop(columns="y")
    y = supervised_df["y"]

    # Train simple XGBoost model
    model = XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        objective="reg:squarederror",
        random_state=42
    )

    model.fit(X, y)
    logger.info("XGBoost model trained")

    # Multi-step forecast (24 hours)
    last_window = X.iloc[-1].values
    forecasts = multi_step_forecast(model, last_window, n_steps=24)

    # Save forecast results
    with open(FORECAST_TXT, "w", encoding="utf-8") as f:
        f.write("MULTI-STEP FORECAST (Next 24 Hours)\n")
        f.write("=" * 50 + "\n")
        for i, val in enumerate(forecasts, 1):
            f.write(f"t+{i}: {val:.2f}\n")

    logger.info("Multi-step forecast completed")
    logger.info(f"Forecast saved to: {FORECAST_TXT}")
    logger.info("=========== END XGBOOST PIPELINE ===========")

if __name__ == "__main__":
    main()
