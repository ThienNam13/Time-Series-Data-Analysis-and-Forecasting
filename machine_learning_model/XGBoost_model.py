import os
import logging
from datetime import datetime

import pandas as pd

# =========================================================
# 1. PATH CONFIGURATION
# =========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_PATH = os.path.join(DATA_DIR, "AEP_hourly.csv")

LOG_DIR = os.path.join(
    BASE_DIR,
    "machine_learning",
    "logs",
    "XGBoost_model",
    "AEP_hourly"
)

RUN_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(LOG_DIR, f"run_{RUN_TIME}")
os.makedirs(RUN_DIR, exist_ok=True)

LOG_FILE = os.path.join(RUN_DIR, "data_check.log")
SUMMARY_FILE = os.path.join(RUN_DIR, "data_summary.txt")
LAG_FILE = os.path.join(RUN_DIR, "lag_features.txt")

# =========================================================
# 2. LOGGING CONFIGURATION
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# =========================================================
# 3. FUNCTIONS
# =========================================================

def load_and_check_data(path: str) -> pd.DataFrame:
    logger.info("Loading dataset...")
    df = pd.read_csv(path)

    logger.info("Parsing datetime...")
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df.set_index("Datetime", inplace=True)

    return df


def check_missing_and_frequency(df: pd.DataFrame) -> None:
    missing_count = df.isna().sum().sum()
    freq = pd.infer_freq(df.index)

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write("DATA CHECK SUMMARY – AEP_hourly\n")
        f.write("=" * 50 + "\n\n")

        f.write("1. Missing Values\n")
        f.write(f"- Total missing values: {missing_count}\n\n")

        f.write("2. Time Frequency Check\n")
        f.write(f"- Inferred frequency: {freq}\n")

        if freq != "H":
            f.write("- WARNING: Missing hours detected!\n")
        else:
            f.write("- Frequency is consistent (Hourly)\n")

    logger.info(f"Missing values: {missing_count}")
    logger.info(f"Frequency inferred: {freq}")


def create_lag_features(df: pd.DataFrame, target_col="AEP_MW", n_lags=24) -> pd.DataFrame:
    logger.info(f"Creating lag features with n_lags = {n_lags}")

    lagged_df = df.copy()

    for lag in range(1, n_lags + 1):
        lagged_df[f"{target_col}_lag_{lag}"] = lagged_df[target_col].shift(lag)

    before_drop = len(lagged_df)
    lagged_df.dropna(inplace=True)
    after_drop = len(lagged_df)

    dropped_rows = before_drop - after_drop

    with open(LAG_FILE, "w", encoding="utf-8") as f:
        f.write("LAG FEATURE ENGINEERING REPORT\n")
        f.write("=" * 55 + "\n\n")

        f.write(f"Number of lags: {n_lags}\n")
        f.write(f"Dropped rows due to lagging: {dropped_rows}\n\n")

        f.write("Lag features created:\n")
        for col in lagged_df.columns:
            if "lag_" in col:
                f.write(f"- {col}\n")

    logger.info(f"Lag features created: {n_lags}")
    logger.info(f"Rows dropped due to lagging: {dropped_rows}")

    return lagged_df

# =========================================================
# 4. MAIN PIPELINE
# =========================================================

def main():
    logger.info("START DATA PREPARATION – XGBOOST")

    df = load_and_check_data(DATA_PATH)
    check_missing_and_frequency(df)

    supervised_df = create_lag_features(df)

    logger.info("END DATA PREPARATION – XGBOOST")
    logger.info(f"Artifacts saved at: {RUN_DIR}")


if __name__ == "__main__":
    main()
