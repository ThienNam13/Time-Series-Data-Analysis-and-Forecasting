import os
import logging
from datetime import datetime

import pandas as pd

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
    logger.info("Datetime parsed and set as index")
    return df


def check_data_quality(df: pd.DataFrame):
    missing_values = df.isna().sum().sum()
    inferred_freq = pd.infer_freq(df.index)

    with open(INFO_TXT, "w", encoding="utf-8") as f:
        f.write("DATA QUALITY CHECK – AEP_hourly\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total missing values: {missing_values}\n")
        f.write(f"Inferred frequency: {inferred_freq}\n\n")

        if missing_values == 0:
            f.write(" No missing values detected.\n")
        else:
            f.write(" Dataset contains missing values.\n")

        if inferred_freq == "H":
            f.write(" Hourly frequency is consistent.\n")
        else:
            f.write(" Hourly frequency is broken or irregular.\n")

    logger.info("Data quality check saved")


def create_lag_features(series: pd.Series, n_lags: int = 24) -> pd.DataFrame:
    df = pd.DataFrame({"y": series})

    for lag in range(1, n_lags + 1):
        df[f"y_lag_{lag}"] = series.shift(lag)

    rows_before = len(df)
    df.dropna(inplace=True)
    rows_after = len(df)

    logger.info(f"Number of lags: {n_lags}")
    logger.info(f"Rows dropped due to lagging: {rows_before - rows_after}")
    logger.info(f"Feature columns: {list(df.columns)}")

    return df

# =========================================================
# 4. MAIN
# =========================================================

def main():
    logger.info("========== START XGBOOST DATA PIPELINE ==========")

    df = load_data(DATA_PATH)
    df = preprocess_datetime(df)

    check_data_quality(df)

    supervised_df = create_lag_features(df.iloc[:, 0], n_lags=24)

    logger.info(f"Supervised dataset shape: {supervised_df.shape}")
    logger.info("=========== END XGBOOST DATA PIPELINE ===========")

if __name__ == "__main__":
    main()
