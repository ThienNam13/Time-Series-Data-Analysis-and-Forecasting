import os
import logging
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# 1. PATH CONFIGURATION
# =========================================================

# Project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data & Log directories
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR = os.path.join(
    BASE_DIR,
    "statistical_model",
    "logs",
    "ARIMA_model",
    "a10"
)

os.makedirs(LOG_DIR, exist_ok=True)

# Timestamp for this run
RUN_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")

# File paths
DATA_PATH = os.path.join(DATA_DIR, "a10.csv")
LOG_FILE = os.path.join(LOG_DIR, f"ARIMA_model_{RUN_TIME}.log")
PLOT_PATH = os.path.join(LOG_DIR, f"time_series_{RUN_TIME}.png")
EDA_TEXT_PATH = os.path.join(LOG_DIR, f"eda_summary_{RUN_TIME}.txt")


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

def load_data(path: str) -> pd.DataFrame:
    """Load dataset from CSV file."""
    try:
        df = pd.read_csv(path)
        logger.info("Dataset loaded successfully")
        logger.info(f"Dataset shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Convert date column to DatetimeIndex."""
    try:
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        logger.info("Date column converted to DatetimeIndex")
        return df
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise


def plot_time_series(df: pd.DataFrame, output_path: str) -> None:
    """Plot and save time series."""
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["value"])
    plt.title("Monthly Retail Sales (a10)")
    plt.xlabel("Time")
    plt.ylabel("Sales")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Time series plot saved at {output_path}")


def write_eda_summary(path: str) -> None:
    """Write EDA summary text file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("EDA SUMMARY – DATASET a10 (Monthly Retail Sales)\n")
        f.write("=" * 55 + "\n\n")

        f.write("Dataset description:\n")
        f.write("- Monthly retail sales data (US).\n")
        f.write("- Frequency: Monthly.\n")
        f.write("- Expected characteristics: trend and seasonality.\n\n")

        f.write("1. Trend:\n")
        f.write(
            "- Quan sát đồ thị chuỗi thời gian cho thấy xu hướng tăng dài hạn rõ rệt.\n"
            "- Điều này cho thấy mức trung bình của chuỗi thay đổi theo thời gian.\n\n"
        )

        f.write("2. Seasonality:\n")
        f.write(
            "- Chuỗi có các dao động lặp lại theo chu kỳ hàng năm (12 tháng).\n"
            "- Biên độ dao động tương đối ổn định, cho thấy tính mùa vụ rõ ràng.\n\n"
        )

        f.write("3. Stationarity:\n")
        f.write(
            "- Chuỗi có xu hướng và mùa vụ, do đó trung bình không ổn định theo thời gian.\n"
            "- Chuỗi nhiều khả năng KHÔNG dừng.\n"
            "- Cần kiểm định ADF và thực hiện sai phân trong các bước tiếp theo.\n\n"
        )

        f.write(
            "Nhận xét trên được đưa ra dựa trên quan sát trực quan từ đồ thị "
            "chuỗi thời gian được lưu trong thư mục logs.\n"
        )

    logger.info(f"EDA summary written to {path}")


# =========================================================
# 4. MAIN PIPELINE
# =========================================================

def main():
    logger.info("========== START EDA PIPELINE ==========")

    df = load_data(DATA_PATH)
    df = preprocess_data(df)
    plot_time_series(df, PLOT_PATH)
    write_eda_summary(EDA_TEXT_PATH)

    logger.info("=========== END EDA PIPELINE ===========")


if __name__ == "__main__":
    main()
