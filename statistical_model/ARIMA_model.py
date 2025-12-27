import os
import logging
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np
from math import sqrt

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
STATIONARITY_PATH = os.path.join(LOG_DIR, f"stationarity_test_{RUN_TIME}.txt")
# thêm path cho đồ thị sai phân và kết quả kiểm định ADF sau sai phân
DIFF_PLOT_PATH = os.path.join(LOG_DIR, f"diff_series_{RUN_TIME}.png")
ADF_DIFF_PATH = os.path.join(LOG_DIR, f"stationarity_test_diff_{RUN_TIME}.txt")
# thêm path cho 4. ACF & PACF cho chuỗi đã dừng
ACF_PATH = os.path.join(LOG_DIR, f"acf_{RUN_TIME}.png")
PACF_PATH = os.path.join(LOG_DIR, f"pacf_{RUN_TIME}.png")
ACF_PACF_TXT = os.path.join(LOG_DIR, f"ACF_PACF_Analysis_{RUN_TIME}.txt")
ZTABLE_PATH = os.path.join(LOG_DIR, f"z_table_{RUN_TIME}.txt")

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


def perform_adf_test(series: pd.Series, output_path: str) -> None:
    """Perform ADF test on a series and write results to a text file."""
    try:
        result = adfuller(series.dropna(), autolag='AIC')
        adf_stat = result[0]
        p_value = result[1]
        used_lag = result[2]
        n_obs = result[3]
        crit_values = result[4]

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("ADF STATIONARITY TEST – ORIGINAL SERIES\n")
            f.write("=" * 55 + "\n\n")

            f.write("H0 (null hypothesis): The time series has a unit root (is non-stationary).\n")
            f.write("H1 (alternative): The time series is stationary.\n\n")

            f.write(f"ADF Statistic: {adf_stat:.6f}\n")
            f.write(f"p-value: {p_value:.6f}\n")
            f.write(f"Used lag: {used_lag}\n")
            f.write(f"Number of observations: {n_obs}\n\n")

            f.write("Critical Values:\n")
            for key, val in crit_values.items():
                f.write(f"  {key}: {val:.6f}\n")
            f.write("\n")

            # Conclusion at 5% significance
            alpha = 0.05
            if p_value < alpha:
                conclusion = "Reject H0 -> The series is stationary (dừng)."
            else:
                conclusion = "Fail to reject H0 -> The series is non-stationary (không dừng)."

            f.write(f"Conclusion (alpha = {alpha}): {conclusion}\n\n")

            f.write("Giải thích ngắn gọn:\n")
            f.write("- H0 của ADF: chuỗi có unit root (không dừng).\n")
            f.write("- p-value cho biết xác suất quan sát được dữ liệu (hoặc mạnh hơn) khi H0 đúng.\n")
            f.write("- Nếu p-value nhỏ hơn mức ý nghĩa (ví dụ 0.05) thì bác bỏ H0, tức chuỗi dừng.\n")

        logger.info(f"ADF test results written to {output_path}")
    except Exception as e:
        logger.error(f"ADF test failed: {e}")
        raise

# thêm hàm thực hiện sai phân và kiểm định ADF sau sai phân
def difference_series(df: pd.DataFrame) -> pd.Series:
    """Perform first differencing."""
    diff_series = df["value"].diff()
    logger.info("First differencing applied (d = 1)")
    return diff_series
# thêm hàm plot chuỗi sai phân
def plot_diff_series(series: pd.Series, output_path: str) -> None:
    """Plot and save differenced time series."""
    plt.figure(figsize=(10, 5))
    plt.plot(series.index, series)
    plt.title("Differenced Time Series (d = 1)")
    plt.xlabel("Time")
    plt.ylabel("Differenced Value")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Differenced series plot saved at {output_path}")
# Thêm hàm vẽ ACF & PACF
def plot_acf_pacf(series: pd.Series):
    """Plot ACF & PACF for differenced stationary series"""
    try:
        # ACF
        plt.figure(figsize=(8, 4))
        plot_acf(series.dropna(), lags=40)
        plt.title("ACF of Differenced Series (d = 1)")
        plt.tight_layout()
        plt.savefig(ACF_PATH)
        plt.close()

        # PACF
        plt.figure(figsize=(8, 4))
        plot_pacf(series.dropna(), lags=40, method="ywm")
        plt.title("PACF of Differenced Series (d = 1)")
        plt.tight_layout()
        plt.savefig(PACF_PATH)
        plt.close()

        logger.info(f"ACF saved → {ACF_PATH}")
        logger.info(f"PACF saved → {PACF_PATH}")

    except Exception as e:
        logger.error(f"Failed to generate ACF/PACF plots: {e}")
        raise
# Hàm tính Z-TABLE + CONFIDENCE
def write_z_table(series: pd.Series):
    """Compute confidence intervals and export z table."""
    n = len(series.dropna())
    z90 = 1.645 / sqrt(n)
    z95 = 1.96 / sqrt(n)
    z99 = 2.576 / sqrt(n)

    with open(ZTABLE_PATH, "w", encoding="utf-8") as f:
        f.write("Z TABLE – CONFIDENCE INTERVALS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Sample size (n): {n}\n\n")
        f.write("Confidence Limits:\n")
        f.write(f"90%  → ±{z90:.6f}\n")
        f.write(f"95%  → ±{z95:.6f}\n")
        f.write(f"99%  → ±{z99:.6f}\n")

    logger.info(f"Z-table written to {ZTABLE_PATH}")

    return z95
# Hàm phân tích ACF – PACF và gợi ý p q
def analyze_acf_pacf(series: pd.Series, ci_threshold: float):
    """Analyze ACF & PACF to determine ARIMA p & q"""

    from statsmodels.tsa.stattools import acf, pacf

    data = series.dropna()
    acf_vals = acf(data, nlags=40)
    pacf_vals = pacf(data, nlags=40)

    sig_acf_lags = [i for i, v in enumerate(acf_vals) if abs(v) > ci_threshold and i != 0]
    sig_pacf_lags = [i for i, v in enumerate(pacf_vals) if abs(v) > ci_threshold and i != 0]

    p = sig_pacf_lags[0] if sig_pacf_lags else 0
    q = sig_acf_lags[0] if sig_acf_lags else 0

    with open(ACF_PACF_TXT, "w", encoding="utf-8") as f:
        f.write("ACF & PACF ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write("Dataset: a10 – Differenced Series (d = 1)\n\n")

        f.write("SIGNIFICANT LAGS (beyond 95% CI):\n")
        f.write(f"- ACF significant lags → {sig_acf_lags}\n")
        f.write(f"- PACF significant lags → {sig_pacf_lags}\n\n")

        f.write("INTERPRETATION:\n")
        f.write("- ACF giúp xác định q (MA part)\n")
        f.write("- PACF giúp xác định p (AR part)\n\n")

        f.write(f"Suggested parameters:\n")
        f.write(f"p = {p}\n")
        f.write(f"d = 1\n")
        f.write(f"q = {q}\n\n")

        f.write("Conclusion:\n")
        if sig_acf_lags or sig_pacf_lags:
            f.write("- Chuỗi sau sai phân vẫn còn cấu trúc tự tương quan.\n")
            f.write("- Mô hình ARIMA có thể phù hợp để mô hình hóa.\n")
        else:
            f.write("- Không có lag nào vượt ngưỡng → gần giống white noise\n")

    logger.info(f"ACF/PACF analysis saved at {ACF_PACF_TXT}")
# =========================================================
# 4. MAIN PIPELINE
# =========================================================

def main():
    logger.info(f"Running file: {__file__}")
    logger.info("========== START EDA PIPELINE ==========")

    df = load_data(DATA_PATH)
    df = preprocess_data(df)

    # Original time series
    plot_time_series(df, PLOT_PATH)
    perform_adf_test(df["value"], STATIONARITY_PATH)
    write_eda_summary(EDA_TEXT_PATH)

    # First differencing (d = 1)
    diff_series = difference_series(df)
    plot_diff_series(diff_series, DIFF_PLOT_PATH)
    perform_adf_test(diff_series, ADF_DIFF_PATH)

    logger.info("=========== END EDA PIPELINE ===========")
    
    logger.info("========== START ACF & PACF ANALYSIS ==========")

    plot_acf_pacf(diff_series)

    ci_95 = write_z_table(diff_series)

    analyze_acf_pacf(diff_series, ci_95)

    logger.info("=========== END ACF & PACF ANALYSIS ===========")


if __name__ == "__main__":
    main()
