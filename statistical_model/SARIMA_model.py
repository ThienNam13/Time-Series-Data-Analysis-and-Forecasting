# statistical_model/SARIMA_model.py

import os
import logging
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX

# =========================================================
# 1. PATH CONFIGURATION
# =========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

LOG_ROOT = os.path.join(
    BASE_DIR,
    "statistical_model",
    "logs",
    "SARIMA_model",
    "AirPassengers"
)

RUN_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(LOG_ROOT, f"run_{RUN_TIME}")
os.makedirs(RUN_DIR, exist_ok=True)

DATA_PATH = os.path.join(DATA_DIR, "airline-passengers.csv")

LOG_FILE = os.path.join(RUN_DIR, "sarima.log")
PLOT_ORIGINAL = os.path.join(RUN_DIR, "series_original.png")
PLOT_DIFF = os.path.join(RUN_DIR, "series_diff_d1.png")
PLOT_SEASONAL_DIFF = os.path.join(RUN_DIR, "series_diff_d1_D1.png")
ACF_PATH = os.path.join(RUN_DIR, "acf.png")
PACF_PATH = os.path.join(RUN_DIR, "pacf.png")
ADF_PATH = os.path.join(RUN_DIR, "adf_results.txt")

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

def load_data():
    """
    Load AirPassengers dataset
    """
    df = pd.read_csv(DATA_PATH)
    df["Month"] = pd.to_datetime(df["Month"])
    df.set_index("Month", inplace=True)
    logger.info("Loaded AirPassengers dataset")
    return df["Passengers"]


def plot_series(series, path, title):
    """
    Vẽ chuỗi thời gian
    (Dùng cho Câu 1 và Câu 3)
    """
    plt.figure(figsize=(10, 4))
    plt.plot(series)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    logger.info(f"Saved plot: {path}")


def adf_test(series, label):
    """
    Kiểm định ADF
    (Dùng cho Câu 2 và kiểm tra lại sau Câu 3)
    """
    result = adfuller(series.dropna())
    with open(ADF_PATH, "a", encoding="utf-8") as f:
        f.write(f"\nADF TEST – {label}\n")
        f.write("=" * 40 + "\n")
        f.write(f"ADF Statistic: {result[0]:.6f}\n")
        f.write(f"p-value: {result[1]:.6f}\n")
    logger.info(f"ADF test completed: {label}")


def seasonal_differencing(series, d=1, D=1, s=12):
    """
    CÂU 3:
    - Sai phân thường (d)
    - Sai phân mùa vụ (D, chu kỳ s)
    """
    diff = series.diff(d)
    seasonal_diff = diff.diff(D * s)
    logger.info("Applied regular differencing (d=1)")
    logger.info("Applied seasonal differencing (D=1, s=12)")
    return diff, seasonal_diff


def plot_acf_pacf(series):
    """
    CÂU 4:
    - Vẽ ACF & PACF
    - Dùng để xác định (p, q) và (P, Q)
    """
    plot_acf(series.dropna(), lags=40)
    plt.tight_layout()
    plt.savefig(ACF_PATH)
    plt.close()

    plot_pacf(series.dropna(), lags=40, method="ywm")
    plt.tight_layout()
    plt.savefig(PACF_PATH)
    plt.close()

    logger.info("Saved ACF & PACF plots")


def fit_sarima(series, order=(1,1,1), seasonal_order=(1,1,1,12)):
    """
    CÂU 5:
    Fit mô hình SARIMA(p,d,q)(P,D,Q,s)
    """
    logger.info(f"Fitting SARIMA: order={order}, seasonal_order={seasonal_order}")
    
    model = SARIMAX(
        series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = model.fit(disp=False)

    summary_path = os.path.join(RUN_DIR, "sarima_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(results.summary().as_text())
        f.write("\n\n")
        f.write(f"AIC: {results.aic:.4f}\n")
        f.write(f"BIC: {results.bic:.4f}\n")

    logger.info(f"SARIMA model fitted. Summary saved: {summary_path}")
    return results


# =========================================================
# 4. MAIN PIPELINE
# =========================================================

def main():
    logger.info("START SARIMA ANALYSIS")

    series = load_data()

    # =====================================================
    # CÂU 1: Vẽ chuỗi thời gian → nhận diện mùa vụ
    # =====================================================
    plot_series(series, PLOT_ORIGINAL, "AirPassengers – Original Series")

    # =====================================================
    # CÂU 2: ADF chuỗi gốc → không dừng
    # =====================================================
    adf_test(series, "Original Series")

    # =====================================================
    # CÂU 3: Sai phân thường + sai phân mùa vụ
    # =====================================================
    diff, seasonal_diff = seasonal_differencing(series)
    plot_series(diff, PLOT_DIFF, "Differenced Series (d=1)")
    plot_series(seasonal_diff, PLOT_SEASONAL_DIFF, "Seasonal Differenced Series (d=1, D=1, s=12)")

    adf_test(seasonal_diff, "After d=1, D=1")

    # =====================================================
    # CÂU 4: ACF / PACF → xác định (p,q,P,Q)
    # =====================================================
    plot_acf_pacf(seasonal_diff)

    # =====================================================
    # CÂU 5: Fit SARIMA
    # =====================================================
    fit_sarima(series, order=(1,1,1), seasonal_order=(1,1,1,12))

    logger.info("END SARIMA ANALYSIS")


if __name__ == "__main__":
    main()
