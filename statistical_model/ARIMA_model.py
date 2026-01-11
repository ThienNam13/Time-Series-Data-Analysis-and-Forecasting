import os
import logging
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np
from math import sqrt

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# =========================================================
#  CẤU HÌNH ĐƯỜNG DẪN & LOGGING
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

# === RUN-SPECIFIC DIRECTORY ===
RUN_DIR = os.path.join(LOG_DIR, f"run_{RUN_TIME}")
os.makedirs(RUN_DIR, exist_ok=True)

# File paths
# =========================================================
# BÀI 1. KHÁM PHÁ DỮ LIỆU (EDA)
# - Load dữ liệu
# - Chuyển date thành index thời gian
# - Vẽ đồ thị chuỗi thời gian
# =========================================================
# Đường dẫn file dữ liệu gốc
DATA_PATH = os.path.join(DATA_DIR, "a10.csv")
# File log ghi lại toàn bộ quá trình chạy
LOG_FILE = os.path.join(RUN_DIR, f"ARIMA_model_{RUN_TIME}.log")
# Lưu đồ thị chuỗi thời gian gốc (EDA)
PLOT_PATH = os.path.join(RUN_DIR, f"time_series_{RUN_TIME}.png")
# Lưu file tổng hợp nhận xét EDA (xu hướng, mùa vụ, tính dừng)
EDA_TEXT_PATH = os.path.join(RUN_DIR, f"eda_summary_{RUN_TIME}.txt")

# thêm path cho 2. Kiểm định tính dừng bằng ADF (chuỗi gốc)
STATIONARITY_PATH = os.path.join(RUN_DIR, f"stationarity_test_{RUN_TIME}.txt")
#3. thêm path cho đồ thị sai phân và kết quả kiểm định ADF sau sai phân
DIFF_PLOT_PATH = os.path.join(RUN_DIR, f"diff_series_{RUN_TIME}.png")
ADF_DIFF_PATH = os.path.join(RUN_DIR, f"stationarity_test_diff_{RUN_TIME}.txt")
# thêm path cho 4. ACF & PACF cho chuỗi đã dừng
ACF_PATH = os.path.join(RUN_DIR, f"acf_{RUN_TIME}.png")
PACF_PATH = os.path.join(RUN_DIR, f"pacf_{RUN_TIME}.png")
ACF_PACF_TXT = os.path.join(RUN_DIR, f"ACF_PACF_Analysis_{RUN_TIME}.txt")
ZTABLE_PATH = os.path.join(RUN_DIR, f"z_table_{RUN_TIME}.txt")
MODEL_COMPARE_TXT = os.path.join(RUN_DIR, f"arima_grid_search_{RUN_TIME}.txt")
MODEL_COMPARE_CSV = os.path.join(RUN_DIR, f"arima_grid_search_{RUN_TIME}.csv")
# thêm path cho 5. xây dựng mô hình ARIMA
MODEL_SUMMARY_PATH = os.path.join(RUN_DIR, f"model_summary_{RUN_TIME}.txt")
FORECAST_PLOT_PATH = os.path.join(RUN_DIR, f"forecast_{RUN_TIME}.png")
FORECAST_VALUE_PATH = os.path.join(RUN_DIR, f"forecast_values_{RUN_TIME}.txt")
RESIDUAL_PLOT_PATH = os.path.join(RUN_DIR, f"residuals_{RUN_TIME}.png")
RESIDUAL_ACF_PATH = os.path.join(RUN_DIR, f"residual_acf_{RUN_TIME}.png")

# =========================================================
# LOGGING CONFIGURATION
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
# BÀI 1. KHÁM PHÁ DỮ LIỆU (EDA)
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
        f.write("EDA SUMMARY - Bai 1 – DATASET a10 (Monthly Retail Sales)\n")
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
# BÀI 2. KIỂM ĐỊNH ADF CHUỖI GỐC
# =========================================================

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

# =========================================================
# BÀI 3. SAI PHÂN & KIỂM ĐỊNH LẠI
# =========================================================

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

# =========================================================
# BÀI 4. PHÂN TÍCH ACF & PACF
# =========================================================

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
    """
    Write full Z-table explanation + confidence thresholds
    following lecturer's reference format.
    """
    n = len(series.dropna())

    # Z-scores (standard normal distribution)
    z_scores = {
        "90%": 1.645,
        "95%": 1.960,
        "99%": 2.576
    }

    # Thresholds for this dataset
    thresholds = {k: v / sqrt(n) for k, v in z_scores.items()}

    with open(ZTABLE_PATH, "w", encoding="utf-8") as f:
        f.write("STANDARD NORMAL DISTRIBUTION (Z-TABLE)\n")
        f.write("Critical Values (Z-Scores) for Confidence Intervals\n")
        f.write("=" * 80 + "\n\n")

        f.write("This table shows z-scores (critical values) from the standard normal distribution.\n")
        f.write("These values are used to calculate confidence intervals for ACF/PACF analysis.\n\n")

        f.write("Formula:\n")
        f.write("CI = ± z_α/2 / √n\n\n")
        f.write("Where:\n")
        f.write("- z_α/2 : critical value from standard normal distribution\n")
        f.write("- n     : sample size\n\n")

        f.write("=" * 80 + "\n")
        f.write("COMMONLY USED CONFIDENCE LEVELS (THEORETICAL Z-SCORES)\n")
        f.write("=" * 80 + "\n")
        f.write("Confidence Level      Z-Score (z_α/2)\n")
        f.write("-" * 50 + "\n")
        for k, v in z_scores.items():
            f.write(f"{k:<20} {v:.4f}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("DATASET-SPECIFIC CONFIDENCE THRESHOLDS (a10 DATASET)\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Sample size (n): {n}\n\n")
        f.write("Calculated thresholds:\n")
        for k, v in thresholds.items():
            f.write(f"{k} confidence interval → ±{v:.6f}\n")

        f.write("\nINTERPRETATION FOR ACF/PACF:\n")
        f.write("- If |ACF| or |PACF| > threshold → statistically significant\n")
        f.write("- If all values lie within thresholds → series resembles white noise\n")

    logger.info(f"Z-table written (full theoretical + applied format) → {ZTABLE_PATH}")

    # Return 95% CI for downstream ACF/PACF analysis
    return thresholds["95%"]

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
# BÀI 5. ARIMA – FORECAST – EVALUATION
# =========================================================

#5.1 Hàm CHIA TRAIN – TEST (80/20)
def split_train_test(series: pd.Series, train_ratio=0.8):
    n = len(series)
    train_size = int(n * train_ratio)

    train = series.iloc[:train_size]
    test = series.iloc[train_size:]

    logger.info(f"Train size: {len(train)}, Test size: {len(test)}")
    return train, test
#5.2 Hàm FIT ARIMA + GHI LOG + SUMMARY
def fit_arima_model(train, order):
    logger.info(f"Fitting ARIMA{order} model...")
    try:
        model = ARIMA(train, order=order)
        model_fit = model.fit()

        logger.info("Model fitted successfully")
        logger.info(f"AIC = {model_fit.aic}, BIC = {model_fit.bic}")

        # Save summary text
        with open(MODEL_SUMMARY_PATH, "w", encoding="utf-8") as f:
            f.write(str(model_fit.summary()))
        
        logger.info(f"Model summary saved: {MODEL_SUMMARY_PATH}")
        return model_fit

    except Exception as e:
        logger.error(f"Failed to fit ARIMA model: {e}")
        raise
# HÀM KHẢO SÁT HỆ SỐ
def arima_grid_search(series, d=1, p_range=range(4), q_range=range(4), train_ratio=0.8):
    """
    Grid search ARIMA(p,d,q) with p,q in given ranges.
    Compare using AIC, BIC, RMSE, MAPE.
    """

    logger.info("START ARIMA GRID SEARCH (p,q ∈ {0..3}, d=1)")

    # Train / Test split
    train, test = split_train_test(series, train_ratio)

    results = []

    for p in p_range:
        for q in q_range:
            try:
                order = (p, d, q)
                logger.info(f"Testing ARIMA{order}")

                model = ARIMA(train, order=order)
                model_fit = model.fit()

                forecast = model_fit.forecast(steps=len(test))

                rmse = np.sqrt(mean_squared_error(test, forecast))
                mae = mean_absolute_error(test, forecast)
                mape = mean_absolute_percentage_error(test, forecast) * 100

                results.append({
                    "p": p,
                    "d": d,
                    "q": q,
                    "AIC": model_fit.aic,
                    "BIC": model_fit.bic,
                    "MAE": mae,
                    "RMSE": rmse,
                    "MAPE": mape
                })

                logger.info(
                    f"ARIMA{order} | AIC={model_fit.aic:.2f}, "
                    f"BIC={model_fit.bic:.2f}, RMSE={rmse:.4f}, MAPE={mape:.2f}%"
                )

            except Exception as e:
                logger.warning(f"ARIMA({p},{d},{q}) failed: {e}")

    # Convert to DataFrame
    df_results = pd.DataFrame(results)

    # Save CSV
    df_results.to_csv(MODEL_COMPARE_CSV, index=False)

    # Save TXT summary
    with open(MODEL_COMPARE_TXT, "w", encoding="utf-8") as f:
        f.write("ARIMA GRID SEARCH RESULTS (p,q ∈ {0..3}, d = 1)\n")
        f.write("=" * 70 + "\n\n")
        f.write(df_results.sort_values("AIC").to_string(index=False))
        f.write("\n\n")
        f.write("Model selection criteria:\n")
        f.write("- Prefer lower AIC/BIC\n")
        f.write("- Prefer lower RMSE/MAPE\n")
        f.write("- d fixed = 1 based on ADF test\n")

    logger.info(f"Grid search results saved:")
    logger.info(f"- CSV: {MODEL_COMPARE_CSV}")
    logger.info(f"- TXT: {MODEL_COMPARE_TXT}")

    logger.info("END ARIMA GRID SEARCH")

    return df_results
#5.3 RESIDUAL DIAGNOSTICS (vẽ residual, ACF residual, kiểm tra mean ~ 0)
def residual_diagnostics(model_fit):
    residuals = model_fit.resid

    # Residual plot
    plt.figure(figsize=(10,4))
    plt.plot(residuals)
    plt.title("Residuals Over Time")
    plt.tight_layout()
    plt.savefig(RESIDUAL_PLOT_PATH)
    plt.close()

    logger.info(f"Residual plot saved → {RESIDUAL_PLOT_PATH}")

    # ACF of residuals
    plt.figure(figsize=(8,4))
    plot_acf(residuals.dropna(), lags=40)
    plt.title("ACF of Residuals")
    plt.tight_layout()
    plt.savefig(RESIDUAL_ACF_PATH)
    plt.close()

    logger.info(f"Residual ACF saved → {RESIDUAL_ACF_PATH}")

    # Mean residual check
    mean_resid = residuals.mean()
    logger.info(f"Residual mean = {mean_resid}")

    if abs(mean_resid) < 0.05:
        logger.info("Residual mean ≈ 0 → GOOD")
    else:
        logger.warning("Residual mean far from 0 → BAD")

#5.4 FORECAST + LƯU TXT + VẼ
def forecast_and_plot(train, test, model_fit):
    forecast = model_fit.forecast(steps=len(test))

    # Save forecast + test values
    with open(FORECAST_VALUE_PATH, "w", encoding="utf-8") as f:
        f.write("FORECAST VALUES vs TEST\n")
        f.write("="*50 + "\n\n")
        for t, p in zip(test.values, forecast.values):
            f.write(f"Actual = {t:.4f}   |   Forecast = {p:.4f}\n")

    logger.info(f"Forecast values saved at {FORECAST_VALUE_PATH}")

    # Plot
    plt.figure(figsize=(10,5))
    plt.plot(train.index, train, label="Train")
    plt.plot(test.index, test, label="Test", color="orange")
    plt.plot(test.index, forecast, label="Forecast", color="green")
    plt.title("ARIMA Forecast vs Actual (Test Set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FORECAST_PLOT_PATH)
    plt.close()

    logger.info(f"Forecast plot saved → {FORECAST_PLOT_PATH}")

    return forecast

# =========================================================
#  MAIN PIPELINE
# =========================================================
def main():
    logger.info(f"Running file: {__file__}")

    # =====================================================
    # BÀI 1 + BÀI 2 + BÀI 3: EDA – ADF – DIFFERENCING
    # =====================================================
    logger.info("========== START EDA PIPELINE ==========")

    # Load & preprocess data
    df = load_data(DATA_PATH)
    df = preprocess_data(df)

    # Bài 1: Vẽ chuỗi thời gian (EDA)
    # Original time series
    plot_time_series(df, PLOT_PATH)
    write_eda_summary(EDA_TEXT_PATH)

    # Bài 2: Kiểm định ADF chuỗi gốc
    perform_adf_test(df["value"], STATIONARITY_PATH)

    # Bài 3: Sai phân bậc 1 + ADF sau sai phân
    # First differencing (d = 1)
    diff_series = difference_series(df)
    plot_diff_series(diff_series, DIFF_PLOT_PATH)
    perform_adf_test(diff_series, ADF_DIFF_PATH)

    logger.info("=========== END EDA PIPELINE ===========")
    # =====================================================
    # BÀI 4: PHÂN TÍCH ACF & PACF – ĐỀ XUẤT (p, d, q)
    # =====================================================
    logger.info("========== START ACF & PACF ANALYSIS ==========")

    plot_acf_pacf(diff_series)

    ci_95 = write_z_table(diff_series)
    analyze_acf_pacf(diff_series, ci_95)

    logger.info("=========== END ACF & PACF ANALYSIS ===========")
    # =========================================================
    # Bài 5. Xây mô hình ARIMA – Forecast – Evaluation
    # =========================================================
    logger.info("========== START ARIMA MODELING ==========")

    # Use original series for modeling (not differenced manually)
    series = df["value"]

    # ---------------------------------------------------------
    # (1) TRAIN – TEST SPLIT (80% / 20%) – KHÔNG SHUFFLE
    # ---------------------------------------------------------
    train, test = split_train_test(series)
    logger.info("Task 5.1: Train/Test split completed (80/20, no shuffle)")
    
    # ---------------------------------------------------------
    # (EXTRA) GRID SEARCH ARIMA(p,d,q)
    # ---------------------------------------------------------
    grid_results = arima_grid_search(
        series=series,
        d=1,
        p_range=range(4),
        q_range=range(4)
    )

    logger.info("ARIMA grid search completed")

    # ---------------------------------------------------------
    # (2) FIT ARIMA MODEL
    # ---------------------------------------------------------
    arima_order = (1, 1, 1)
    model_fit = fit_arima_model(train, arima_order)
    logger.info("Task 5.2: ARIMA model fitted")

    # ---------------------------------------------------------
    # (4) FORECASTING ON TEST SET
    # ---------------------------------------------------------
    forecast = forecast_and_plot(train, test, model_fit)
    logger.info("Task 5.4: Forecasting completed on test set")

    # ---------------------------------------------------------
    # (5.5) MODEL EVALUATION: MAE, RMSE, MAPE
    # ---------------------------------------------------------
    mae = mean_absolute_error(test, forecast)
    rmse = np.sqrt(mean_squared_error(test, forecast))
    mape = mean_absolute_percentage_error(test, forecast) * 100

    logger.info("Task 5.5: Evaluation metrics calculated")
    logger.info(f"MAE  = {mae}")
    logger.info(f"RMSE = {rmse}")
    logger.info(f"MAPE = {mape:.2f}%")

    # ---------------------------------------------------------
    # RESIDUAL DIAGNOSTICS
    # ---------------------------------------------------------
    residual_diagnostics(model_fit)

    logger.info("=========== END ARIMA MODELING ===========")


if __name__ == "__main__":
    main()
