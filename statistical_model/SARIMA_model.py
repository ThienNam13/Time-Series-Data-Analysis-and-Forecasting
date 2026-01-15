# statistical_model/SARIMA_model.py

import os
import logging
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
import shutil
import csv

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

# Additional target: copy run outputs into existing ARIMA_model/a10 logs folder
EXTRA_LOG_ROOT = os.path.join(
    BASE_DIR,
    "statistical_model",
    "logs",
    "ARIMA_model",
    "a10"
)

# Flag để bật/tắt việc copy run outputs vào thư mục logs khác
ENABLE_COPY_TO_EXTRA = False

DATA_PATH = os.path.join(DATA_DIR, "airline-passengers.csv")

LOG_FILE = os.path.join(RUN_DIR, "sarima.log")
PLOT_ORIGINAL = os.path.join(RUN_DIR, "series_original.png")
PLOT_DIFF = os.path.join(RUN_DIR, "series_diff_d1.png")
PLOT_SEASONAL_DIFF = os.path.join(RUN_DIR, "series_diff_d1_D1.png")
ACF_PATH = os.path.join(RUN_DIR, "acf.png")
PACF_PATH = os.path.join(RUN_DIR, "pacf.png")
ADF_PATH = os.path.join(RUN_DIR, "adf_results.txt")
PLOT_FORECAST = os.path.join(RUN_DIR, "train_test_forecast.png")
METRICS_PATH = os.path.join(RUN_DIR, "sarima_metrics.txt")

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


def forecast_and_evaluate(series, order=(1,1,1), seasonal_order=(1,1,1,12), test_size=12):
    """
    Split into train/test (last `test_size` points), fit SARIMA on train,
    forecast the test horizon, compute MAE, RMSE, MAPE, and save a plot.
    """
    logger.info("Starting train/test split and forecasting")

    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]

    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = model.fit(disp=False)

    # Forecast
    pred = results.get_forecast(steps=test_size)
    forecast = pred.predicted_mean

    # Align indexes if needed
    forecast.index = test.index

    # Metrics
    mae = np.mean(np.abs(forecast - test))
    rmse = np.sqrt(np.mean((forecast - test) ** 2))
    # avoid division by zero for MAPE
    mask = test != 0
    if mask.sum() == 0:
        mape = np.nan
    else:
        mape = np.mean(np.abs((forecast[mask] - test[mask]) / test[mask])) * 100

    # Save metrics
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        f.write("SARIMA Forecast Metrics\n")
        f.write("=" * 40 + "\n")
        f.write(f"MAE: {mae:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"MAPE: {mape if np.isnan(mape) else f'{mape:.2f}%'}\n")

    logger.info(f"Saved metrics: {METRICS_PATH}")

    # Also write metrics and forecast values to the main run log
    logger.info(f"Forecast metrics — MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {('N/A' if np.isnan(mape) else f'{mape:.2f}%')}")
    try:
        logger.info("Forecast values:\n" + forecast.to_string())
    except Exception:
        logger.info("Forecast values: " + str(forecast.values))

    # Plot train, test, forecast
    plt.figure(figsize=(10, 5))
    plt.plot(train.index, train, label="Train")
    plt.plot(test.index, test, label="Test")
    plt.plot(forecast.index, forecast, label="SARIMA Forecast", linestyle="--")
    plt.legend()
    plt.title("Train / Test / SARIMA Forecast")
    plt.tight_layout()
    plt.savefig(PLOT_FORECAST)
    plt.close()
    logger.info(f"Saved forecast plot: {PLOT_FORECAST}")

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": (None if np.isnan(mape) else float(mape)),
        "forecast_path": PLOT_FORECAST,
        "metrics_path": METRICS_PATH,
    }


def copy_run_to_extra_logs():
    """Copy the entire RUN_DIR into EXTRA_LOG_ROOT/run_<TIMESTAMP> if EXTRA_LOG_ROOT exists."""
    if not ENABLE_COPY_TO_EXTRA:
        logger.info("Chức năng copy vào EXTRA_LOG_ROOT đang tắt. Bỏ qua.")
        return
    try:
        if os.path.exists(EXTRA_LOG_ROOT):
            dest = os.path.join(EXTRA_LOG_ROOT, os.path.basename(RUN_DIR))
            # copytree with dirs_exist_ok available in Python 3.8+
            shutil.copytree(RUN_DIR, dest, dirs_exist_ok=True)
            logger.info(f"Copied run outputs to extra log folder: {dest}")
        else:
            logger.info(f"Extra log root does not exist, skipping copy: {EXTRA_LOG_ROOT}")
    except Exception as e:
        logger.error(f"Failed to copy run outputs to extra log folder: {e}")


def compare_models(series, model_specs=None, test_size=12):
    """
    So sánh nhiều cấu hình ARIMA/SARIMA.
    - `model_specs`: list of tuples (order, seasonal_order, name)
    - Trả về list rows và lưu file CSV.
    """
    if model_specs is None:
        model_specs = [
            ((1, 1, 1), (0, 0, 0, 0), "ARIMA(1,1,1)"),
            ((2, 1, 1), (0, 0, 0, 0), "ARIMA(2,1,1)"),
            ((1, 1, 1), (1, 1, 1, 12), "SARIMA(1,1,1)(1,1,1,12)"),
        ]

    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]

    rows = []
    vi_lines = []
    vi_lines.append("BẢNG SO SÁNH MÔ HÌNH SARIMA/ARIMA")
    vi_lines.append("=" * 60)
    vi_lines.append(f"Dữ liệu: train={train.index[0].date()}..{train.index[-1].date()}, test={test.index[0].date()}..{test.index[-1].date()}")

    for order, seasonal_order, name in model_specs:
        logger.info(f"Fitting for comparison: {name} order={order} seasonal_order={seasonal_order}")
        try:
            model = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                            enforce_stationarity=False, enforce_invertibility=False)
            res = model.fit(disp=False)

            pred = res.get_forecast(steps=test_size)
            fc = pred.predicted_mean
            fc.index = test.index

            mae = np.mean(np.abs(fc - test))
            rmse = np.sqrt(np.mean((fc - test) ** 2))
            mask = test != 0
            mape = np.nan if mask.sum() == 0 else np.mean(np.abs((fc[mask] - test[mask]) / test[mask])) * 100

            row = {
                "name": name,
                "order": str(order),
                "seasonal_order": str(seasonal_order),
                "aic": float(res.aic),
                "bic": float(res.bic),
                "mae": float(mae),
                "rmse": float(rmse),
                "mape": (None if np.isnan(mape) else float(mape)),
            }
            rows.append(row)

            # Vietnamese log lines
            vi_lines.append(f"Mô hình: {name} | order={order} seasonal_order={seasonal_order}")
            vi_lines.append(f"  AIC: {res.aic:.4f} | BIC: {res.bic:.4f}")
            vi_lines.append(f"  MAE: {mae:.4f} | RMSE: {rmse:.4f} | MAPE: {('N/A' if np.isnan(mape) else f'{mape:.2f}%')}")
            vi_lines.append("-")

        except Exception as e:
            logger.error(f"Error fitting {name}: {e}")
            vi_lines.append(f"Mô hình: {name} — Lỗi khi fitting: {e}")
            vi_lines.append("-")

    # Save CSV
    csv_path = os.path.join(RUN_DIR, "model_comparison.csv")
    with open(csv_path, "w", newline='', encoding='utf-8') as csvfile:
        fieldnames = ["name", "order", "seasonal_order", "aic", "bic", "mae", "rmse", "mape"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Save Vietnamese log
    vi_log_path = os.path.join(RUN_DIR, "so_sanh_mo_hinh_vi.txt")
    with open(vi_log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(vi_lines))

    logger.info(f"Saved model comparison CSV: {csv_path}")
    logger.info(f"Saved Vietnamese comparison log: {vi_log_path}")

    # Also copy to extra logs if present
    try:
        if ENABLE_COPY_TO_EXTRA and os.path.exists(EXTRA_LOG_ROOT):
            dest_csv = os.path.join(EXTRA_LOG_ROOT, os.path.basename(RUN_DIR), os.path.basename(csv_path))
            dest_vi = os.path.join(EXTRA_LOG_ROOT, os.path.basename(RUN_DIR), os.path.basename(vi_log_path))
            # ensure destination run folder exists
            os.makedirs(os.path.dirname(dest_csv), exist_ok=True)
            shutil.copy2(csv_path, dest_csv)
            shutil.copy2(vi_log_path, dest_vi)
            logger.info(f"Copied comparison files to extra log folder: {os.path.dirname(dest_csv)}")
        else:
            logger.info("Chức năng copy comparison files đang tắt hoặc EXTRA_LOG_ROOT không tồn tại. Bỏ qua.")
    except Exception as e:
        logger.error(f"Failed to copy comparison files: {e}")

    return rows, csv_path, vi_log_path


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
    # Fit on full series (optional) and save summary
    fit_sarima(series, order=(1,1,1), seasonal_order=(1,1,1,12))

    # Forecast on test set (last 12 points) and evaluate
    metrics = forecast_and_evaluate(series, order=(1,1,1), seasonal_order=(1,1,1,12), test_size=12)
    logger.info(f"Forecast metrics: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, MAPE={metrics['mape']}")

    # So sánh một số cấu hình mô hình và lưu bảng kết quả (CSV)
    specs = [
        ((1, 1, 1), (0, 0, 0, 0), "ARIMA(1,1,1)"),
        ((2, 1, 1), (0, 0, 0, 0), "ARIMA(2,1,1)"),
        ((1, 1, 1), (1, 1, 1, 12), "SARIMA(1,1,1)(1,1,1,12)"),
    ]
    rows, csv_path, vi_log = compare_models(series, model_specs=specs, test_size=12)

    # Copy run outputs to preferred logs folder (if exists)
    # (được vô hiệu hóa theo mặc định bởi ENABLE_COPY_TO_EXTRA)
    copy_run_to_extra_logs()

    logger.info("END SARIMA ANALYSIS")


if __name__ == "__main__":
    main()
