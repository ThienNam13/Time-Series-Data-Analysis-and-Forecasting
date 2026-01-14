import os
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error

from multivariate_forecasting.data_loader import (
    load_and_prepare_multivariate_data
)

# =========================================================
# PATHS
# =========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR = os.path.join(BASE_DIR, "multivariate_forecasting", "logs", "VAR")

os.makedirs(LOG_DIR, exist_ok=True)

RUN_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(LOG_DIR, f"run_{RUN_TIME}")
os.makedirs(RUN_DIR, exist_ok=True)

ADF_TXT = os.path.join(RUN_DIR, "adf_test.txt")
ORDER_TXT = os.path.join(RUN_DIR, "lag_selection.txt")
EVAL_TXT = os.path.join(RUN_DIR, "evaluation.txt")

# =========================================================
# LOGGING
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(RUN_DIR, "var.log")),
        logging.StreamHandler()
    ],
    force=True
)
logger = logging.getLogger(__name__)

# =========================================================
# FUNCTIONS
# =========================================================

def adf_test_all(df: pd.DataFrame):
    """
    ADF test cho tất cả biến
    """
    with open(ADF_TXT, "w", encoding="utf-8") as f:
        f.write("ADF TEST RESULTS\n")
        f.write("=" * 50 + "\n")

        for col in df.columns:
            result = adfuller(df[col].dropna())
            p_value = result[1]

            f.write(f"\nVariable: {col}\n")
            f.write(f"ADF Statistic: {result[0]:.4f}\n")
            f.write(f"p-value: {p_value:.4f}\n")

            if p_value < 0.05:
                f.write("=> Stationary\n")
            else:
                f.write("=> Non-stationary\n")

    logger.info("ADF test completed")


def make_stationary(df: pd.DataFrame):
    """
    Sai phân nếu chuỗi không dừng
    """
    df_diff = df.diff().dropna()
    logger.info("Applied first-order differencing to all variables")
    return df_diff


def select_var_lag(train_df, max_lag=24):
    """
    Chọn bậc trễ bằng AIC
    """
    model = VAR(train_df)
    order_results = model.select_order(maxlags=max_lag)

    selected_lag = order_results.aic

    with open(ORDER_TXT, "w", encoding="utf-8") as f:
        f.write("VAR LAG SELECTION\n")
        f.write("=" * 50 + "\n")
        f.write(str(order_results.summary()))
        f.write(f"\n\nSelected lag (AIC): {selected_lag}\n")

    logger.info(f"Selected VAR lag (AIC): {selected_lag}")
    return selected_lag


# =========================================================
# MAIN
# =========================================================

def main():
    logger.info("========== START VAR MODEL ==========")

    # Load multivariate data
    df = load_and_prepare_multivariate_data(
        load_path=os.path.join(DATA_DIR, "LD2011_2014.txt"),
        weather_path=os.path.join(DATA_DIR, "weather.csv"),
        freq="H"
    )

    df = df[["load", "temperature", "humidity", "wind_speed"]]

    # =====================================================
    # 1. ADF TEST
    # =====================================================
    adf_test_all(df)

    # =====================================================
    # 2. DIFFERENCING
    # =====================================================
    df_diff = make_stationary(df)

    # =====================================================
    # 3. TRAIN / TEST SPLIT
    # =====================================================
    test_size = 24
    train = df_diff.iloc[:-test_size]
    test = df_diff.iloc[-test_size:]

    # =====================================================
    # 4. LAG SELECTION
    # =====================================================
    p = select_var_lag(train, max_lag=24)

    # =====================================================
    # 5. FIT VAR
    # =====================================================
    model = VAR(train)
    results = model.fit(p)

    # =====================================================
    # 6. FORECAST VECTOR
    # =====================================================
    forecast = results.forecast(train.values[-p:], steps=test_size)
    forecast_df = pd.DataFrame(
        forecast,
        index=test.index,
        columns=train.columns
    )

    # =====================================================
    # 7. EVALUATION (LOAD ONLY)
    # =====================================================
    y_true = test["load"]
    y_pred = forecast_df["load"]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    with open(EVAL_TXT, "w", encoding="utf-8") as f:
        f.write("VAR MODEL EVALUATION (LOAD ONLY)\n")
        f.write("=" * 50 + "\n")
        f.write(f"MAE : {mae:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")

    logger.info(f"VAR evaluation completed — MAE={mae:.4f}, RMSE={rmse:.4f}")
    logger.info("=========== END VAR MODEL ============")


if __name__ == "__main__":
    main()