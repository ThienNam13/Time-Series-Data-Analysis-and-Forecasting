import pandas as pd
import numpy as np
import os
import logging

from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ===================== CONFIG =====================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")

DATA_FILE = os.path.join(LOG_DIR, "feature_engineered_data.csv")
RESULT_FILE = os.path.join(LOG_DIR, "var_forecast.csv")

TARGET_COL = "load"
VAR_COLS = ["load", "temperature", "humidity", "wind_speed"]

TRAIN_RATIO = 0.8
MAX_LAG = 24

# ===================== LOGGING =====================
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "var_model.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ===================== FUNCTIONS =====================
def adf_test(series, name):
    result = adfuller(series, autolag="AIC")
    p_value = result[1]
    logging.info(f"ADF Test - {name}: p-value = {p_value:.4f}")
    return p_value < 0.05


def make_stationary(df):
    stationary = True
    df_diff = df.copy()

    for col in df.columns:
        if not adf_test(df[col], col):
            df_diff[col] = df[col].diff()
            stationary = False
            logging.info(f"Differencing applied to {col}")

    df_diff.dropna(inplace=True)
    return df_diff, stationary


def train_test_split(df, ratio):
    split = int(len(df) * ratio)
    return df.iloc[:split], df.iloc[split:]


# ===================== MAIN =====================
def main():
    logging.info("START VAR MODEL")

    # Load data (chỉ dùng biến gốc)
    df = pd.read_csv(DATA_FILE, index_col=0)
    df = df[VAR_COLS]

    logging.info(f"Original data shape: {df.shape}")

    # Stationarity check
    df_stationary, is_stationary = make_stationary(df)

    if is_stationary:
        logging.info("All series are stationary")
    else:
        logging.info("Non-stationary series detected → differencing applied")

    # Train / Test split
    train_df, test_df = train_test_split(df_stationary, TRAIN_RATIO)

    #  Lag order selection
    model = VAR(train_df)
    lag_selection = model.select_order(MAX_LAG)

    selected_lag = lag_selection.aic
    logging.info(f"Selected lag (AIC): {selected_lag}")

    #  Train VAR
    var_model = model.fit(selected_lag)
    logging.info("VAR model fitted")

    # Forecast
    steps = len(test_df)
    forecast = var_model.forecast(train_df.values, steps=steps)

    forecast_df = pd.DataFrame(
        forecast,
        columns=VAR_COLS,
        index=test_df.index
    )

    # Evaluation (ONLY load)
    y_true = test_df[TARGET_COL]
    y_pred = forecast_df[TARGET_COL]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    logging.info(f"VAR Load MAE: {mae:.4f}")
    logging.info(f"VAR Load RMSE: {rmse:.4f}")

    # Save forecast
    forecast_df.to_csv(RESULT_FILE)
    logging.info(f"Forecast saved to {RESULT_FILE}")
    logging.info("VAR MODEL COMPLETED")


if __name__ == "__main__":
        main()