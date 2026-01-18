# models/var_model.py

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


# =====================================================
# ADF TEST
# =====================================================
def adf_test(series, name="", verbose=True):
    result = adfuller(series.dropna())
    p_value = result[1]

    if verbose:
        print(f"ADF Test - {name}: p-value = {p_value:.5f}")

    return p_value


# =====================================================
# STATIONARITY CHECK & DIFFERENCING
# =====================================================
def make_stationary(df, significance=0.05):
    """
    Check ADF for each column.
    If any non-stationary → apply 1st order differencing to all.
    """

    need_diff = False

    for col in df.columns:
        p = adf_test(df[col], name=col, verbose=False)
        if p > significance:
            need_diff = True
            break

    if need_diff:
        df_diff = df.diff().dropna()
        return df_diff, True
    else:
        return df.copy(), False


# =====================================================
# TRAIN VAR MODEL (ON SCALED DATA)
# =====================================================
def train_var_model(train_df, max_lag=24, verbose=True):
    """
    Train VAR model on scaled & stationary data
    """

    model = VAR(train_df)

    lag_results = model.select_order(max_lag)
    selected_lag = lag_results.bic

    if verbose:
        print("VAR Lag Selection:")
        print(lag_results.summary())
        print(f"Selected lag (BIC): {selected_lag}")

    fitted_model = model.fit(selected_lag)

    return fitted_model, selected_lag


# =====================================================
# FORECAST VAR (ON SCALED SPACE)
# =====================================================
def forecast_var(model, train_df, steps, lag_order):
    """
    Forecast future values in scaled space
    """

    forecast = model.forecast(
        y=train_df.values[-lag_order:],
        steps=steps
    )

    forecast_df = pd.DataFrame(
        forecast,
        columns=train_df.columns
    )

    return forecast_df


# =====================================================
# INVERSE LOAD ONLY
# =====================================================
def inverse_scale_load(scaler, load_scaled, n_features):
    """
    Inverse transform only 'load' column (assumed index 0)
    """

    dummy = np.zeros((len(load_scaled), n_features))
    dummy[:, 0] = load_scaled

    inv = scaler.inverse_transform(dummy)
    return inv[:, 0]


# =====================================================
# EVALUATION (ON ORIGINAL SCALE)
# =====================================================
def evaluate_forecast(
    y_true_original,
    y_pred_original,
    model_name="VAR",
    log_file=None
):

    rmse = np.sqrt(mean_squared_error(y_true_original, y_pred_original))
    mae = mean_absolute_error(y_true_original, y_pred_original)
    mape = mean_absolute_percentage_error(y_true_original, y_pred_original) * 100

    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n=== {model_name} EVALUATION ===\n")
            f.write(f"RMSE: {rmse:.4f}\n")
            f.write(f"MAE : {mae:.4f}\n")
            f.write(f"MAPE: {mape:.2f}%\n")

    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape
    }
