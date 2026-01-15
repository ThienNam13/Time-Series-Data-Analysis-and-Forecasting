import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller


def adf_test(series, name="", log_file=None):
    result = adfuller(series.dropna())
    p_value = result[1]

    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"ADF Test - {name}: p-value = {p_value:.5f}\n")

    return p_value


def make_stationary(df, log_file=None):
    df_diff = df.copy()
    need_diff = False

    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write("\n=== ADF TEST ===\n")

    for col in df.columns:
        p_value = adf_test(df[col], col, log_file)
        if p_value > 0.05:
            need_diff = True

    if need_diff:
        if log_file:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write("\nChuỗi không dừng → Thực hiện differencing bậc 1\n")
        df_diff = df.diff().dropna()
    else:
        if log_file:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write("\nTất cả chuỗi đều dừng\n")

    return df_diff


def train_var_model(
    train_df,
    max_lag=24,
    criterion="bic",
    log_file=None
):
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write("\n=== VAR LAG SELECTION ===\n")

    model = VAR(train_df)
    lag_order_results = model.select_order(max_lag)

    selected_lag = getattr(lag_order_results, criterion)

    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"Chọn lag = {selected_lag} theo {criterion.upper()}\n")

    var_model = model.fit(selected_lag)

    return var_model


def forecast_var(model, train_df, steps):
    forecast = model.forecast(
        y=train_df.values[-model.k_ar:],
        steps=steps
    )

    forecast_df = pd.DataFrame(
        forecast,
        columns=train_df.columns
    )

    return forecast_df
