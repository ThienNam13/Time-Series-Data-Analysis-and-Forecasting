# evaluation/backtesting.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping


# =====================================================
# METRICS
# =====================================================
def evaluate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    return rmse, mae, mape


# =====================================================
# SAVE RESULTS
# =====================================================
def save_results(y_true, y_pred, model_name, log_dir):

    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred
    })

    df.to_csv(
        os.path.join(log_dir, f"walk_forward_{model_name}.csv"),
        index=False
    )

    plt.figure(figsize=(12, 5))
    plt.plot(y_true, label="Actual")
    plt.plot(y_pred, label="Forecast")
    plt.legend()
    plt.title(f"Walk-forward Forecast - {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f"walk_forward_{model_name}.png"))
    plt.close()


# =====================================================
# INVERSE LOAD ONLY
# =====================================================
def inverse_scale_load_only(scaler, load_scaled):
    """
    Inverse only load using scaler fitted on [load, temp, hum, wind]
    """
    dummy = np.zeros((len(load_scaled), scaler.mean_.shape[0]))
    dummy[:, 0] = load_scaled
    inv = scaler.inverse_transform(dummy)
    return inv[:, 0]


# =====================================================
# WALK-FORWARD VAR
# =====================================================
def walk_forward_var(train_df, test_df, scaler, max_lag=24, log_dir=None):

    history = train_df.copy()
    preds_scaled = []

    n_features = train_df.shape[1]

    model = VAR(train_df)
    lag_order = model.select_order(max_lag).bic

    for t in range(len(test_df)):

        fitted = VAR(history).fit(lag_order)

        forecast = fitted.forecast(
            y=history.values[-lag_order:],
            steps=1
        )

        yhat_scaled = forecast[0, history.columns.get_loc("load")]
        preds_scaled.append(yhat_scaled)

        history = pd.concat([history, test_df.iloc[t:t+1]])

        if t % 50 == 0:
            print(f"VAR walk-forward step {t}/{len(test_df)}")

    y_true_inv = inverse_scale_load_only(scaler, test_df["load"].values)
    y_pred_inv = inverse_scale_load_only(scaler, np.array(preds_scaled))

    if log_dir:
        save_results(y_true_inv, y_pred_inv, "VAR", log_dir)

    return y_true_inv, y_pred_inv


# =====================================================
# WALK-FORWARD XGBOOST
# =====================================================
def walk_forward_xgboost(
    train_df,
    test_df,
    scaler,
    params,
    num_boost_round=100,
    log_dir=None
):

    history = train_df.copy()
    preds_scaled = []

    n_features = train_df.shape[1]

    for t in range(len(test_df)):

        X_train = history.drop(columns=["load"])
        y_train = history["load"]

        dtrain = xgb.DMatrix(X_train, label=y_train)

        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            verbose_eval=False
        )

        X_test = test_df.iloc[t:t+1].drop(columns=["load"])
        dtest = xgb.DMatrix(X_test)

        yhat_scaled = model.predict(dtest)[0]
        preds_scaled.append(yhat_scaled)

        history = pd.concat([history, test_df.iloc[t:t+1]])

        if t % 50 == 0:
            print(f"XGBoost walk-forward step {t}/{len(test_df)}")

    y_true_inv = inverse_scale_load_only(scaler, test_df["load"].values)
    y_pred_inv = inverse_scale_load_only(scaler, np.array(preds_scaled))

    if log_dir:
        save_results(y_true_inv, y_pred_inv, "XGBoost", log_dir)

    return y_true_inv, y_pred_inv


# =====================================================
# WALK-FORWARD LSTM
# =====================================================
def build_lstm(input_shape):
    model = Sequential([
        LSTM(64, input_shape=input_shape),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def walk_forward_lstm(
    train_df,
    test_df,
    scaler,
    window_size=24,
    epochs=10,
    batch_size=64,
    log_dir=None
):

    history = train_df.copy()
    preds_scaled = []

    n_features = train_df.shape[1]

    for t in range(len(test_df)):

        values = history.values

        X_train, y_train = [], []
        for i in range(window_size, len(values)):
            X_train.append(values[i - window_size:i])
            y_train.append(values[i, 0])  # load

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        model = build_lstm((window_size, n_features))

        early = EarlyStopping(patience=3, restore_best_weights=True)

        model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=[early]
        )

        last_window = history.values[-window_size:]
        X_test = last_window.reshape(1, window_size, n_features)

        yhat_scaled = model.predict(X_test, verbose=0)[0, 0]
        preds_scaled.append(yhat_scaled)

        history = pd.concat([history, test_df.iloc[t:t+1]])

        if t % 20 == 0:
            print(f"LSTM walk-forward step {t}/{len(test_df)}")

    y_true_inv = inverse_scale_load_only(scaler, test_df["load"].values)
    y_pred_inv = inverse_scale_load_only(scaler, np.array(preds_scaled))

    if log_dir:
        save_results(y_true_inv, y_pred_inv, "LSTM", log_dir)

    return y_true_inv, y_pred_inv
