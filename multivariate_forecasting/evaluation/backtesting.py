import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

from models.lstm_model import build_lstm, create_sequences
import tensorflow as tf


# =========================================================
# METRICS
# =========================================================
def evaluate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    return rmse, mae, mape


def save_results(y_true, y_pred, model_name, log_dir):
    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred
    })

    csv_path = os.path.join(log_dir, f"walk_forward_{model_name}.csv")
    df.to_csv(csv_path, index=False)

    plt.figure(figsize=(12, 5))
    plt.plot(y_true, label="Actual")
    plt.plot(y_pred, label="Forecast")
    plt.legend()
    plt.title(f"Walk-forward Forecast - {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f"walk_forward_{model_name}.png"))
    plt.close()

    return csv_path


# =========================================================
# WALK-FORWARD FOR VAR
# =========================================================
def walk_forward_var(train_df, test_df, max_lag=24, log_dir=None):

    history = train_df.copy()
    predictions = []

    model = VAR(train_df)
    lag_order = model.select_order(max_lag).bic

    for t in range(len(test_df)):
        fitted = VAR(history).fit(lag_order)

        forecast = fitted.forecast(
            y=history.values[-lag_order:],
            steps=1
        )

        yhat = forecast[0, history.columns.get_loc("load")]
        predictions.append(yhat)

        history = pd.concat([history, test_df.iloc[t:t+1]])

        if t % 50 == 0:
            print(f"VAR walk-forward step {t}/{len(test_df)}")

    y_true = test_df["load"].values
    y_pred = np.array(predictions)

    if log_dir:
        save_results(y_true, y_pred, "VAR", log_dir)

    return y_true, y_pred


# =========================================================
# WALK-FORWARD FOR XGBOOST
# =========================================================
def walk_forward_xgboost(train_df, test_df, params, num_boost_round=300, log_dir=None):

    history = train_df.copy()
    predictions = []

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

        yhat = model.predict(dtest)[0]
        predictions.append(yhat)

        history = pd.concat([history, test_df.iloc[t:t+1]])
        
        if t % 50 == 0:
            print(f"XGBoost walk-forward step {t}/{len(test_df)}")

    y_true = test_df["load"].values
    y_pred = np.array(predictions)

    if log_dir:
        save_results(y_true, y_pred, "XGBoost", log_dir)

    return y_true, y_pred


# =========================================================
# WALK-FORWARD FOR LSTM
# =========================================================
def walk_forward_lstm(train_data, test_data, window_size=24, epochs=3):
    history = train_data.copy().values
    preds = []

    for i in range(len(test_data)):
        X_train, y_train = create_sequences(history, window_size)

        X_train = X_train.reshape(
            (X_train.shape[0], X_train.shape[1], X_train.shape[2])
        )

        model = build_lstm((window_size, history.shape[1]))
        model.fit(X_train, y_train, epochs=epochs, verbose=0)

        last_seq = history[-window_size:]
        last_seq = last_seq.reshape((1, window_size, history.shape[1]))

        yhat = model.predict(last_seq, verbose=0)
        preds.append(yhat[0, 0])

        # add true observation
        history = np.vstack([history, test_data.iloc[i].values])

        if i % 10 == 0:
            print(f"LSTM walk-forward step {i}/{len(test_data)}")

    y_true = test_data["load"].values
    return y_true, np.array(preds)

def inverse_load_only(preds, scaler, n_features):
    dummy = np.zeros((len(preds), n_features))
    dummy[:, 0] = preds
    inv = scaler.inverse_transform(dummy)
    return inv[:, 0]
