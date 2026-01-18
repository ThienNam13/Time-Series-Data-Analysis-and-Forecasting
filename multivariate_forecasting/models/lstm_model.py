# models/lstm_model.py

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


# =====================================================
# DATASET CREATION
# =====================================================
def create_lstm_dataset(data, target_col="load", window_size=24):
    """
    data: scaled dataframe
    """
    values = data.values
    target_idx = data.columns.get_loc(target_col)

    X, y = [], []
    for i in range(window_size, len(values)):
        X.append(values[i - window_size:i, :])
        y.append(values[i, target_idx])

    return np.array(X), np.array(y)


# =====================================================
# INVERSE LOAD ONLY
# =====================================================
def inverse_scale_load(scaler, load_scaled, n_features):
    dummy = np.zeros((len(load_scaled), n_features))
    dummy[:, 0] = load_scaled
    inv = scaler.inverse_transform(dummy)
    return inv[:, 0]


# =====================================================
# TRAIN + EVALUATE LSTM (STATIC TEST SET)
# =====================================================
def train_lstm(
    train_df,
    val_df,
    test_df,
    scaler,
    log_dir,
    window_size=24,
    batch_size=64,
    epochs=50
):
    """
    All data must be already scaled
    Metrics are computed on ORIGINAL scale
    """

    # ===== CREATE DATASETS =====
    X_train, y_train = create_lstm_dataset(train_df, window_size=window_size)
    X_val, y_val = create_lstm_dataset(val_df, window_size=window_size)
    X_test, y_test = create_lstm_dataset(test_df, window_size=window_size)

    # ===== MODEL =====
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(window_size, train_df.shape[1])),
        LSTM(32),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")

    # ===== CALLBACK =====
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    # ===== TRAIN =====
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )

    # ===== PREDICT (SCALED) =====
    y_pred_scaled = model.predict(X_test).flatten()

    # ===== INVERSE SCALE =====
    n_features = train_df.shape[1]
    y_test_inv = inverse_scale_load(scaler, y_test, n_features)
    y_pred_inv = inverse_scale_load(scaler, y_pred_scaled, n_features)

    # ===== METRICS (ORIGINAL) =====
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    mape = mean_absolute_percentage_error(y_test_inv, y_pred_inv) * 100

    # ===== SAVE MODEL =====
    model_path = os.path.join(log_dir, "lstm_model.h5")
    model.save(model_path)

    # ===== SAVE RESULTS =====
    result_path = os.path.join(log_dir, "lstm_results.txt")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write("=== LSTM RESULTS (ORIGINAL SCALE) ===\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"MAE : {mae:.4f}\n")
        f.write(f"MAPE: {mape:.2f}%\n")

    return y_pred_inv, rmse, mae, mape
