import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


def create_lstm_dataset(data, target_col="load", window_size=24):
    """
    Convert dataframe to LSTM 3D tensor
    X shape: (samples, window_size, n_features)
    y shape: (samples,)
    """
    values = data.values
    target_idx = data.columns.get_loc(target_col)

    X, y = [], []
    for i in range(window_size, len(values)):
        X.append(values[i - window_size:i, :])
        y.append(values[i, target_idx])

    return np.array(X), np.array(y)


def train_lstm(
    train_df,
    val_df,
    test_df,
    log_dir,
    window_size=24,
    batch_size=64,
    epochs=50
):
    """
    Train LSTM model for load forecasting
    """

    # ===== CREATE DATASETS =====
    X_train, y_train = create_lstm_dataset(train_df, window_size=window_size)
    X_val, y_val = create_lstm_dataset(val_df, window_size=window_size)
    X_test, y_test = create_lstm_dataset(test_df, window_size=window_size)

    # ===== MODEL ARCHITECTURE =====
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(window_size, train_df.shape[1])),
        LSTM(32),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    # ===== CALLBACK =====
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    # ===== TRAIN =====
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )

    # ===== PREDICT =====
    y_pred = model.predict(X_test).flatten()

    # ===== METRICS =====
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    # ===== SAVE MODEL =====
    model_path = os.path.join(log_dir, "lstm_model.h5")
    model.save(model_path)

    # ===== SAVE RESULTS =====
    result_path = os.path.join(log_dir, "lstm_results.txt")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write("=== LSTM RESULTS ===\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"MAE: {mae:.4f}\n")
        f.write(f"MAPE: {mape*100:.2f}%\n")

    return y_pred, rmse, mae, mape
