import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # Ẩn INFO + WARNING của TF

import h5py
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
import logging

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# ===============================
# 0. LOGGING CONFIG (QUAN TRỌNG)
# ===============================
LOG_DIR = "logs_metr_la"
os.makedirs(LOG_DIR, exist_ok=True)

run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = f"{LOG_DIR}/train_log_{run_time}.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logging.info("===== START TRAINING METR-LA =====")

# ===============================
# 1. LOAD DATA
# ===============================
DATA_PATH = r"deep_learning_model\data\METR-LA.h5"

df = pd.read_hdf(DATA_PATH)

logging.info(f"DataFrame shape: {df.shape}")
logging.info(f"Time range: {df.index.min()} → {df.index.max()}")

# ===============================
# 2. PREPROCESSING
# ===============================
data = df.ffill().values

scaler = StandardScaler()
data = scaler.fit_transform(data)

logging.info("Data normalized (StandardScaler)")

# ===============================
# 3. CREATE SLIDING WINDOW
# ===============================
INPUT_LEN = 12     # 60 phút
OUTPUT_LEN = 12    # 60 phút
NUM_NODES = data.shape[1]

def create_dataset(data, input_len, output_len):
    X, y = [], []
    for i in range(len(data) - input_len - output_len):
        X.append(data[i:i+input_len])
        y.append(data[i+input_len:i+input_len+output_len])
    return np.array(X), np.array(y)

X, y = create_dataset(data, INPUT_LEN, OUTPUT_LEN)

logging.info(f"X shape: {X.shape}")
logging.info(f"y shape: {y.shape}")

# ===============================
# 4. SPLIT DATA
# ===============================
num_samples = X.shape[0]
train_size = int(num_samples * 0.7)
val_size = int(num_samples * 0.1)

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

logging.info("Train / Val / Test split done")

# ===============================
# 5. BUILD SEQ2SEQ LSTM
# ===============================
latent_dim = 128

encoder_inputs = Input(shape=(INPUT_LEN, NUM_NODES))
_, state_h, state_c = LSTM(latent_dim, return_state=True)(encoder_inputs)

decoder_inputs = Input(shape=(OUTPUT_LEN, NUM_NODES))
decoder_outputs = LSTM(latent_dim, return_sequences=True)(
    decoder_inputs, initial_state=[state_h, state_c]
)
decoder_outputs = Dense(NUM_NODES)(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer="adam", loss="mse")

logging.info("Model built successfully")

# ===============================
# 6. TRAIN
# ===============================
decoder_input_train = np.zeros_like(y_train)
decoder_input_val = np.zeros_like(y_val)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    [X_train, decoder_input_train],
    y_train,
    validation_data=([X_val, decoder_input_val], y_val),
    epochs=30,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1   # HIỂN THỊ GỌN TRONG CONSOLE
)

logging.info("Training completed")

# ===============================
# 7. LOSS CURVE
# ===============================
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Val")
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend()
plt.grid()

loss_path = f"{LOG_DIR}/loss_curve_{run_time}.png"
plt.savefig(loss_path)
plt.close()

logging.info(f"Loss curve saved: {loss_path}")

# ===============================
# 8. EVALUATION
# ===============================
decoder_input_test = np.zeros_like(y_test)
y_pred = model.predict([X_test, decoder_input_test], verbose=0)

def evaluate_horizon(y_true, y_pred, h):
    yt = y_true[:, h-1, :]
    yp = y_pred[:, h-1, :]
    mae = mean_absolute_error(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    return mae, rmse

metric_path = f"{LOG_DIR}/metrics_{run_time}.txt"
with open(metric_path, "w") as f:
    f.write("METR-LA Traffic Forecasting\n")
    for h, mins in zip([3, 6, 12], [15, 30, 60]):
        mae, rmse = evaluate_horizon(y_test, y_pred, h)
        msg = f"{mins} min -> MAE: {mae:.4f}, RMSE: {rmse:.4f}"
        f.write(msg + "\n")
        logging.info(msg)

# ===============================
# 9. PLOT FORECAST SAMPLE
# ===============================
def plot_prediction(horizon, sensor_id=0):
    plt.figure(figsize=(10, 4))
    plt.plot(y_test[:200, horizon-1, sensor_id], label="Actual")
    plt.plot(y_pred[:200, horizon-1, sensor_id], label="Predicted")
    plt.title(f"{horizon*5}-Minute Forecast | Sensor {sensor_id}")
    plt.legend()
    plt.grid()

    path = f"{LOG_DIR}/forecast_{horizon*5}min_sensor{sensor_id}_{run_time}.png"
    plt.savefig(path)
    plt.close()
    logging.info(f"Forecast plot saved: {path}")

plot_prediction(3)
plot_prediction(6)
plot_prediction(12)

logging.info("===== END TRAINING =====")
