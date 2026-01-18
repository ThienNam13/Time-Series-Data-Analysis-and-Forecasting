# main.py

import os
from datetime import datetime

import numpy as np
import pandas as pd

from preprocessing.data_loader import load_and_prepare_multivariate_data
from preprocessing.scaling import split_and_scale_time_series
from preprocessing.feature_engineering import build_ml_features

from models.var_model import train_var_model, make_stationary
from models.xgboost_model import train_xgboost
from models.lstm_model import train_lstm

from evaluation.backtesting import (
    walk_forward_var,
    walk_forward_xgboost,
    walk_forward_lstm,
    evaluate_metrics
)


# =====================================================
# LOG DIR
# =====================================================
run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"logs/run_{run_time}"
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, "analysis_log.txt")


def write_log(text):
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(text + "\n")


# =====================================================
# 1. LOAD & MERGE DATA
# =====================================================
write_log("=== LOAD DATA ===")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

data = load_and_prepare_multivariate_data(
    load_path=os.path.join(DATA_DIR, "LD2011_2014.txt"),
    weather_path=os.path.join(DATA_DIR, "weather.csv"),
    freq="H"
)

write_log(f"Shape: {data.shape}")
write_log(f"Columns: {list(data.columns)}")
write_log(f"Time range: {data.index.min()} -> {data.index.max()}\n")


# =====================================================
# 2. SPLIT + SCALE (FIT ON TRAIN ONLY)
# =====================================================
write_log("=== SPLIT & SCALE ===")

train_scaled, val_scaled, test_scaled, scaler = split_and_scale_time_series(
    data,
    features=["load", "temperature", "humidity", "wind_speed"],
    train_ratio=0.7,
    val_ratio=0.15
)

write_log("Scaling: StandardScaler (fit on train only)")
write_log(f"Train: {train_scaled.shape}")
write_log(f"Val  : {val_scaled.shape}")
write_log(f"Test : {test_scaled.shape}\n")


# =====================================================
# 3. FEATURE ENGINEERING (SUPERVISED)
# =====================================================
write_log("=== FEATURE ENGINEERING ===")

WINDOW = 24
# ===== DEBUG / FAST RUN SETTINGS =====
TEST_SUBSET_VAR_XGB = 300
TEST_SUBSET_LSTM = 100

train_sup, _ = build_ml_features(train_scaled, variables=["load","temperature","humidity","wind_speed"], max_lag=WINDOW)
val_sup, _   = build_ml_features(val_scaled,   variables=["load","temperature","humidity","wind_speed"], max_lag=WINDOW)
test_sup, _  = build_ml_features(test_scaled,  variables=["load","temperature","humidity","wind_speed"], max_lag=WINDOW)

write_log(f"Train supervised: {train_sup.shape}")
write_log(f"Val supervised  : {val_sup.shape}")
write_log(f"Test supervised : {test_sup.shape}\n")


# =====================================================
# 4. TRAIN MODELS (ONE-SHOT)
# =====================================================
write_log("=== TRAIN MODELS ===")

# ---------- VAR ----------
write_log("Training VAR model...")

var_model, var_lag = train_var_model(train_scaled, max_lag=24, logger=write_log)

# ---------- XGBOOST ----------
write_log("Training XGBoost model...")

xgb_params = {
    "objective": "reg:squarederror",
    "max_depth": 6,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8
}

xgb_preds, xgb_rmse, xgb_mae, xgb_mape = train_xgboost(
    train_sup,
    val_sup,
    test_sup,
    scaler,
    log_dir,
    logger=write_log
)

# ---------- LSTM ----------
write_log("Training LSTM model...")

lstm_preds, lstm_rmse, lstm_mae, lstm_mape = train_lstm(
    train_scaled,
    val_scaled,
    test_scaled,
    scaler,
    log_dir=log_dir,
    window_size=WINDOW,
    epochs=30
)

write_log("LSTM Metrics (on scaled test):")
write_log(f"RMSE: {lstm_rmse:.4f}")
write_log(f"MAE : {lstm_mae:.4f}")
write_log(f"MAPE: {lstm_mape*100:.2f}%\n")


# ===== LIMIT TEST SIZE FOR FAST BACKTESTING =====
test_scaled_var_xgb = test_scaled.iloc[:TEST_SUBSET_VAR_XGB]
test_sup_var_xgb = test_sup.iloc[:TEST_SUBSET_VAR_XGB]

test_scaled_lstm = test_scaled.iloc[:TEST_SUBSET_LSTM]

# =====================================================
# 5. WALK-FORWARD BACKTESTING
# =====================================================
write_log("=== WALK-FORWARD BACKTESTING ===")

# ---------- VAR ----------
write_log("VAR Walk-forward...")

train_var_hist = train_scaled
test_var_hist = test_scaled_var_xgb

y_true_var, y_pred_var = walk_forward_var(
    train_var_hist,
    test_var_hist,
    scaler,
    max_lag=24,
    log_dir=log_dir
)

rmse_var, mae_var, mape_var = evaluate_metrics(y_true_var, y_pred_var)

write_log("VAR Walk-forward Metrics:")
write_log(f"RMSE: {rmse_var:.4f}")
write_log(f"MAE : {mae_var:.4f}")
write_log(f"MAPE: {mape_var:.2f}%\n")


# ---------- XGBOOST ----------
write_log("XGBoost Walk-forward...")

y_true_xgb, y_pred_xgb = walk_forward_xgboost(
    train_sup,
    test_sup_var_xgb,
    scaler,
    params=xgb_params,
    num_boost_round=100,
    log_dir=log_dir
)

rmse_xgb, mae_xgb, mape_xgb = evaluate_metrics(y_true_xgb, y_pred_xgb)


write_log("XGBoost Walk-forward Metrics:")
write_log(f"RMSE: {rmse_xgb:.4f}")
write_log(f"MAE : {mae_xgb:.4f}")
write_log(f"MAPE: {mape_xgb:.2f}%\n")


# ---------- LSTM ----------
write_log("LSTM Walk-forward...")

y_true_lstm, y_pred_lstm = walk_forward_lstm(
    train_scaled,
    test_scaled_lstm,
    scaler,
    window_size=WINDOW,
    epochs=3,
    batch_size=64,
    log_dir=log_dir
)

rmse_lstm, mae_lstm, mape_lstm = evaluate_metrics(y_true_lstm, y_pred_lstm)

write_log("LSTM Walk-forward Metrics:")
write_log(f"RMSE: {rmse_lstm:.4f}")
write_log(f"MAE : {mae_lstm:.4f}")
write_log(f"MAPE: {mape_lstm:.2f}%\n")

# =====================================================
# 10. ĐÁNH GIÁ & SO SÁNH MÔ HÌNH (WALK-FORWARD)
# =====================================================
write_log("=== ĐÁNH GIÁ & SO SÁNH MÔ HÌNH (WALK-FORWARD) ===")

write_log("Mục tiêu:")
write_log("- So sánh các mô hình dựa trên kết quả walk-forward validation.")
write_log("- Mô phỏng đúng kịch bản dự báo trong thực tế.\n")

write_log("Chỉ số đánh giá:")
write_log("- RMSE: Độ lệch tổng thể, nhạy với sai số lớn.")
write_log("- MAE : Sai số tuyệt đối trung bình.")
write_log("- MAPE: Sai số phần trăm.\n")

write_log("Bảng so sánh kết quả:")

write_log("| Model   | RMSE   | MAE    | MAPE (%) |")
write_log("|---------|--------|--------|----------|")
write_log(f"| VAR     | {rmse_var:.2f} | {mae_var:.2f} | {mape_var:.2f} |")
write_log(f"| XGBoost | {rmse_xgb:.2f} | {mae_xgb:.2f} | {mape_xgb:.2f} |")
write_log(f"| LSTM    | {rmse_lstm:.2f} | {mae_lstm:.2f} | {mape_lstm:.2f} |")

write_log("\nNhận xét:")
write_log("- VAR: Mô hình tuyến tính, hiệu quả hạn chế với dữ liệu phi tuyến.")
write_log("- XGBoost: Hiệu quả cao, học tốt quan hệ phi tuyến.")
write_log("- LSTM: Dự báo ổn định, nắm bắt phụ thuộc theo thời gian.\n")

write_log("Kết luận:")
write_log("- Mô hình có RMSE, MAE và MAPE thấp nhất được đánh giá là tốt nhất.")
write_log("- Kết quả này được dùng để lựa chọn mô hình triển khai.\n")

write_log("=== DONE ===")
print(f"ALL FINISHED. Logs at: {log_dir}")