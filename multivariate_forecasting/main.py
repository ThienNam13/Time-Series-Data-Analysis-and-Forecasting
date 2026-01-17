import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from data_loader import load_and_prepare_multivariate_data
from sklearn.preprocessing import StandardScaler

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
from models.var_model import (
    make_stationary,
    train_var_model,
    forecast_var
)
from models.var_model import evaluate_forecast
from models.xgboost_model import train_xgboost
from models.lstm_model import train_lstm

# =====================================================
# TẠO THƯ MỤC LOG CHO MỖI LẦN CHẠY
# =====================================================
run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_DIR = os.path.dirname(__file__)
log_dir = os.path.join(BASE_DIR, "logs", f"run_{run_time}")
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, "analysis_log.txt")

def write_log(text):
    """Ghi log ra file"""
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(text + "\n")


# =====================================================
# 1. LOAD & MERGE DATASET 
# =====================================================
"""
- Dataset: Electricity Load Diagrams (UCI)
- Biến: load, temperature, humidity, wind_speed
"""

BASE_DIR = os.path.dirname(__file__)

data = load_and_prepare_multivariate_data(
    load_path=os.path.join(BASE_DIR, "data", "LD2011_2014.txt"),
    weather_path=os.path.join(BASE_DIR, "data", "weather.csv"),
    freq="h"
)

write_log("=== LOAD DATA ===")
write_log(f"Số dòng: {data.shape[0]}")
write_log(f"Số cột: {data.shape[1]}")
write_log(f"Các biến: {list(data.columns)}\n")


# =====================================================
# 2. KIỂM TRA DATETIME INDEX
# =====================================================
"""
- Parse datetime
- Set datetime làm index
"""

write_log("=== DATETIME INDEX ===")
write_log(f"Kiểu index: {type(data.index)}")
write_log(f"Thời gian: {data.index.min()} -> {data.index.max()}\n")


# =====================================================
# 3. KIỂM TRA MISSING VALUES
# =====================================================
"""
Kiểm tra missing values
"""

write_log("=== MISSING VALUES ===")
write_log(str(data.isnull().sum()) + "\n")


# =====================================================
# 4. KIỂM TRA OUTLIERS (Z-SCORE)
# =====================================================
"""
Kiểm tra outliers
Sử dụng Z-score > 3
"""

z_score = np.abs((data - data.mean()) / data.std())
outliers = (z_score > 3).sum()

write_log("=== OUTLIERS (Z-score > 3) ===")
write_log(str(outliers) + "\n")


# =====================================================
# 5. KIỂM TRA TẦN SUẤT DỮ LIỆU
# =====================================================
"""
- Kiểm tra seasonality
- Trước hết cần đảm bảo dữ liệu đủ giờ
"""

freq_check = data.index.to_series().diff().value_counts()

write_log("=== TẦN SUẤT DỮ LIỆU ===")
write_log(str(freq_check.head()) + "\n")


# =====================================================
# 6. BIỂU ĐỒ LOAD THEO THỜI GIAN
# =====================================================
"""
- Plot load theo thời gian
"""

plt.figure(figsize=(12, 5))
plt.plot(data.index, data["load"])
plt.title("Electricity Load Over Time")
plt.xlabel("Time")
plt.ylabel("Load")
plt.tight_layout()
plt.savefig(os.path.join(log_dir, "load_time_series.png"))
plt.close()

# =====================================================
# 7. BIỂU ĐỒ CHU KỲ DAILY SEASONALITY
# =====================================================
data["hour"] = data.index.hour
hourly_avg = data.groupby("hour")["load"].mean()

plt.figure(figsize=(8,4))
plt.plot(hourly_avg)
plt.title("Average Load by Hour (Daily Seasonality)")
plt.xlabel("Hour of Day")
plt.ylabel("Average Load")
plt.tight_layout()
plt.savefig(os.path.join(log_dir, "seasonality_daily.png"))
plt.close()

# =====================================================
# 8. BIỂU ĐỒ CHU KỲ WEEKLY SEASONALITY
# =====================================================
data["weekday"] = data.index.weekday
weekly_avg = data.groupby("weekday")["load"].mean()

plt.figure(figsize=(8,4))
plt.plot(weekly_avg)
plt.title("Average Load by Weekday (Weekly Seasonality)")
plt.xlabel("Weekday (0=Mon)")
plt.ylabel("Average Load")
plt.tight_layout()
plt.savefig(os.path.join(log_dir, "seasonality_weekly.png"))
plt.close()

# =====================================================
# 9. BIỂU ĐỒ LOAD VS TEMPERATURE
# =====================================================
"""
- Plot load vs temperature
"""

plt.figure(figsize=(6, 5))
plt.scatter(data["temperature"], data["load"], alpha=0.3)
plt.xlabel("Temperature")
plt.ylabel("Load")
plt.title("Load vs Temperature")
plt.tight_layout()
plt.savefig(os.path.join(log_dir, "load_vs_temperature.png"))
plt.close()


# =====================================================
# 10. MA TRẬN TƯƠNG QUAN
# =====================================================
"""
- Correlation matrix giữa:
  load – temperature – humidity – wind_speed
"""

corr = data[["load", "temperature", "humidity", "wind_speed"]].corr()

plt.figure(figsize=(6, 5))
plt.imshow(corr, cmap="coolwarm")
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns, rotation=45)
plt.yticks(range(len(corr)), corr.columns)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig(os.path.join(log_dir, "correlation_matrix.png"))
plt.close()

write_log("=== CORRELATION MATRIX ===")
write_log(str(corr) + "\n")


# =====================================================
# 11. NHẬN XÉT 
# =====================================================
"""
- Biến thời tiết nào liên quan mạnh với load?
- Có chu kỳ ngày / tuần không?
"""

write_log("=== NHẬN XÉT ===")
write_log(
    "- Temperature có mức tương quan cao nhất với load.\n"
    "- Humidity và wind_speed có tương quan thấp hơn.\n"
    "- Load thể hiện chu kỳ ngày rõ rệt và có dấu hiệu chu kỳ tuần.\n"
)


# ===============================
# 12. PREPROCESSING
# ===============================
write_log("=== PREPROCESSING ===")

features = ["load", "temperature", "humidity", "wind_speed"]
data_pp = data[features].copy()

# ===============================
# 12.1 CHUẨN HÓA DỮ LIỆU
# ===============================
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_pp)

data_scaled = pd.DataFrame(
    data_scaled,
    columns=features,
    index=data_pp.index
)

write_log("Chuẩn hóa dữ liệu bằng StandardScaler")
write_log("- Tránh bias do khác đơn vị đo")
write_log("- Giúp LSTM học ổn định")
write_log("- Cải thiện hiệu quả VAR và XGBoost\n")



# =====================================================
# 13. CHIA TẬP DỮ LIỆU THEO THỜI GIAN
# =====================================================
"""
Theo đề:
- Không shuffle
- Train 70% | Val 15% | Test 15%
"""

n = len(data_scaled)
train_size = int(n * 0.70)
val_size = int(n * 0.15)

train = data_scaled.iloc[:train_size]
val = data_scaled.iloc[train_size:train_size + val_size]
test = data_scaled.iloc[train_size + val_size:]

write_log("=== DATA SPLIT ===")

write_log("TRAIN SET:")
write_log(f"- Số mẫu: {len(train)}")
write_log(f"- Thời gian: {train.index.min()} -> {train.index.max()}\n")

write_log("VALIDATION SET:")
write_log(f"- Số mẫu: {len(val)}")
write_log(f"- Thời gian: {val.index.min()} -> {val.index.max()}\n")

write_log("TEST SET:")
write_log(f"- Số mẫu: {len(test)}")
write_log(f"- Thời gian: {test.index.min()} -> {test.index.max()}\n")

print(f"Hoàn thành Data Understanding & Preprocessing.")
print(f"Kết quả được lưu tại: {log_dir}")

# =====================================================
# 14. FEATURE ENGINEERING - LAG FEATURES (ML & DL)
# =====================================================
write_log("\n=== FEATURE ENGINEERING: LAG FEATURES ===")

L = 24
lag_features = []

fe_data = data.copy()

for var in ["load", "temperature", "humidity", "wind_speed"]:
    for l in range(1, L + 1):
        col_name = f"{var}_lag_{l}"
        fe_data[col_name] = fe_data[var].shift(l)
        lag_features.append(col_name)

rows_before = len(fe_data)
fe_data.dropna(inplace=True)
rows_after = len(fe_data)

write_log(f"Số lag (L): {L}")
write_log(f"Số feature lag tạo ra: {len(lag_features)}")
write_log(f"Số dòng bị drop do lag: {rows_before - rows_after}\n")

# =====================================================
# 15.FEATURE ENGINEERING - ROLLING FEATURES (XGBOOST)
# =====================================================
write_log("=== FEATURE ENGINEERING: ROLLING FEATURES (XGBOOST) ===")

fe_data["load_roll_mean_6"] = fe_data["load"].rolling(window=6).mean()
fe_data["load_roll_std_24"] = fe_data["load"].rolling(window=24).std()
fe_data["load_trend_24"] = fe_data["load"] - fe_data["load"].shift(24)

rows_before = len(fe_data)
fe_data.dropna(inplace=True)
rows_after = len(fe_data)

write_log("Rolling features được tạo:")
write_log("- Rolling mean (6h)")
write_log("- Rolling std (24h)")
write_log("- Load trend: load(t) - load(t-24)")
write_log(f"Số dòng bị drop sau rolling: {rows_before - rows_after}\n")
print("Hoàn thành Feature Engineering.")

# =====================================================
# 16. VAR MODEL - STATIONARITY CHECK (ADF)
# =====================================================
from statsmodels.tsa.stattools import adfuller

def adf_test(series, name):
    result = adfuller(series.dropna())
    p_value = result[1]
    write_log(f"ADF Test - {name}: p-value = {p_value:.5f}")
    return p_value

write_log("\n=== ADF TEST (VAR) ===")

var_vars = ["load", "temperature", "humidity", "wind_speed"]
var_data = data[var_vars].copy()

need_diff = False
for col in var_vars:
    if adf_test(var_data[col], col) > 0.05:
        need_diff = True

if need_diff:
    write_log(" Chuỗi không dừng, thực hiện differencing bậc 1\n")
    var_data = var_data.diff().dropna()
else:
    write_log(" Tất cả chuỗi đều dừng\n")
print("Hoàn thành ADF Test cho VAR Model.")

# =====================================================
# 17. VAR - TRAIN / TEST SPLIT
# =====================================================
n_var = len(var_data)
train_size_var = int(n_var * 0.7)

var_train = var_data.iloc[:train_size_var]
var_test = var_data.iloc[train_size_var:]

write_log("=== VAR DATA SPLIT ===")
write_log(f"Train: {var_train.index.min()} -> {var_train.index.max()}")
write_log(f"Test: {var_test.index.min()} -> {var_test.index.max()}\n")
print("Hoàn thành Train/Test Split cho VAR Model.")

# =====================================================
# 18.VAR - LAG SELECTION & TRAINING
# =====================================================
from statsmodels.tsa.api import VAR

max_lag = 24
model = VAR(var_train)

lag_results = model.select_order(max_lag)
selected_lag = lag_results.bic

write_log("=== VAR LAG SELECTION ===")
write_log(str(lag_results.summary()))
write_log(f"→ Chọn lag = {selected_lag} theo BIC\n")

var_model = model.fit(selected_lag)
write_log("VAR model đã được huấn luyện.\n")
print("Hoàn thành VAR Model Training.")

# =====================================================
# 19.VAR - FORECAST & EVALUATION (LOAD ONLY)
# =====================================================
forecast_steps = len(var_test)

forecast = var_model.forecast(
    y=var_train.values[-selected_lag:],
    steps=forecast_steps
)

forecast_df = pd.DataFrame(
    forecast,
    index=var_test.index,
    columns=var_vars
)

plt.figure(figsize=(12, 5))
plt.plot(var_test.index, var_test["load"], label="Actual Load")
plt.plot(forecast_df.index, forecast_df["load"], label="VAR Forecast")
plt.legend()
plt.title("VAR Forecast vs Actual (Load)")
plt.tight_layout()
plt.savefig(os.path.join(log_dir, "var_forecast_load.png"))
plt.close()

write_log("=== VAR FORECAST ===")
write_log(f"Số bước forecast: {forecast_steps}")

y_true_load = var_test["load"].values
y_pred_load = forecast_df["load"].values

metrics_var = evaluate_forecast(
    y_true=y_true_load,
    y_pred=y_pred_load,
    log_file=log_file,
    model_name="VAR"
)

print("KẾT QUẢ ĐÁNH GIÁ VAR:")
for k, v in metrics_var.items():
    print(f"{k}: {v:.4f}")

# =====================================================
# 20. XGBOOST MODEL (Machine Learning)
# =====================================================
write_log("\n=== XGBOOST MODEL ===")

"""
Input:
- Vector phẳng từ cửa sổ lag:
  [load_lag_1 ... load_lag_L,
   temperature_lag_1 ...,
   humidity_lag_1 ...,
   wind_speed_lag_1 ...]

Output:
- y = load
"""

# Chia lại dữ liệu feature engineering theo thời gian
n_fe = len(fe_data)
train_size = int(n_fe * 0.70)
val_size = int(n_fe * 0.15)

xgb_train = fe_data.iloc[:train_size]
xgb_val = fe_data.iloc[train_size:train_size + val_size]
xgb_test = fe_data.iloc[train_size + val_size:]

write_log("XGBoost data split:")
write_log(f"- Train size: {len(xgb_train)}")
write_log(f"- Validation size: {len(xgb_val)}")
write_log(f"- Test size: {len(xgb_test)}\n")

# Train XGBoost
y_pred_xgb, rmse_xgb, mae_xgb, mape_xgb = train_xgboost(
    train_df=xgb_train,
    val_df=xgb_val,
    test_df=xgb_test,
    log_dir=log_dir,
    max_depth=6,
    n_estimators=500
)

write_log("XGBoost Results:")
write_log(f"- RMSE: {rmse_xgb:.4f}")
write_log(f"- MAE: {mae_xgb:.4f}")
write_log(f"- MAPE: {mape_xgb:.2f}%\n")

print("Hoàn thành XGBoost Model.")

# =====================================================
# 21. LSTM MODEL (Deep Learning)
# =====================================================
write_log("\n=== LSTM MODEL ===")

"""
Input:
- Tensor 3D (samples, 24 timesteps, 4 variables)
- Variables: load, temperature, humidity, wind_speed

Output:
- y = load
"""

# LSTM sử dụng dữ liệu đã scale (data_scaled)
n_lstm = len(data_scaled)

train_size = int(n_lstm * 0.70)
val_size = int(n_lstm * 0.15)

lstm_train = data_scaled.iloc[:train_size]
lstm_val = data_scaled.iloc[train_size:train_size + val_size]
lstm_test = data_scaled.iloc[train_size + val_size:]

write_log("LSTM data split:")
write_log(f"- Train size: {len(lstm_train)}")
write_log(f"- Validation size: {len(lstm_val)}")
write_log(f"- Test size: {len(lstm_test)}\n")

# Train LSTM
y_pred_lstm, rmse_lstm, mae_lstm, mape_lstm = train_lstm(
    train_df=lstm_train,
    val_df=lstm_val,
    test_df=lstm_test,
    log_dir=log_dir,
    window_size=24,
    epochs=50
)

write_log("LSTM Results:")
write_log(f"- RMSE: {rmse_lstm:.4f}")
write_log(f"- MAE: {mae_lstm:.4f}")
write_log(f"- MAPE: {mape_lstm*100:.2f}%\n")

print("Hoàn thành LSTM Model.")

# =====================================================
# 22. WALK-FORWARD BACKTESTING
# =====================================================
from evaluation.backtesting import (
    walk_forward_var,
    walk_forward_xgboost,
    walk_forward_lstm,
    evaluate_metrics
)

write_log("\n=== WALK-FORWARD BACKTESTING ===")

# ===== VAR =====
wf_test_var = var_test.iloc[:300]
y_true_var, y_pred_var = walk_forward_var(var_train, wf_test_var, log_dir=log_dir)

rmse_var, mae_var, mape_var = evaluate_metrics(y_true_var, y_pred_var)
write_log("VAR Walk-forward:")
write_log(f"RMSE: {rmse_var:.4f}")
write_log(f"MAE : {mae_var:.4f}")
write_log(f"MAPE: {mape_var:.2f}%\n")

# ===== XGBOOST =====
xgb_params = {
    "objective": "reg:squarederror",
    "max_depth": 6,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 42
}

wf_test_xgb = xgb_test.iloc[:300]
y_true_xgb, y_pred_xgb = walk_forward_xgboost(
    xgb_train, wf_test_xgb, params=xgb_params, num_boost_round=100, log_dir=log_dir
)

rmse_xgb, mae_xgb, mape_xgb = evaluate_metrics(y_true_xgb, y_pred_xgb)
write_log("XGBoost Walk-forward:")
write_log(f"RMSE: {rmse_xgb:.4f}")
write_log(f"MAE : {mae_xgb:.4f}")
write_log(f"MAPE: {mape_xgb:.2f}%\n")


# ===== LSTM =====
test_subset = lstm_test.iloc[:200]

y_true_lstm, y_pred_lstm = walk_forward_lstm(
    lstm_train, test_subset,
    window_size=24,
    epochs=3,
    log_dir=log_dir
)

rmse_lstm, mae_lstm, mape_lstm = evaluate_metrics(y_true_lstm, y_pred_lstm)
write_log("LSTM Walk-forward:")
write_log(f"RMSE: {rmse_lstm:.4f}")
write_log(f"MAE : {mae_lstm:.4f}")
write_log(f"MAPE: {mape_lstm:.2f}%\n")
