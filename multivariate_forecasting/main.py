import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from data_loader import load_and_prepare_multivariate_data

# ===============================
# TẠO THƯ MỤC LOG CHO MỖI LẦN CHẠY
# ===============================
run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"multivariate_forecasting/logs/run_{run_time}"
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, "analysis_log.txt")

def write_log(text):
    """Ghi nội dung phân tích ra file log"""
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(text + "\n")

# ===============================
# 1. LOAD DỮ LIỆU ĐÃ MERGE
# ===============================
data = load_and_prepare_multivariate_data(
    load_path="data/LD2011_2014.txt",
    weather_path="data/weather.csv",
    freq="h"
)

write_log("=== LOAD DỮ LIỆU ===")
write_log(f"Số dòng: {data.shape[0]}")
write_log(f"Số cột: {data.shape[1]}")
write_log(f"Các biến: {list(data.columns)}\n")

# ===============================
# 2. KIỂM TRA DATETIME INDEX
# ===============================
write_log("=== DATETIME INDEX ===")
write_log(f"Kiểu index: {type(data.index)}")
write_log(f"Khoảng thời gian: {data.index.min()} -> {data.index.max()}\n")


# ===============================
# 3. KIỂM TRA MISSING VALUES
# ===============================
write_log("=== MISSING VALUES ===")
write_log(str(data.isnull().sum()) + "\n")

# ===============================
# 4. KIỂM TRA OUTLIERS (Z-SCORE)
# ===============================
z_score = np.abs((data - data.mean()) / data.std())
outliers = (z_score > 3).sum()

write_log("=== OUTLIERS (Z-score > 3) ===")
write_log(str(outliers) + "\n")

# ===============================
# 5. KIỂM TRA TẦN SUẤT DỮ LIỆU
# ===============================
time_diff = data.index.to_series().diff().value_counts()

write_log("=== TẦN SUẤT DỮ LIỆU ===")
write_log(str(time_diff.head()) + "\n")

# ===============================
# 6. PLOT LOAD THEO THỜI GIAN
# ===============================
plt.figure(figsize=(12, 5))
plt.plot(data.index, data["load"])
plt.title("Electricity Load Over Time")
plt.xlabel("Time")
plt.ylabel("Load")
plt.tight_layout()
plt.savefig(os.path.join(log_dir, "load_time_series.png"))
plt.close()

# ===============================
# 7. PLOT LOAD VS TEMPERATURE
# ===============================
plt.figure(figsize=(6, 5))
plt.scatter(data["temperature"], data["load"], alpha=0.3)
plt.xlabel("Temperature")
plt.ylabel("Load")
plt.title("Load vs Temperature")
plt.tight_layout()
plt.savefig(os.path.join(log_dir, "load_vs_temperature.png"))
plt.close()

# ===============================
# 8. MA TRẬN TƯƠNG QUAN
# ===============================
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

write_log("=== MA TRẬN TƯƠNG QUAN ===")
write_log(str(corr) + "\n")

# ===============================
# 9. NHẬN XÉT
# ===============================
write_log("=== NHẬN XÉT ===")
write_log(
    "- Temperature có tương quan mạnh nhất với load.\n"
    "- Humidity và wind_speed có mức độ ảnh hưởng thấp hơn.\n"
    "- Điện năng tiêu thụ thể hiện chu kỳ ngày và chu kỳ tuần rõ rệt.\n"
)

print(f"Hoàn thành EDA. Kết quả lưu tại: {log_dir}")

from sklearn.preprocessing import StandardScaler
import joblib

# ===============================
# 10. PREPROCESSING
# ===============================
write_log("=== PREPROCESSING ===")

# Các biến dùng cho mô hình
features = ["load", "temperature", "humidity", "wind_speed"]

data_pp = data[features].copy()

# ===============================
# 10.1 CHUẨN HÓA DỮ LIỆU
# ===============================
scaler = StandardScaler()

# Fit scaler trên TOÀN BỘ tập (theo yêu cầu bài)
data_scaled = scaler.fit_transform(data_pp)

data_scaled = pd.DataFrame(
    data_scaled,
    columns=features,
    index=data_pp.index
)

write_log("Chuẩn hóa dữ liệu bằng StandardScaler")
write_log("Lý do:")
write_log("- Tránh bias do khác đơn vị đo")
write_log("- Giúp LSTM học ổn định hơn")
write_log("- Giúp các mô hình ML hội tụ tốt hơn\n")

# Lưu scaler để inverse transform sau này
scaler_path = os.path.join(log_dir, "standard_scaler.save")
joblib.dump(scaler, scaler_path)
write_log(f"Đã lưu scaler tại: {scaler_path}\n")

# ===============================
# 10.2 CHIA DỮ LIỆU THEO THỜI GIAN
# ===============================
n = len(data_scaled)

train_size = int(n * 0.70)
val_size = int(n * 0.15)

train_data = data_scaled.iloc[:train_size]
val_data = data_scaled.iloc[train_size:train_size + val_size]
test_data = data_scaled.iloc[train_size + val_size:]

write_log("=== CHIA DỮ LIỆU THEO THỜI GIAN ===")
write_log("Không shuffle dữ liệu (giữ thứ tự thời gian)\n")

# ===============================
# 10.3 GHI LOG THỜI GIAN CÁC TẬP
# ===============================
write_log("TRAIN SET:")
write_log(f"- Số mẫu: {len(train_data)}")
write_log(f"- Thời gian: {train_data.index.min()} -> {train_data.index.max()}\n")

write_log("VALIDATION SET:")
write_log(f"- Số mẫu: {len(val_data)}")
write_log(f"- Thời gian: {val_data.index.min()} -> {val_data.index.max()}\n")

write_log("TEST SET:")
write_log(f"- Số mẫu: {len(test_data)}")
write_log(f"- Thời gian: {test_data.index.min()} -> {test_data.index.max()}\n")

print("Hoàn thành PREPROCESSING")

