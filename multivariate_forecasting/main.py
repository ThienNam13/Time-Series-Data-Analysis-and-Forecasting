import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from data_loader import load_and_prepare_multivariate_data
from sklearn.preprocessing import StandardScaler

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
# 7. BIỂU ĐỒ LOAD VS TEMPERATURE
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
# 8. MA TRẬN TƯƠNG QUAN
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
# 9. NHẬN XÉT 
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
# 10. PREPROCESSING
# ===============================
write_log("=== PREPROCESSING ===")

features = ["load", "temperature", "humidity", "wind_speed"]
data_pp = data[features].copy()

# ===============================
# 10.1 CHUẨN HÓA DỮ LIỆU
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
# 11. CHIA TẬP DỮ LIỆU THEO THỜI GIAN
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
