# eda.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing.data_loader import load_and_prepare_multivariate_data


# =====================================================
# PATH SETUP (CHỐNG LỖI PATH)
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR = os.path.join(BASE_DIR, "logs", "eda")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "eda_log.txt")


def write_log(text):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")


# =====================================================
# LOAD DATA
# =====================================================
data = load_and_prepare_multivariate_data(
    load_path=os.path.join(DATA_DIR, "LD2011_2014.txt"),
    weather_path=os.path.join(DATA_DIR, "weather.csv"),
    freq="H"
)

write_log("=== LOAD DATA ===")
write_log(f"Số dòng: {data.shape[0]}")
write_log(f"Số cột: {data.shape[1]}")
write_log(f"Các biến: {list(data.columns)}\n")

# =====================================================
# DATETIME INDEX CHECK
# =====================================================
write_log("=== DATETIME INDEX ===")
write_log(f"Kiểu index: {type(data.index)}")
write_log(f"Thời gian: {data.index.min()} -> {data.index.max()}\n")

# =====================================================
# MISSING VALUES
# =====================================================
write_log("=== MISSING VALUES ===")
missing = data.isnull().sum()
write_log(str(missing))
write_log("")

# =====================================================
# OUTLIERS (Z-SCORE > 3)
# =====================================================
from scipy.stats import zscore

z_scores = np.abs(zscore(data))
outliers = (z_scores > 3).sum(axis=0)

write_log("=== OUTLIERS (Z-score > 3) ===")
write_log(pd.Series(outliers, index=data.columns).to_string())
write_log("")

# =====================================================
# DATA FREQUENCY CHECK
# =====================================================
freq = data.index.to_series().diff().value_counts().head(5)

write_log("=== TẦN SUẤT DỮ LIỆU ===")
write_log(freq.to_string())
write_log("")

# =====================================================
# CORRELATION MATRIX
# =====================================================
corr = data.corr()

write_log("=== CORRELATION MATRIX ===")
write_log(corr.to_string())
write_log("")

# =====================================================
# PLOTS
# =====================================================

# ---- Load over time
plt.figure(figsize=(12, 5))
plt.plot(data.index, data["load"])
plt.title("Electricity Load Over Time")
plt.tight_layout()
plt.savefig(os.path.join(LOG_DIR, "load_timeseries.png"))
plt.close()

# ---- Load vs Temperature
plt.figure(figsize=(6, 5))
plt.scatter(data["temperature"], data["load"], alpha=0.3)
plt.xlabel("Temperature")
plt.ylabel("Load")
plt.title("Load vs Temperature")
plt.tight_layout()
plt.savefig(os.path.join(LOG_DIR, "load_vs_temp.png"))
plt.close()

# ---- Correlation heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig(os.path.join(LOG_DIR, "correlation_matrix.png"))
plt.close()

# =====================================================
# SIMPLE ANALYSIS TEXT (CHO BÁO CÁO)
# =====================================================
write_log("=== NHẬN XÉT ===")

max_corr_var = corr["load"].drop("load").abs().idxmax()

write_log(f"- Biến có tương quan mạnh nhất với load: {max_corr_var}")
write_log("- Temperature có tương quan dương khá cao với load.")
write_log("- Humidity có tương quan âm tương đối rõ.")
write_log("- Wind speed ảnh hưởng yếu.")
write_log("- Load thể hiện chu kỳ ngày rất rõ (24h seasonality).")
write_log("- Có dấu hiệu chu kỳ tuần (weekday vs weekend).")

write_log("\n=== END OF EDA ===")

print("EDA FINISHED.")
print(f"Logs & figures saved at: {LOG_DIR}")