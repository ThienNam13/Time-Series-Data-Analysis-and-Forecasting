# eda.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from preprocessing.data_loader import load_and_prepare_multivariate_data


# =====================================================
# PATH SETUP
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
plt.xlabel("Time")
plt.ylabel("Electricity Load (kW)")
plt.title("Electricity Load Over Time")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(LOG_DIR, "load_timeseries.png"))
plt.close()

# ---- Load vs Temperature
plt.figure(figsize=(6, 5))
plt.scatter(data["temperature"], data["load"], alpha=0.3)
plt.xlabel("Temperature (°C)")
plt.ylabel("Electricity Load (kW)")
plt.title("Load vs Temperature")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(LOG_DIR, "load_vs_temp.png"))
plt.close()

# ---- Correlation heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.xlabel("Variables")
plt.ylabel("Variables")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig(os.path.join(LOG_DIR, "correlation_matrix.png"))
plt.close()

# ---- Average load by hour (daily seasonality)
hourly_avg = data.groupby(data.index.hour)["load"].mean()

plt.figure(figsize=(8, 4))
plt.plot(hourly_avg.index, hourly_avg.values, marker="o")
plt.xlabel("Hour of Day")
plt.ylabel("Average Load (kW)")
plt.title("Average Electricity Load by Hour (Daily Pattern)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(LOG_DIR, "daily_seasonality.png"))
plt.close()

# ---- Weekly pattern after detrending
detrended = data["load"] - data["load"].rolling(24*7).mean()

weekday_avg_dt = detrended.groupby(data.index.weekday).mean()

plt.figure(figsize=(8,4))
plt.plot(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"], weekday_avg_dt.values, marker="o")
plt.xlabel("Day of Week")
plt.ylabel("Detrended Load")
plt.title("Weekly Pattern")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(LOG_DIR, "weekly_pattern_detrended.png"))
plt.close()

# ---- Trend via Moving Average
window = 24 * 7  # 7-day trend

data["load_ma"] = data["load"].rolling(window=window).mean()

plt.figure(figsize=(12, 5))
plt.plot(data.index, data["load"], alpha=0.4, label="Original Load")
plt.plot(data.index, data["load_ma"], color="red", label="7-day Moving Average")

plt.xlabel("Time")
plt.ylabel("Electricity Load (kW)")
plt.title("Load Trend (7-day)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(LOG_DIR, "load_trend_moving_average.png"))
plt.close()
