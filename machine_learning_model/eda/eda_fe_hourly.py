import os
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# PATH CONFIG
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "FE_hourly.csv")

EDA_DIR = os.path.join(BASE_DIR, "eda", "figures")
os.makedirs(EDA_DIR, exist_ok=True)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATA_PATH)
df["Datetime"] = pd.to_datetime(df["Datetime"])
df.set_index("Datetime", inplace=True)
df.sort_index(inplace=True)

# =========================
# 1. TIME SERIES PLOT
# =========================
plt.figure(figsize=(12, 5))
plt.plot(df.index, df.iloc[:, 0])
plt.xlabel("Datetime")
plt.ylabel("Electricity Consumption")
plt.title("Hourly Electricity Consumption Time Series")
plt.tight_layout()

ts_plot = os.path.join(EDA_DIR, "time_series_plot.png")
plt.savefig(ts_plot)
plt.close()

# =========================
# 2. SEASONALITY BY HOUR
# =========================
df["hour"] = df.index.hour
hourly_mean = df.groupby("hour").mean()

plt.figure(figsize=(10, 5))
plt.plot(hourly_mean.index, hourly_mean.iloc[:, 0])
plt.xlabel("Hour of Day")
plt.ylabel("Average Electricity Consumption")
plt.title("Average Electricity Consumption by Hour of Day")
plt.xticks(range(0, 24))
plt.tight_layout()

season_plot = os.path.join(EDA_DIR, "average_consumption_by_hour.png")
plt.savefig(season_plot)
plt.close()

print("EDA plots saved:")
print(ts_plot)
print(season_plot)
