import os
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

# =========================================================
# 1. PATH CONFIGURATION
# =========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "machine_learning_model\data")
LOG_DIR = os.path.join(
    BASE_DIR,
    "machine_learning_model",
    "logs",
    "XGBoost_model",
    "AEPhourly"
)

os.makedirs(LOG_DIR, exist_ok=True)
# Timestamp for this run
RUN_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")

# === RUN-SPECIFIC DIRECTORY ===
RUN_DIR = os.path.join(LOG_DIR, f"run_{RUN_TIME}")
os.makedirs(RUN_DIR, exist_ok=True)

DATA_PATH = os.path.join(DATA_DIR, "AEP_hourly.csv")
LOG_FILE = os.path.join(RUN_DIR, f"xgboost_pipeline_{RUN_TIME}.log")
INFO_TXT = os.path.join(RUN_DIR, f"data_check_{RUN_TIME}.txt")
FORECAST_TXT = os.path.join(RUN_DIR, f"multi_step_forecast_{RUN_TIME}.txt")

SUPERVISED_TXT = os.path.join(RUN_DIR, f"X_y_split_{RUN_TIME}.txt")
SPLIT_INFO_TXT = os.path.join(RUN_DIR, f"train_test_split_{RUN_TIME}.txt")
MODEL_INFO_TXT = os.path.join(RUN_DIR, f"xgboost_model_info_{RUN_TIME}.txt")

# =========================================================
# 2. LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ],
    force=True
)

logger = logging.getLogger(__name__)

# =========================================================
# 3. FUNCTIONS
# =========================================================

def load_data(path: str) -> pd.DataFrame:
    logger.info(f"Loading dataset from: {path}")
    df = pd.read_csv(path)
    logger.info(f"Dataset shape: {df.shape}")
    return df


def preprocess_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df.set_index("Datetime", inplace=True)
    df.sort_index(inplace=True)
    logger.info("Datetime parsed & set as index")
    return df


def check_data_quality(df: pd.DataFrame):
    missing = df.isna().sum().sum()
    freq = pd.infer_freq(df.index)

    with open(INFO_TXT, "w", encoding="utf-8") as f:
        f.write("DATA QUALITY CHECK – AEP_hourly\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Missing values: {missing}\n")
        f.write(f"Inferred frequency: {freq}\n")

        if missing == 0:
            f.write("→ No missing values detected.\n")
        else:
            f.write("→ Dataset contains missing values.\n")

        if freq == "H":
            f.write("→ Hourly frequency is consistent.\n")
        else:
            f.write("→ WARNING: Hourly frequency is broken.\n")

    logger.info("Data quality check saved")


def create_lag_features(series: pd.Series, n_lags: int = 24) -> pd.DataFrame:
    """
    Convert time series to supervised learning format
    """
    df = pd.DataFrame({"y": series})

    for lag in range(1, n_lags + 1):
        df[f"y_lag_{lag}"] = series.shift(lag)

    before = len(df)
    df.dropna(inplace=True)
    after = len(df)

    logger.info(f"Number of lags: {n_lags}")
    logger.info(f"Rows dropped due to lagging: {before - after}")
    logger.info(f"Feature columns: {list(df.columns)}")

    return df


def multi_step_forecast(model, last_window, n_steps=24):
    """
    Recursive multi-step forecast
    """
    forecasts = []
    current_window = last_window.copy()

    for step in range(n_steps):
        pred = model.predict(current_window.reshape(1, -1))[0]
        forecasts.append(pred)

        current_window = np.roll(current_window, -1)
        current_window[-1] = pred

    return forecasts
# CHIA X-y
def split_X_y(supervised_df: pd.DataFrame):
    """
    Split supervised dataframe into X and y
    """
    X = supervised_df.drop(columns="y")
    y = supervised_df["y"]

    logger.info("Split supervised data into X and y")
    logger.info(f"X shape: {X.shape}")
    logger.info(f"y shape: {y.shape}")

    # ===== WRITE REPORT FILE =====
    with open(SUPERVISED_TXT, "w", encoding="utf-8") as f:
        f.write("TASK 3 – SUPERVISED LEARNING FORMAT\n")
        f.write("=" * 50 + "\n\n")
        f.write("Target variable (y): current value\n")
        f.write("Features (X): lagged values\n\n")
        f.write(f"Total samples: {len(supervised_df)}\n")
        f.write(f"X shape: {X.shape}\n")
        f.write(f"y shape: {y.shape}\n\n")
        f.write("Feature columns:\n")
        for col in X.columns:
            f.write(f"- {col}\n")

    logger.info(f"Supervised X/y split info saved → {SUPERVISED_TXT}")

    return X, y

# TRAIN / TEST SPLIT (THEO THỜI GIAN – KHÔNG SHUFFLE)
def time_series_train_test_split(X, y, train_ratio=0.8):
    """
    Split data into train/test sets without shuffling
    """
    split_idx = int(len(X) * train_ratio)

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    logger.info("Time series train-test split completed")
    logger.info(f"Train size: {len(X_train)}")
    logger.info(f"Test size: {len(X_test)}")

    # ===== WRITE REPORT FILE =====
    with open(SPLIT_INFO_TXT, "w", encoding="utf-8") as f:
        f.write("TASK 4 – TIME SERIES TRAIN / TEST SPLIT\n")
        f.write("=" * 55 + "\n\n")
        f.write("Split strategy:\n")
        f.write("- Train: 80%\n")
        f.write("- Test : 20%\n")
        f.write("- No shuffling (time order preserved)\n\n")

        f.write(f"Total samples: {len(X)}\n")
        f.write(f"Train samples: {len(X_train)}\n")
        f.write(f"Test samples : {len(X_test)}\n\n")

        f.write("Train period:\n")
        f.write(f"  From: {X_train.index.min()}\n")
        f.write(f"  To  : {X_train.index.max()}\n\n")

        f.write("Test period:\n")
        f.write(f"  From: {X_test.index.min()}\n")
        f.write(f"  To  : {X_test.index.max()}\n\n")

        f.write("Rationale:\n")
        f.write(
            "- Time series data must not be shuffled.\n"
            "- Future information must not leak into training data.\n"
        )

    logger.info(f"Train/test split info saved → {SPLIT_INFO_TXT}")

    return X_train, X_test, y_train, y_test

# =========================================================
# 4. MAIN
# =========================================================

def main():
    logger.info("========== START XGBOOST PIPELINE ==========")

    # Load & preprocess
    df = load_data(DATA_PATH)
    df = preprocess_datetime(df)

    # Data quality check
    check_data_quality(df)

    # Supervised learning
    supervised_df = create_lag_features(df.iloc[:, 0], n_lags=24)

    # =============================
    # Split X / y
    # =============================
    X, y = split_X_y(supervised_df)

    # =============================
    # Train / Test split
    # =============================
    X_train, X_test, y_train, y_test = time_series_train_test_split(X, y)

    # =============================
    # Train XGBoost on TRAIN ONLY
    # =============================
    model = XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        objective="reg:squarederror",
        random_state=42
    )

    model.fit(X_train, y_train)
    logger.info("XGBoost model trained on TRAIN set only")
    # =============================
    # SAVE MODEL INFO
    # ============================= 
    with open(MODEL_INFO_TXT, "w", encoding="utf-8") as f:
        f.write("BASELINE XGBOOST MODEL\n")
        f.write("=" * 55 + "\n\n")

        f.write("Model type: XGBoost Regressor\n")
        f.write("Purpose: Baseline model (no hyperparameter tuning)\n\n")

        f.write("Training strategy:\n")
        f.write("- Train set only (time-series split)\n")
        f.write("- No data shuffling\n")
        f.write("- No validation / CV at this stage\n\n")

        f.write("Model hyperparameters:\n")
        f.write(f"- n_estimators   : {model.n_estimators}\n")
        f.write(f"- max_depth     : {model.max_depth}\n")
        f.write(f"- learning_rate : {model.learning_rate}\n")
        f.write(f"- objective     : {model.objective}\n")
        f.write(f"- random_state  : {model.random_state}\n\n")

        f.write("Input features:\n")
        f.write(f"- Number of lag features: {X_train.shape[1]}\n")
        f.write("- Feature description: y(t-1) ... y(t-24)\n\n")

        f.write("Target variable:\n")
        f.write("- y(t): current hourly power consumption\n\n")

        f.write("Notes:\n")
        f.write(
            "- This model serves as a baseline for later comparison.\n"
            "- Performance evaluation and tuning will be done in later tasks.\n"
        )

    logger.info(f"Baseline model info saved → {MODEL_INFO_TXT}")

    # Multi-step forecast (24 hours)
    last_window = X.iloc[-1].values
    forecasts = multi_step_forecast(model, last_window, n_steps=24)

    # Save forecast results
    with open(FORECAST_TXT, "w", encoding="utf-8") as f:
        f.write("MULTI-STEP FORECAST (Next 24 Hours)\n")
        f.write("=" * 50 + "\n")
        for i, val in enumerate(forecasts, 1):
            f.write(f"t+{i}: {val:.2f}\n")

    logger.info("Multi-step forecast completed")
    logger.info(f"Forecast saved to: {FORECAST_TXT}")
    logger.info("=========== END XGBOOST PIPELINE ===========")

if __name__ == "__main__":
    main()
