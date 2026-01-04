import os
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error


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
EVALUATION_TXT = os.path.join(RUN_DIR, f"evaluation_{RUN_TIME}.txt")
FEATURE_IMPORTANCE_TXT = os.path.join(RUN_DIR, f"feature_importance_{RUN_TIME}.txt")
ROLLING_STATS_TXT = os.path.join(RUN_DIR, f"rolling_stats_{RUN_TIME}.txt")

SUPERVISED_TXT = os.path.join(RUN_DIR, f"X_y_split_{RUN_TIME}.txt")
SPLIT_INFO_TXT = os.path.join(RUN_DIR, f"train_test_split_{RUN_TIME}.txt")
MODEL_INFO_TXT = os.path.join(RUN_DIR, f"xgboost_model_info_{RUN_TIME}.txt")

TUNING_TXT = os.path.join(RUN_DIR, f"xgboost_tuning_{RUN_TIME}.txt")

MODEL_COMPARISON_TXT = os.path.join(RUN_DIR,"model_comparison.txt")



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

        logger.info(f"Supervised X/y split info saved -> {SUPERVISED_TXT}")

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

        logger.info(f"Train/test split info saved -> {SPLIT_INFO_TXT}")

    return X_train, X_test, y_train, y_test

# Tạo hàm tune XGBoost
def tune_xgboost_timeseries(X_train, y_train, n_splits=3):
    """
    Tune XGBoost hyperparameters using TimeSeriesSplit
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    param_grid = [
        {"max_depth": 3, "learning_rate": 0.1, "n_estimators": 100},
        {"max_depth": 5, "learning_rate": 0.05, "n_estimators": 200},
        {"max_depth": 6, "learning_rate": 0.05, "n_estimators": 300},
    ]

    results = []

    for params in param_grid:
        fold_mae = []

        for train_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model = XGBRegressor(
                objective="reg:squarederror",
                random_state=42,
                **params
            )
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            mae = mean_absolute_error(y_val, preds)
            fold_mae.append(mae)

        avg_mae = np.mean(fold_mae)
        results.append((params, avg_mae))

        logger.info(f"Tuning result {params} → MAE={avg_mae:.4f}")

    best_params = sorted(results, key=lambda x: x[1])[0]

    return best_params

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

    logger.info(f"Baseline model info saved -> {MODEL_INFO_TXT}")

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
    #predict: Task 6 — Predict & Evaluate
    logger.info("Starting Task 6: Predict on TEST and evaluate")

    # Predict on test set
    y_pred = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    # MAPE (handle zero actuals safely)
    y_test_arr = y_test.values
    denom = np.where(y_test_arr == 0, np.nan, y_test_arr)
    mape = np.nanmean(np.abs((y_test_arr - y_pred) / denom)) * 100

    # Write evaluation log
    with open(EVALUATION_TXT, "w", encoding="utf-8") as f:
        f.write("TASK 6 – PREDICT & EVALUATE\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Test samples: {len(X_test)}\n")
        f.write(f"Test period from: {X_test.index.min()}\n")
        f.write(f"Test period to  : {X_test.index.max()}\n\n")
        f.write(f"MAE : {mae:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        if np.isnan(mape):
            f.write("MAPE: NaN (division by zero in actuals)\n")
        else:
            f.write(f"MAPE: {mape:.2f}%\n")

    logger.info(f"Evaluation metrics saved -> {EVALUATION_TXT}")

    # Plot: Forecast vs Actual (TEST)
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(y_test.index, y_test.values, label="Actual", color="tab:blue")
        plt.plot(y_test.index, y_pred, label="Predicted", color="tab:orange")
        plt.xlabel("Datetime")
        plt.ylabel("Value")
        plt.title(f"Forecast vs Actual (Test) — MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")
        plt.legend()
        plot_path = os.path.join(RUN_DIR, f"forecast_vs_actual_{RUN_TIME}.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Forecast vs Actual plot saved -> {plot_path}")
    except Exception as e:
        logger.exception(f"Failed to save Forecast vs Actual plot: {e}")

        
    # Task 7 — Feature importance
    logger.info("Starting Task 7: Feature importance analysis")
    try:
        importances = model.feature_importances_
        feat_names = X_train.columns
        importance_series = pd.Series(importances, index=feat_names).sort_values(ascending=False)

        with open(FEATURE_IMPORTANCE_TXT, "w", encoding="utf-8") as f:
            f.write("TASK 7 – FEATURE IMPORTANCE\n")
            f.write("=" * 50 + "\n\n")
            f.write("Feature importances (sorted):\n")
            for feat, imp in importance_series.items():
                f.write(f"{feat}: {imp:.6f}\n")

            f.write("\nTop 5 features:\n")
            for feat, imp in importance_series.head(5).items():
                f.write(f"- {feat}: {imp:.6f}\n")

            near = [f"y_lag_{i}" for i in range(1, 7)]
            far = [f"y_lag_{i}" for i in range(7, 25)]
            near_imp = importance_series.reindex(near).fillna(0).mean()
            far_imp = importance_series.reindex(far).fillna(0).mean()

            f.write("\nAverage importance:\n")
            f.write(f"- Near lags (1-6): {near_imp:.6f}\n")
            f.write(f"- Far  lags (7-24): {far_imp:.6f}\n")

            if near_imp > far_imp:
                f.write("\n=> Conclusion: Recent lags (near) are more important on average.\n")
            elif near_imp < far_imp:
                f.write("\n=> Conclusion: Distant lags (far) are more important on average.\n")
            else:
                f.write("\n=> Conclusion: Near and far lags have similar average importance.\n")

        logger.info(f"Feature importance saved -> {FEATURE_IMPORTANCE_TXT}")

        try:
            plt.figure(figsize=(12, 6))
            importance_series.plot(kind="bar", color="tab:green")
            plt.xlabel("Feature")
            plt.ylabel("Importance")
            plt.title("Feature Importance (lag features)")
            plt.tight_layout()
            fi_plot = os.path.join(RUN_DIR, f"feature_importance_{RUN_TIME}.png")
            plt.savefig(fi_plot)
            plt.close()
            logger.info(f"Feature importance plot saved -> {fi_plot}")
        except Exception as e:
            logger.exception(f"Failed to save feature importance plot: {e}")

    except Exception as e:
        logger.exception(f"Feature importance analysis failed: {e}")

    # Task 1 (advanced) — Rolling mean / std (post-hoc)
    logger.info("Starting Task 1 (advanced): rolling mean/std analysis (post-hoc)")
    try:
        ROLL_WINDOWS = [3, 6]
        rolling_store = {}
        for w in ROLL_WINDOWS:
            roll_mean = df.iloc[:, 0].shift(1).rolling(window=w, min_periods=1).mean()
            roll_std = df.iloc[:, 0].shift(1).rolling(window=w, min_periods=1).std()
            rolling_store[f"roll_mean_{w}"] = roll_mean
            rolling_store[f"roll_std_{w}"] = roll_std

        rolling_df = pd.DataFrame(index=supervised_df.index)
        for name, series in rolling_store.items():
            # align rolling series to supervised_df index robustly (handles duplicate timestamps)
            vals = [series.get(ts, float('nan')) for ts in supervised_df.index]
            rolling_df[name] = vals

        before_rows = len(rolling_df)
        rolling_df.dropna(inplace=True)
        after_rows = len(rolling_df)

        with open(ROLLING_STATS_TXT, "w", encoding="utf-8") as f:
            f.write("TASK 1 (ADVANCED) – ROLLING MEAN / STD (POST-HOC)\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Rolling windows used: {ROLL_WINDOWS}\n")
            f.write(f"Rows aligned to supervised_df: {before_rows}\n")
            f.write(f"Rows after dropping NaNs: {after_rows}\n\n")
            f.write("Added rolling features (in rolling_df):\n")
            for w in ROLL_WINDOWS:
                f.write(f"- roll_mean_{w}\n")
                f.write(f"- roll_std_{w}\n")

            if before_rows - after_rows > 0:
                f.write(f"\nRows dropped due to NaNs after alignment: {before_rows - after_rows}\n")
            else:
                f.write("\nNo rows dropped after alignment.\n")

        logger.info(f"Rolling stats saved -> {ROLLING_STATS_TXT}")
        rolling_csv = os.path.join(RUN_DIR, f"rolling_features_sample_{RUN_TIME}.csv")
        rolling_df.head(200).to_csv(rolling_csv)
        logger.info(f"Rolling features sample saved -> {rolling_csv}")

    except Exception as e:
        logger.exception(f"Rolling features (advanced) failed: {e}")

    logger.info("=========== END XGBOOST PIPELINE ===========")

    best_params, best_mae = tune_xgboost_timeseries(X_train, y_train)

    with open(TUNING_TXT, "w", encoding="utf-8") as f:
        f.write("XGBOOST HYPERPARAMETER TUNING (TimeSeriesSplit)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Best params: {best_params}\n")
        f.write(f"Validation MAE: {best_mae:.4f}\n")

    logger.info("XGBoost hyperparameter tuning completed")


if __name__ == "__main__":
    main()
