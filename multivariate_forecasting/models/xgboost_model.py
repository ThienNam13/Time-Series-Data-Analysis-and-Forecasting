import os
import numpy as np
import xgboost as xgb
import joblib

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error
)


def prepare_xy(df, target_col="load"):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def train_xgboost(
    train_df,
    val_df,
    test_df,
    log_dir,
    max_depth=6,
    n_estimators=500,
    learning_rate=0.05,
    early_stopping_rounds=20
):
    """
    Train XGBoost model using xgb.train (compatible with old versions)
    """

    # ===================== DATA =====================
    X_train, y_train = prepare_xy(train_df)
    X_val, y_val = prepare_xy(val_df)
    X_test, y_test = prepare_xy(test_df)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # ===================== PARAMS =====================
    params = {
        "objective": "reg:squarederror",
        "max_depth": max_depth,
        "eta": learning_rate,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42
    }

    evals = [(dtrain, "train"), (dval, "val")]

    # ===================== TRAIN =====================
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=n_estimators,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=True
    )

    # ===================== PREDICT =====================
    y_pred = model.predict(dtest)

    # ===================== METRICS =====================
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100

    # ===================== SAVE MODEL =====================
    model_path = os.path.join(log_dir, "xgboost_model.json")
    model.save_model(model_path)

    # ===================== SAVE RESULTS =====================
    result_path = os.path.join(log_dir, "xgboost_results.txt")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write("=== XGBOOST RESULTS ===\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"MAE: {mae:.4f}\n")
        f.write(f"MAPE: {mape:.2f}%\n")
        f.write(f"Best iteration: {model.best_iteration}\n")

    return y_pred, rmse, mae, mape
