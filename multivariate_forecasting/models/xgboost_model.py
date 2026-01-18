# models/xgboost_model.py

import os
import numpy as np
import xgboost as xgb

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


# =====================================================
# INVERSE LOAD ONLY
# =====================================================
def inverse_scale_load(scaler, load_scaled, n_features):
    dummy = np.zeros((len(load_scaled), n_features))
    dummy[:, 0] = load_scaled
    inv = scaler.inverse_transform(dummy)
    return inv[:, 0]


# =====================================================
# TRAIN + EVALUATE XGBOOST (WITH EARLY STOPPING)
# =====================================================
def train_xgboost(
    train_df,
    val_df,
    test_df,
    scaler,
    log_dir,
    target_col="load",
    max_depth=6,
    n_estimators=500,
    learning_rate=0.05
):
    """
    All data must be:
        - scaled
        - feature engineered (lag + rolling)
    Early stopping is applied on validation set.
    """

    # =========================
    # SPLIT FEATURES / TARGET
    # =========================
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    X_val = val_df.drop(columns=[target_col])
    y_val = val_df[target_col]

    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    # =========================
    # DMATRIX (XGBOOST NATIVE)
    # =========================
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test)

    # =========================
    # PARAMS
    # =========================
    params = {
        "objective": "reg:squarederror",
        "max_depth": max_depth,
        "eta": learning_rate,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "rmse"
    }

    # =========================
    # TRAIN WITH EARLY STOPPING
    # =========================
    evals = [(dtrain, "train"), (dval, "val")]

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=n_estimators,
        evals=evals,
        early_stopping_rounds=30,
        verbose_eval=False
    )

    # =========================
    # PREDICT (SCALED)
    # =========================
    y_pred_scaled = model.predict(dtest)

    # =========================
    # INVERSE SCALE LOAD ONLY
    # =========================
    n_features = scaler.mean_.shape[0]

    y_test_inv = inverse_scale_load(scaler, y_test.values, n_features)
    y_pred_inv = inverse_scale_load(scaler, y_pred_scaled, n_features)

    # =========================
    # METRICS (ORIGINAL SCALE)
    # =========================
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    mape = mean_absolute_percentage_error(y_test_inv, y_pred_inv) * 100

    # =========================
    # SAVE MODEL
    # =========================
    model_path = os.path.join(log_dir, "xgboost_model.json")
    model.save_model(model_path)

    # =========================
    # SAVE RESULTS
    # =========================
    result_path = os.path.join(log_dir, "xgboost_results.txt")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write("=== XGBOOST RESULTS (ORIGINAL SCALE) ===\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"MAE : {mae:.4f}\n")
        f.write(f"MAPE: {mape:.2f}%\n")

    return y_pred_inv, rmse, mae, mape
