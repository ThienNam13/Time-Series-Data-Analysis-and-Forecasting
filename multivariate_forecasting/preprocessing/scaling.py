# preprocessing/scaling.py

import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler


def split_and_scale_time_series(
    data: pd.DataFrame,
    features: list,
    train_ratio=0.7,
    val_ratio=0.15,
    scaler_save_path=None
):
    """
    Split time series data by time order and scale using StandardScaler
    (fit ONLY on train set to avoid data leakage)

    Returns:
        train_scaled, val_scaled, test_scaled, scaler
    """

    # =========================
    # 1. TIME-BASED SPLIT
    # =========================
    n = len(data)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)

    train_raw = data.iloc[:train_size][features]
    val_raw = data.iloc[train_size:train_size + val_size][features]
    test_raw = data.iloc[train_size + val_size:][features]

    # =========================
    # 2. FIT SCALER ON TRAIN ONLY
    # =========================
    scaler = StandardScaler()
    scaler.fit(train_raw)

    # =========================
    # 3. TRANSFORM ALL SETS
    # =========================
    train_scaled = pd.DataFrame(
        scaler.transform(train_raw),
        columns=features,
        index=train_raw.index
    )

    val_scaled = pd.DataFrame(
        scaler.transform(val_raw),
        columns=features,
        index=val_raw.index
    )

    test_scaled = pd.DataFrame(
        scaler.transform(test_raw),
        columns=features,
        index=test_raw.index
    )

    # =========================
    # 4. SAVE SCALER (OPTIONAL)
    # =========================
    if scaler_save_path:
        os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)
        joblib.dump(scaler, scaler_save_path)

    return train_scaled, val_scaled, test_scaled, scaler


# =====================================================
# INVERSE ONLY LOAD VARIABLE (FOR METRICS)
# =====================================================
def inverse_scale_load(scaler, load_scaled, n_features=4):
    """
    Inverse transform only the 'load' variable (assumed to be column 0)

    load_scaled: array-like (n_samples,)
    """

    import numpy as np

    dummy = np.zeros((len(load_scaled), n_features))
    dummy[:, 0] = load_scaled

    inv = scaler.inverse_transform(dummy)
    return inv[:, 0]
