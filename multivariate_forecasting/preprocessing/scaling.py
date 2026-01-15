from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib
import os

def scale_features(
    df,
    save_path,
    log_file=None
):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    df_scaled = pd.DataFrame(
        df_scaled,
        columns=df.columns,
        index=df.index
    )

    joblib.dump(scaler, save_path)

    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write("=== SCALING ===\n")
            f.write("Scaler: StandardScaler\n")
            f.write(f"Số feature sau scaling: {df.shape[1]}\n")
            f.write(f"Đã lưu scaler tại: {save_path}\n\n")

    return df_scaled