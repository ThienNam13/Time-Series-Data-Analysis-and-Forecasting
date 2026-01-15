import pandas as pd
import logging
import os
import joblib
from sklearn.preprocessing import StandardScaler

# ===================== CONFIG =====================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")

INPUT_FILE = os.path.join(LOG_DIR, "feature_engineered_data.csv")
OUTPUT_FILE = os.path.join(LOG_DIR, "scaled_data_lstm.csv")
SCALER_FILE = os.path.join(LOG_DIR, "standard_scaler.save")

TARGET_COL = "load"

# ===================== LOGGING =====================
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "scaling.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ===================== MAIN =====================
def main():
    logging.info("START SCALING PROCESS")

    # Load feature-engineered data
    df = pd.read_csv(INPUT_FILE, index_col=0)

    logging.info(f"Original shape: {df.shape}")

    # Separate target & features
    y = df[[TARGET_COL]]
    X = df.drop(columns=[TARGET_COL])

    # Scaling (for DL)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    df_scaled = pd.DataFrame(
        X_scaled,
        columns=X.columns,
        index=df.index
    )

    # Add target back (NOT scaled)
    df_scaled[TARGET_COL] = y

    # Save outputs
    df_scaled.to_csv(OUTPUT_FILE)
    joblib.dump(scaler, SCALER_FILE)

    logging.info(f"Scaled data saved to: {OUTPUT_FILE}")
    logging.info(f"Scaler saved to: {SCALER_FILE}")
    logging.info("SCALING COMPLETED")

if __name__ == "__main__":
    main()