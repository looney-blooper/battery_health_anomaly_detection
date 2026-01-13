import os
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

from load_and_clean import load_and_clean
from extract_window import extract_lstm_windows

RAW_DIR = "data/raw/regular_alt_batteries"
OUT_DIR = "data/processed"

WINDOW_SIZE = 60  

os.makedirs(OUT_DIR, exist_ok=True)

# =============================
# CONFIG
# =============================

TRAIN_RATIO = 0.30    # early life = normal

def main():
    all_windows = []

    csv_files = glob.glob(os.path.join(RAW_DIR, "*.csv"))

    for csv in csv_files:
        print(f"Processing {os.path.basename(csv)}")
        df = load_and_clean(csv)

        if len(df) < WINDOW_SIZE * 2:
            continue

        windows = extract_lstm_windows(df)
        if len(windows) > 0:
            all_windows.append(windows)

    X_all = np.vstack(all_windows)
    print(f"Total windows: {X_all.shape}")

    # ----------------------------
    # TIME-BASED SPLIT
    # ----------------------------
    split_idx = int(len(X_all) * TRAIN_RATIO)
    X_train = X_all[:split_idx]
    X_drift = X_all[split_idx:]

    # ----------------------------
    # FEATURE-WISE SCALING
    # ----------------------------
    scaler = StandardScaler()

    # Fit ONLY on normal data (flatten time dimension)
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    scaler.fit(X_train_flat)

    # Apply scaling
    def scale(X):
        shape = X.shape
        X_flat = X.reshape(-1, shape[-1])
        X_scaled = scaler.transform(X_flat)
        return X_scaled.reshape(shape)

    X_train_scaled = scale(X_train)
    X_drift_scaled = scale(X_drift)

    # ----------------------------
    # SAVE OUTPUTS
    # ----------------------------
    np.save(os.path.join(OUT_DIR, "train_normal.npy"), X_train_scaled)
    np.save(os.path.join(OUT_DIR, "drift_data.npy"), X_drift_scaled)
    joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.joblib"))

    print("Preprocessing complete.")
    print("Train shape:", X_train_scaled.shape)
    print("Drift shape:", X_drift_scaled.shape)

if __name__ == "__main__":
    main()
