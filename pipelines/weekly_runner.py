"""
Weekly Runner with Shadow Mode

Runs weekly battery health evaluation.
In SHADOW_MODE:
- decisions are logged only
- no notifications
- no retraining
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime

from models.factory import get_model
from data_preprocessing.load_and_clean import load_and_clean
from data_preprocessing.extract_window import extract_lstm_windows

# =========================
# CONFIG (LOCKED)
# =========================

BATTERY_CSV = "data/raw/regular_alt_batteries/battery10.csv"

SHADOW_MODE = True

WINDOW_SIZE = 60
WEEKS_PER_MONTH = 4
GREEN_SIGMA = 1.0
RED_SIGMA = 3.0

LOG_PATH = "logs/weekly_shadow_log.csv"



BASE_MODEL_PATH = Path("edge_models/base_model.pt")
PERSONALIZED_MODEL_PATH = Path("edge_models/personalized_user.pt")

MODEL_TYPE = "LSTM2LayerAutoencoder"   # must match promoted model architecture
WINDOW_SIZE = 60
N_FEATURES = 4

# Choose weights
if PERSONALIZED_MODEL_PATH.exists():
    weights_path = PERSONALIZED_MODEL_PATH
    model_source = "personalized"
else:
    weights_path = BASE_MODEL_PATH
    model_source = "mlflow_production"

# Build architecture
model = get_model(
    model_type=MODEL_TYPE,
    window_size=WINDOW_SIZE,
    n_features=N_FEATURES
)

# Load weights
model.load_state_dict(torch.load(weights_path, map_location="cpu"))
model.eval()

print(f"Weekly runner using {model_source} model from {weights_path}")


# =========================
# LOAD DATA (ONE BATTERY)
# =========================
df = load_and_clean(BATTERY_CSV)
windows = extract_lstm_windows(df)

X = torch.tensor(windows, dtype=torch.float32)

with torch.no_grad():
    recon = model(X)
    errors = torch.mean((recon - X) ** 2, dim=(1, 2)).numpy()


# =========================
# WEEKLY AGGREGATION
# =========================
windows_per_week = len(errors) // max(1, len(errors) // 12)
weeks = np.array_split(errors, len(errors) // windows_per_week)


# =========================
# BASELINE STATE
# =========================
baseline_mean = None
baseline_std = None
previous_week_red = False
red_in_current_month = False

logs = []


# =========================
# WEEKLY LOOP
# =========================
for week_idx, week_errors in enumerate(weeks):
    month_idx = week_idx // WEEKS_PER_MONTH

    weekly_mean = np.mean(week_errors)
    weekly_p95 = np.percentile(week_errors, 95)

    decision = "🟢"

    if baseline_mean is not None:
        is_red = weekly_mean > baseline_mean + RED_SIGMA * baseline_std
        is_yellow = weekly_mean > baseline_mean + GREEN_SIGMA * baseline_std

        if is_red and previous_week_red:
            decision = "🔴"
            red_in_current_month = True
        elif is_red or is_yellow:
            decision = "🟡"

        previous_week_red = is_red
    else:
        decision = "🟢"  # warm-up

    logs.append({
        "timestamp": datetime.utcnow().isoformat(),
        "week_index": week_idx,
        "month_index": month_idx,
        "weekly_mean_error": weekly_mean,
        "weekly_p95_error": weekly_p95,
        "decision": decision,
        "shadow_mode": SHADOW_MODE
    })

    # -------------------------
    # MONTH BOUNDARY
    # -------------------------
    if (week_idx + 1) % WEEKS_PER_MONTH == 0:
        month_errors = np.concatenate(
            weeks[week_idx - WEEKS_PER_MONTH + 1 : week_idx + 1]
        )

        if not red_in_current_month:
            baseline_mean = np.mean(month_errors)
            baseline_std = np.std(month_errors)

        previous_week_red = False
        red_in_current_month = False

    # -------------------------
    # SHADOW GATE
    # -------------------------
    if decision == "🔴":
        if SHADOW_MODE:
            # no notification, no retraining
            pass
        else:
            # future: notify user, trigger workflows
            pass


# =========================
# SAVE SHADOW LOG
# =========================
Path("logs").mkdir(exist_ok=True)

df_log = pd.DataFrame(logs)

if Path(LOG_PATH).exists():
    df_log.to_csv(LOG_PATH, mode="a", header=False, index=False)
else:
    df_log.to_csv(LOG_PATH, index=False)

print("Weekly runner completed.")
print(f"Shadow log updated at: {LOG_PATH}")
