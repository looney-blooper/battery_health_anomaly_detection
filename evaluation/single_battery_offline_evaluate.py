"""
Single-Battery Offline Evaluation

Evaluates ONE battery pack as ONE device:
- weekly inference
- rolling monthly baseline (previous month)
- 🟢 / 🟡 / 🔴 decision logic
- 🔴 requires 2 consecutive weeks

Assumptions (LOCKED):
- one battery CSV = one laptop
- weekly inference
- monthly rolling baseline
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path

from training.model import LSTMAutoEncoder
from data_preprocessing.load_and_clean import load_and_clean
from data_preprocessing.extract_window import extract_lstm_windows


# =========================
# CONFIG (LOCKED)
# =========================
BATTERY_CSV = "data/raw/regular_alt_batteries/battery10.csv"
MODEL_PATH = "models/lstm_model_v1.pt"
OUTPUT_LOG = "evaluation/weekly_health_battery10.csv"

WINDOW_SIZE = 60
WEEKS_PER_MONTH = 4

GREEN_SIGMA = 1.0
RED_SIGMA = 3.0


# =========================
# LOAD MODEL
# =========================
model = LSTMAutoEncoder(n_features=4)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()


# =========================
# LOAD + PREPROCESS ONE BATTERY
# =========================
df = load_and_clean(BATTERY_CSV)
windows = extract_lstm_windows(df)

X = torch.tensor(windows, dtype=torch.float32)

with torch.no_grad():
    recon = model(X)
    window_errors = torch.mean((recon - X) ** 2, dim=(1, 2)).numpy()


# =========================
# SPLIT INTO WEEKS
# =========================
windows_per_week = len(window_errors) // max(1, len(window_errors) // 12)
weeks = np.array_split(window_errors, len(window_errors) // windows_per_week)


# =========================
# BASELINE STATE
# =========================
baseline_mean = None
baseline_std = None
baseline_p95 = None

previous_week_red = False
red_in_current_month = False

logs = []


# =========================
# MAIN EVALUATION LOOP
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
        decision = "🟢"  # warm-up month

    logs.append({
        "week_index": week_idx,
        "month_index": month_idx,
        "weekly_mean_error": weekly_mean,
        "weekly_p95_error": weekly_p95,
        "baseline_mean": baseline_mean,
        "baseline_p95": baseline_p95,
        "decision": decision
    })

    # =========================
    # MONTH BOUNDARY
    # =========================
    if (week_idx + 1) % WEEKS_PER_MONTH == 0:
        month_errors = np.concatenate(
            weeks[week_idx - WEEKS_PER_MONTH + 1 : week_idx + 1]
        )

        if not red_in_current_month:
            baseline_mean = np.mean(month_errors)
            baseline_std = np.std(month_errors)
            baseline_p95 = np.percentile(month_errors, 95)

        previous_week_red = False
        red_in_current_month = False


# =========================
# SAVE RESULTS
# =========================
Path("evaluation").mkdir(exist_ok=True)
pd.DataFrame(logs).to_csv(OUTPUT_LOG, index=False)

print("Single-battery offline evaluation complete.")
print(f"Saved to {OUTPUT_LOG}")
