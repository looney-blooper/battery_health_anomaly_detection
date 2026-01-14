# Simulation of the On-Device Battery Health Auditing System

"""
Offline Evaluation for On-Device Battery Health Auditing System

Simulates:
- weekly inference
- rolling monthly baseline
- 3-state health decision logic (🟢 🟡 🔴)

Assumptions (LOCKED):
- Single-device timeline
- Weekly inference
- Monthly rolling baseline (previous month)
- 🔴 requires 2 consecutive weeks
- Baseline updates only if no 🔴 in the month
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path

from training.model import LSTMAutoEncoder


# =========================
# CONFIG (LOCKED)
# =========================
WINDOWS_PATH = "data/processed/train_normal.npy"
MODEL_PATH = "models/lstm_model_v1.pt"
OUTPUT_LOG = "evaluation/weekly_health_log.csv"

WEEKS_PER_MONTH = 4

GREEN_SIGMA = 1.0
RED_SIGMA = 3.0


# =========================
# LOAD MODEL & DATA
# =========================
X = torch.tensor(np.load(WINDOWS_PATH), dtype=torch.float32)

model = LSTMAutoEncoder(n_features=X.shape[2])
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

with torch.no_grad():
    recon = model(X)
    window_errors = torch.mean((recon - X) ** 2, dim=(1, 2)).numpy()


# =========================
# TIME SIMULATION
# =========================
total_windows = len(window_errors)
windows_per_week = total_windows // (total_windows // WEEKS_PER_MONTH * WEEKS_PER_MONTH)

weeks = np.array_split(window_errors, total_windows // windows_per_week)


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
        # RED condition
        is_red = weekly_mean > baseline_mean + RED_SIGMA * baseline_std

        # YELLOW condition
        is_yellow = weekly_mean > baseline_mean + GREEN_SIGMA * baseline_std

        if is_red and previous_week_red:
            decision = "🔴"
            red_in_current_month = True
        elif is_red:
            decision = "🟡"  # first red-like spike, wait for confirmation
        elif is_yellow:
            decision = "🟡"

        previous_week_red = is_red
    else:
        # Warm-up phase
        decision = "🟢"

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
    # MONTH BOUNDARY LOGIC
    # =========================
    if (week_idx + 1) % WEEKS_PER_MONTH == 0:
        month_errors = np.concatenate(
            weeks[week_idx - WEEKS_PER_MONTH + 1 : week_idx + 1]
        )

        if not red_in_current_month:
            baseline_mean = np.mean(month_errors)
            baseline_std = np.std(month_errors)
            baseline_p95 = np.percentile(month_errors, 95)

        # reset month state
        red_in_current_month = False
        previous_week_red = False


# =========================
# SAVE RESULTS
# =========================
Path("evaluation").mkdir(exist_ok=True)

df = pd.DataFrame(logs)
df.to_csv(OUTPUT_LOG, index=False)

print(f"Offline evaluation complete.")
print(f"Weekly health log saved to: {OUTPUT_LOG}")
