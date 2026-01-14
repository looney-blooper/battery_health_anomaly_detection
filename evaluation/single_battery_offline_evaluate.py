"""
⚠️ REFERENCE / DEBUG SCRIPT

Mirrors the offline evaluation logic used in ZenML.
Not used for model promotion.
"""

import pandas as pd
import torch
from pathlib import Path

from models.factory import build_model
from data_preprocessing.load_and_clean import load_and_clean
from data_preprocessing.extract_window import extract_lstm_windows
from evaluation.core_evaluator import evaluate_single_battery


# =========================
# CONFIG
# =========================
BATTERY_CSV = "data/raw/regular_alt_batteries/battery10.csv"
MODEL_PATH = "models/lstm_model_v1.pt"

MODEL_TYPE = "lstm_2layer"
WINDOW_SIZE = 60
N_FEATURES = 4

OUTPUT_LOG = "evaluation/weekly_health_battery10.csv"


# =========================
# LOAD MODEL
# =========================
model = build_model(
    model_type=MODEL_TYPE,
    window_size=WINDOW_SIZE,
    n_features=N_FEATURES
)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()


# =========================
# LOAD DATA
# =========================
df = load_and_clean(BATTERY_CSV)
windows = extract_lstm_windows(df)
X = torch.tensor(windows, dtype=torch.float32)


with torch.no_grad():
    recon = model(X)
    errors = torch.mean((recon - X) ** 2, dim=(1, 2)).numpy()


# =========================
# CORE EVALUATION
# =========================
logs = evaluate_single_battery(errors)


# =========================
# SAVE
# =========================
Path("evaluation").mkdir(exist_ok=True)
pd.DataFrame(logs).to_csv(OUTPUT_LOG, index=False)

print("Offline evaluation complete.")
print(f"Saved to {OUTPUT_LOG}")
