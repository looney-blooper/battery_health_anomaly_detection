"""
Initial On-Device Personalization (Month 1)

- Fine-tunes ENCODER ONLY
- Decoder is frozen
- Uses only Month-1 data
- Low LR, few epochs
- Derives from MLflow Production model
"""

import torch
import json
from pathlib import Path
from datetime import datetime

from models.factory import build_model
from data_preprocessing.load_and_clean import load_and_clean
from data_preprocessing.extract_window import extract_lstm_windows


# =========================
# CONFIG (LOCKED)
# =========================
BATTERY_CSV = "data/raw/regular_alt_batteries/battery10.csv"

BASE_MODEL_PATH = Path("edge_models/base_model.pt")
PERSONALIZED_MODEL_PATH = Path("edge_models/personalized_user.pt")
METADATA_PATH = Path("edge_models/personalization_metadata.json")

MODEL_TYPE = "lstm_2layer"   # must match MLflow Production model
WINDOW_SIZE = 60
N_FEATURES = 4

EPOCHS = 3
LEARNING_RATE = 1e-4
BATCH_SIZE = 32


# =========================
# LOAD MONTH-1 DATA
# =========================
df = load_and_clean(BATTERY_CSV)
windows = extract_lstm_windows(df)
X = torch.tensor(windows, dtype=torch.float32)


# =========================
# LOAD BASE MODEL (FROM MLFLOW SYNC)
# =========================
assert BASE_MODEL_PATH.exists(), "Base model not synced from MLflow"

model = build_model(
    model_type=MODEL_TYPE,
    window_size=WINDOW_SIZE,
    n_features=N_FEATURES
)

model.load_state_dict(torch.load(BASE_MODEL_PATH, map_location="cpu"))
model.train()


# =========================
# FREEZE DECODER (CRITICAL)
# =========================
for param in model.decoder.parameters():
    param.requires_grad = False


# =========================
# OPTIMIZER (ENCODER ONLY)
# =========================
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE
)

criterion = torch.nn.MSELoss()


# =========================
# FINE-TUNING LOOP
# =========================
loader = torch.utils.data.DataLoader(
    X, batch_size=BATCH_SIZE, shuffle=True
)

for epoch in range(EPOCHS):
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        recon = model(batch)
        loss = criterion(recon, batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.6f}")


# =========================
# SAVE PERSONALIZED MODEL
# =========================
PERSONALIZED_MODEL_PATH.parent.mkdir(exist_ok=True)
torch.save(model.state_dict(), PERSONALIZED_MODEL_PATH)


# =========================
# SAVE METADATA (NO RAW DATA)
# =========================
metadata = {
    "base_model_source": "mlflow_production",
    "model_type": MODEL_TYPE,
    "personalization_date": datetime.utcnow().isoformat(),
    "epochs": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "data_source": BATTERY_CSV
}

with open(METADATA_PATH, "w") as f:
    json.dump(metadata, f, indent=2)

print("Personalization complete.")
print(f"Saved personalized model to: {PERSONALIZED_MODEL_PATH}")
