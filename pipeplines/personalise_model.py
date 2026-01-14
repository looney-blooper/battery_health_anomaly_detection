"""
Initial On-Device Personalization (Month 1)

- Fine-tunes ENCODER ONLY
- Decoder is frozen
- Uses only Month-1 data
- Low LR, few epochs
"""

import torch
import numpy as np
from pathlib import Path
from datetime import datetime

from training.model import LSTMAutoEncoder
from data_preprocessing.load_and_clean import load_and_clean
from data_preprocessing.extract_window import extract_lstm_windows

# =========================
# CONFIG (LOCKED)
# =========================
BATTERY_CSV = "data/raw/regular_alt_batteries/battery10.csv"
BASE_MODEL_PATH = "models/lstm_model_v1.pt"

PERSONALIZED_MODEL_PATH = "models/lstm_personalized_user.pt"
METADATA_PATH = "models/personalization_metadata.json"

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
# LOAD MODEL
# =========================
model = LSTMAutoEncoder(n_features=4)
model.load_state_dict(torch.load(BASE_MODEL_PATH))
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
Path("models").mkdir(exist_ok=True)
torch.save(model.state_dict(), PERSONALIZED_MODEL_PATH)


# =========================
# SAVE METADATA (NO RAW DATA)
# =========================
metadata = {
    "base_model": BASE_MODEL_PATH,
    "personalized_model": PERSONALIZED_MODEL_PATH,
    "personalization_date": datetime.utcnow().isoformat(),
    "epochs": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "data_source": BATTERY_CSV
}

import json
with open(METADATA_PATH, "w") as f:
    json.dump(metadata, f, indent=2)

print("Personalization complete.")
print(f"Saved personalized model to: {PERSONALIZED_MODEL_PATH}")
