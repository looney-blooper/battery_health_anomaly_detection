import torch 
import numpy as np
from model import LSTMAutoEncoder

X = torch.tensor(
    np.load("data/processed/train_normal.npy"),
    dtype=torch.float32
)

model = LSTMAutoEncoder(n_features=X.shape[2])
model.load_state_dict(torch.load("models/lstm_model_v1.pt"))
model.eval()

with torch.no_grad():
    recon = model(X)
    errors = torch.mean((recon - X) ** 2, dim=(1,2)).numpy()

threshold = errors.mean() + 3 * errors.std()
np.save("models/lstm_threshold.npy", threshold)

print("Threshold:", threshold)