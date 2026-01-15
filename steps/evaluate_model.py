from zenml import step
import torch
import numpy as np
import mlflow

from models.factory import get_model

@step
def evaluate_model(
    model_path: str,
    windows: np.ndarray,
    model_type: str,
    window_size: int,
    n_features: int,
) -> float:
    model = get_model(model_type, n_features=n_features, window_size=window_size)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    X = torch.tensor(windows, dtype=torch.float32)
    with torch.no_grad():
        recon = model(X)
        error = torch.mean((recon - X) ** 2).item()

    mlflow.log_metric("reconstruction_error", error)
    return error
