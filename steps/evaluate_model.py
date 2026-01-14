from zenml import step
import torch
import numpy as np
import mlflow

@step
def evaluate_model(model, windows: np.ndarray) -> float:
    """
    Computes mean reconstruction error.
    """
    model.eval()
    X = torch.tensor(windows, dtype=torch.float32)

    with torch.no_grad():
        recon = model(X)
        error = torch.mean((recon - X) ** 2).item()

    mlflow.log_metric("reconstruction_error", error)
    return error
