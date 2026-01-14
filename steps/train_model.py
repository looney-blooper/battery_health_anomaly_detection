from zenml import step
import numpy as np
import torch
import mlflow 

from models.factory import get_model

@step
def train_model(
    windows: np.ndarray,
    model_type: str,
    window_size: int,
    n_features: int,
    epochs: int = 50,
    lr: float = 1e-3,
):
    """
    Trains a selected AutoEncoder model on the data

    inputs:
    - windows: np.ndarray
        Preprocessed LSTM windows
    - model_type: str
        Type of model to train ("LSTMAutoEncoder" or "LSTMDeepAutoEncoder" or "DenseAutoEncoder")
    - window_size: int
        Size of each LSTM window
    - n_features: int
        Number of features in the input data
    - epochs: int, default=50
        Number of training epochs
    - lr: float, default=1e-3
        Learning rate for the optimizer

    outputs:
    - model: torch.nn.Module
        Trained AutoEncoder model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(model_type, n_features=n_features, window_size=window_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    X = torch.tensor(windows, dtype=torch.float32).to(device)

    mlflow.log_param("model_type", model_type)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("learning_rate", lr)

    print(f"Using device: {device} for : ")
    print(f"Training {model_type} model for {epochs} epochs...")
    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()
        recon = model(X)
        loss = criterion(recon, X)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.6f}")
            mlflow.log_metric("training_loss", loss.item(), step=epoch+1)
        
    return model