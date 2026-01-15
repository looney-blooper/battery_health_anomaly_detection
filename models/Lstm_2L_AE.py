import torch
import torch.nn as nn
from models.base import BaseAutoEncoder


class LSTM2LayerAutoencoder(BaseAutoEncoder):
    """
    LSTM-based Autoencoder with 2 layers.
    """
    def __init__(self, n_features: int, hidden_dim: int = 64):
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim

        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=n_features,
            batch_first=True
        )

    def forward(self, x):
        z, _ = self.encoder(x)
        recon, _ = self.decoder(z)
        return recon

    def get_name(self):
        return "LSTM2LayerAutoencoder"

    def get_config(self):
        return {
            "n_features": self.n_features,
            "hidden_dim": self.hidden_dim
        }
