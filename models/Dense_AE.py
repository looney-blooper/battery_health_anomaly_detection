import torch
import torch.nn as nn

from base import BaseAutoEncoder

class DenseAutoEncoder(BaseAutoEncoder):
    """
    A Dense (Fully Connected) AutoEncoder for time-series data.
    """

    def __init__(self, window_size: int , n_features: int, latent_dim: int = 64):
        super().__init__()
        self.window_size = window_size
        self.n_features = n_features
        
        input_dims = window_size * n_features

        self.encoder = nn.Sequential(
            nn.Linear(input_dims, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dims),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, f = x.shape
        x_flat = x.view(b, -1)
        z = self.encoder(x_flat)
        recon = self.decoder(z)
        return recon.view(b, t, f)

    def get_name(self) -> str:
        return "DenseAutoEncoder"

    def get_config(self) -> dict:
        return {
            "window_size": self.window_size,
            "n_features": self.n_features,
            "latent_dim": self.encoder[-1].out_features,
        }      
        