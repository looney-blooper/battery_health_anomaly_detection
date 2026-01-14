import torch 
import torch.nn as nn

from base import BaseAutoEncoder

class LSTMDeepAutoEncoder(BaseAutoEncoder):
    """
    Docstring for LSTMDeepAutoEncoder
    """

    def __init__(self, n_features: int, hidden_dims: int = 128, num_layers : int = 3):
        super().__init__()
        self.n_features = n_features
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dims,
            num_layers=num_layers,
            batch_first=True
        )

        self.decoder = nn.LSTM(
            input_size=hidden_dims,
            hidden_size=n_features,
            num_layers=num_layers,
            batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, _ = self.encoder(x)
        recon, _ = self.decoder(z)
        return recon
    
    def get_name(self) -> str:
        return "LSTMDeepAutoEncoder"

    def get_config(self) -> dict:
        return {
            "n_features": self.n_features,
            "hidden_dims": self.hidden_dims,
            "num_layers": self.num_layers
        }
