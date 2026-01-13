import torch as pt
import torch.nn as nn

class LSTMAutoEncoder(nn.Module):
    def __init__(self, n_features=4, hidden_size=32):
        super().__init__()

        self.encoder = nn.LSTM(
            input_size = n_features,
            hidden_size = hidden_size,
            batch_first = True,
        )

        self.decoder = nn.LSTM(
            input_size = hidden_size,
            hidden_size = n_features,
            batch_first = True,
        )

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        h = h.repeat(x.size(1), 1, 1).permute(1, 0, 2)
        recon, _ = self.decoder(h)
        return recon