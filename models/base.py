import torch 
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseAutoEncoder(nn.Module, ABC):
    """
    Common interface for all AutoEncoder models.

    inputs:
        batch,
        time,
        features

    outputs:
        same shape as input
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def get_config(self) -> dict:
        pass
