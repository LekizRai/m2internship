import torch

from abc import ABC
from dataclasses import dataclass


@dataclass
class Config(ABC):
    device: torch.device = torch.device("cpu")
    dataset_path: str = "."
