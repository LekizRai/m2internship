import torch

from abc import ABC
from typing import Dict
from dataclasses import dataclass, field


@dataclass
class Config(ABC):
    device: torch.device = torch.device("cpu")
    dataset_path: str = "."
