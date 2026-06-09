import torch

from abc import ABC
from typing import Dict
from dataclasses import dataclass, field


@dataclass
class Config(ABC):
    device: str = "cpu"
    dataset_path: str = "."
