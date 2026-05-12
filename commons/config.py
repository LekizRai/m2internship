from dataclasses import dataclass
import torch


@dataclass
class Config:
    device: torch.Device
