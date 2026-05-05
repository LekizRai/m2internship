import torch
from torch import nn

from models.mlp import MLP


class TacGraspNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
