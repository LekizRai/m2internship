import torch
from torch import nn

from models.tacgraspnet.tacgraspnet_config import TacGraspNetConfig
from models.mlp import MLP


class GraphNetBlock(nn.Module):
    """Module to do message passing"""

    def __init__(
            self,
            config: TacGraspNetConfig,
    ):
        super().__init__()
        self._config = config

        # Initialize MLPs for all types of edge (mesh, contact, ...) feature update
        self._edge_mlps = {}

        # Initialize MLP for node feature update
        self._node_mlp = None

        # Initialize MLP for tetrahedral feature update
        self._tetra_mlp = None

    def _update_node(self, x):
        pass

    def _update_edge(self, x):
        pass

    def forward(self, x):
        return x
