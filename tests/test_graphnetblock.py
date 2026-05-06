from typing import Dict

import torch

from models.graphnetblock import GraphNetBlock
from models.tacgraspnet.tacgraspnet_config import TacGraspNetConfig


def test_graphnetblock():
    # Define sample data
    V = 5 # Number of nodes
    E_MESH = 18 # Number of mesh edges
    E_CONTACT = 2 # Number of contact edges (or world edges)
    T = 2 # Number of tetrahedra

    # Dimension of node features
    # 6 = node velocity (3) + number of node types (3)
    V_FEATURE = 6

    # Dimension of mesh edge features. There are two options
    # 4 = relative displacement in template (rest object) (3) + its norm (1)
    E_MESH_FEATURE = 4
    # # 8 = relative displacement in first and second frame (6) + their norm (2)
    # E_MESH_FEATURE = 8

    # Dimension of contact edge (world edge) features
    # 5 = relative displacement in current frame (3) + its norm (1) + applied force in contact (1)
    E_CONTACT_FEATURE = 5

    # Dimension of tetrahedral features
    # 1 = stress (1)
    T_FEATURE = 1
    batch = {
        "nodes.features": torch.rand(V, V_FEATURE),
        "mesh_edges": torch.tensor(
            [[0, 1], [0, 2], [0, 3], [1, 2], [2, 3], [1, 3], [1, 4], [2, 4],[3, 4],
             [1, 0], [2, 0], [3, 0], [2, 1], [3, 2], [3, 1], [4, 1], [4, 2], [4, 3]]
        ).long(),
        "contact_edges": torch.tensor(
            [[0, 4], [4, 0]]
        ).long(),
        "mesh_edges.features": torch.rand(E_MESH, E_MESH_FEATURE),
        "contact_edges.features": torch.rand(E_CONTACT, E_CONTACT_FEATURE),
        "tetrahedra": torch.tensor(
            [[0, 1, 2, 3],
             [1, 2, 3, 4]]
        ),
        "tetrahedra.features": torch.rand(T, T_FEATURE),
    }

    # Test forward
    config = TacGraspNetConfig()
    model = GraphNetBlock(config)
    batch = model.forward(batch)
    assert isinstance(batch, Dict)
