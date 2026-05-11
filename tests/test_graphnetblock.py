from typing import Dict
import torch

from models.graphnetblock import GraphNetBlock
from models.tacgraspnet.tacgraspnet_config import TacGraspNetConfig


def test_graphnetblock():
    # Define sample data
    V = 8 # Number of nodes
    E_MESH = 24 # Number of mesh edges
    E_CONTACT = 2 # Number of contact edges (or world edges)
    T = 2 # Number of tetrahedra

    D_V = 128 # Dimension of (encoded) node features
    D_E_MESH = 128 # Dimension of (encoded) mesh edge features
    D_E_CONTACT = 128 # Dimension of (encoded) contact edge (world edge) features
    D_T = 128 # Dimension of (encoded) tetrahedral features

    batch = {
        "nodes.features": torch.rand(V, D_V),
        "mesh_edges": torch.tensor(
            [[0, 1], [1, 2], [2, 0], [0, 3], [1, 3], [2, 3], [4, 5], [5, 6], [4, 6], [4, 7], [5, 7], [6, 7],
             [1, 0], [2, 1], [0, 2], [3, 0], [3, 1], [3, 2], [5, 4], [6, 5], [6, 4], [7, 4], [7, 5], [7, 6]]
        ).long(),
        "contact_edges": torch.tensor(
            [[3, 7], [7, 3]]
        ).long(),
        "mesh_edges.features": torch.rand(E_MESH, D_E_MESH),
        "contact_edges.features": torch.rand(E_CONTACT, D_E_CONTACT),
        "tetrahedra": torch.tensor(
            [[0, 1, 2, 3],
             [4, 5, 6, 7]]
        ),
        "tetrahedra.features": torch.rand(T, D_T),
    }

    # Test forward
    config = TacGraspNetConfig()
    model = GraphNetBlock(config)
    batch = model.forward(batch)
    assert isinstance(batch, Dict)
    assert batch["nodes.features"].shape == (V, D_V)
    assert batch["mesh_edges.features"].shape == (E_MESH, D_E_MESH)
    assert batch["contact_edges.features"].shape == (E_CONTACT, D_E_CONTACT)
    assert batch["tetrahedra.features"].shape == (T, D_T)
