from typing import Dict
import torch

from models.tacgraspnet.tacgraspnet import TacGraspNet
from models.tacgraspnet.tacgraspnet_config import TacGraspNetConfig


def test_tacgraspnet():
    # Define sample data
    V = 8 # Number of nodes
    E_MESH = 24 # Number of mesh edges
    E_CONTACT = 2 # Number of contact edges (or world edges)
    T = 2 # Number of tetrahedra

    # Dimension of node features
    # 6 = node velocity (3) + number of node types (3)
    D_V = 6

    # Dimension of mesh edge features. There are two options
    # 4 = relative displacement in template (at-rest object) (3) + its norm (1)
    D_E_MESH = 4
    # # 8 = relative displacement in first and second frame (6) + their norm (2)
    # E_MESH_FEATURE = 8

    # Dimension of contact edge (world edge) features
    # 5 = relative displacement in current frame (3) + its norm (1) + applied force in contact (1)
    D_E_CONTACT = 5

    # Dimension of tetrahedral features
    # 1 = stress (1)
    D_T = 1

    # Dimension of outputs
    D_V_OUTPUT = 3 # Displacement (3)
    D_T_OUTPUT = 1 # Stress (1)

    batch = {
        "nodes.features": torch.rand(V, D_V),
        "mesh_edges": torch.tensor(
            [[1, 0], [2, 1], [0, 2], [3, 0], [3, 1], [3, 2], [5, 4], [6, 5], [6, 4], [7, 4], [7, 5], [7, 6],
             [0, 1], [1, 2], [2, 0], [0, 3], [1, 3], [2, 3], [4, 5], [5, 6], [4, 6], [4, 7], [5, 7], [6, 7]]
        ).long(),
        "contact_edges": torch.tensor(
            [[7, 3], [3, 7]]
        ).long(),
        "mesh_edges.features": torch.rand(E_MESH, D_E_MESH),
        "contact_edges.features": torch.rand(E_CONTACT, D_E_CONTACT),
        "tetrahedra": torch.tensor(
            [[4, 5, 6, 7],
             [0, 1, 2, 3]]
        ),
        "tetrahedra.features": torch.rand(T, D_T),
    }

    # Test forward
    config = TacGraspNetConfig()
    model = TacGraspNet(config)
    batch = model.forward(batch)
    assert isinstance(batch, Dict)
    assert batch["nodes.outputs"].shape == (V, D_V_OUTPUT)
    assert batch["tetrahedra.outputs"].shape == (T, D_T_OUTPUT)

if __name__ == "__main__":
    test_tacgraspnet()
    print("Test passed.")