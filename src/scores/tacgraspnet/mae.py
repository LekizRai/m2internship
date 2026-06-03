import torch

from torch import nn

from commons.datatype import Databatch, NodeType
from models.tacgraspnet.tacgraspnet_config import TacGraspNetConfig


class DisplacementMAE(nn.Module):
    def __init__(self, config: TacGraspNetConfig):
        super().__init__()
        self._config = config

    @staticmethod
    def forward(batch: Databatch) -> torch.Tensor:
        # Extract only target and predicted displacements from tactile sensor nodes
        target_disps = batch["vertices.positions"][batch["nodes.types"] != NodeType.OBJECT]
        pred_disps = batch["predictions.vertices.positions"][batch["nodes.types"] != NodeType.OBJECT]
        disp_l1_error = torch.norm(pred_disps - target_disps, p=1, dim=-1)
        return torch.mean(disp_l1_error)

    def __str__(self) -> str:
        return "displacement_mae"


class StressMAE(nn.Module):
    def __init__(self, config: TacGraspNetConfig):
        super().__init__()
        self._config = config

    def forward(self, batch: Databatch) -> torch.Tensor:
        if self._config.use_node_tetra_separate_decoders: # Check if separate decoders for nodes and tetrahedra are used
            # Extract target and predicted stresses from tetrahedra
            target_stresses = batch["targets.tetrahedra.normalized_outputs"]
            pred_stresses = batch["tetrahedra.outputs"]
        else: # Otherwise
            # Extract only target and predicted outputs from tactile sensor nodes
            target = batch["targets.nodes.normalized_outputs"]
            pred = batch["nodes.outputs"][batch["nodes.types"] != NodeType.OBJECT]
            _, target_stresses = torch.split(target, [3, 1], dim=-1)
            _, pred_stresses = torch.split(pred, [3, 1], dim=-1)

        stress_l1_error = torch.norm(pred_stresses - target_stresses, p=1, dim=-1)
        return torch.mean(stress_l1_error)

    def __str__(self) -> str:
        return "stress_mae"
