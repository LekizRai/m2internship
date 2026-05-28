import torch

from torch import nn
from torchmetrics import R2Score

from commons.datatype import Databatch, NodeType
from models.tacgraspnet.tacgraspnet_config import TacGraspNetConfig


class DisplacementR2PerSample(nn.Module):
    def __init__(self, config: TacGraspNetConfig):
        super().__init__()
        self._config = config
        self._r2_score_fn = R2Score().to(config.device)

    def forward(self, batch: Databatch) -> torch.Tensor:
        if self._config.use_node_tetra_separate_decoders: # Check if separate decoders for nodes and tetrahedra are used
            # Extract only target and predicted displacements from tactile sensor nodes
            target_disps = batch["targets.tactile_sensors.nodes.normalized_outputs"]
            pred_disps = batch["nodes.outputs"][batch["nodes.types"] != NodeType.OBJECT]
        else: # Otherwise
            # Extract only target and predicted outputs from tactile sensor nodes
            target = batch["targets.tactile_sensors.nodes.normalized_outputs"]
            pred = batch["nodes.outputs"][batch["nodes.types"] != NodeType.OBJECT]
            target_disps, _ = torch.split(target, [3, 1], dim=-1)
            pred_disps, _ = torch.split(pred, [3, 1], dim=-1)

        self._r2_score_fn.reset()
        self._r2_score_fn.update(pred_disps, target_disps)
        return self._r2_score_fn.compute()

    def __str__(self) -> str:
        return "displacement_r2"


class StressR2PerSample(nn.Module):
    def __init__(self, config: TacGraspNetConfig):
        super().__init__()
        self._config = config
        self._r2_score_fn = R2Score().to(config.device)

    def forward(self, batch: Databatch) -> torch.Tensor:
        if self._config.use_node_tetra_separate_decoders: # Check if separate decoders for nodes and tetrahedra are used
            # Extract target and predicted stresses from tetrahedra
            target_stresses = batch["targets.tetrahedra.normalized_outputs"]
            pred_stresses = batch["tetrahedra.outputs"]
        else: # Otherwise
            # Extract only target and predicted outputs from tactile sensor nodes
            target = batch["targets.tactile_sensors.nodes.normalized_outputs"]
            pred = batch["nodes.outputs"][batch["nodes.types"] != NodeType.OBJECT]
            _, target_stresses = torch.split(target, [3, 1], dim=-1)
            _, pred_stresses = torch.split(pred, [3, 1], dim=-1)

        self._r2_score_fn.reset()
        self._r2_score_fn.update(pred_stresses, target_stresses)
        return self._r2_score_fn.compute()

    def __str__(self) -> str:
        return "stress_r2"
