import torch

from torch import nn
import torch.nn.functional as F

from models.tacgraspnet.tacgraspnet_config import TacGraspNetConfig
from commons.datatype import Databatch, NodeType


class MSE(nn.Module):
    def __init__(self, config: TacGraspNetConfig):
        super().__init__()
        self._config = config

    def forward(self, batch: Databatch) -> torch.Tensor:
        if self._config.use_node_tetra_separate_decoders: # Check if separate decoders for nodes and tetrahedra are used
            # Extract only target and predicted displacements from tactile sensor nodes
            target_disps = batch["targets.tactile_sensors.nodes.normalized_outputs"]
            pred_disps = batch["nodes.outputs"][batch["nodes.types"] != NodeType.OBJECT]
            # Extract target and predicted stresses from tetrahedra
            target_stresses = batch["targets.tetrahedra.normalized_outputs"]
            pred_stresses = batch["tetrahedra.outputs"]
        else: # Otherwise
            # Extract only target and predicted outputs from tactile sensor nodes
            target = batch["targets.tactile_sensors.nodes.normalized_outputs"]
            pred = batch["nodes.outputs"][batch["nodes.types"] != NodeType.OBJECT]
            target_disps, target_stresses = torch.split(target, [3, 1], dim=-1)
            pred_disps, pred_stresses = torch.split(pred, [3, 1], dim=-1)

        #########################################################################
        # disp_l2_error = torch.norm(pred_disps - target_disps, p=2, dim=-1)
        # stress_l2_error = torch.norm(pred_stresses - target_stresses, p=2, dim=-1)
        # disp_mse = torch.mean(disp_l2_error)
        # stress_mse = torch.mean(stress_l2_error)
        #########################################################################
        # Elementwise MSE
        disp_mse = F.mse_loss(pred_disps, target_disps)
        stress_mse = F.mse_loss(pred_stresses, target_stresses)
        return disp_mse + stress_mse

    def __str__(self) -> str:
        return "mse"
