from typing import Dict, Any
import torch
from torch import nn

from models.tacgraspnet.tacgraspnet_config import TacGraspNetConfig
from models.mlp import MLP


class GraphNetBlock(nn.Module):
    """Module for message passing"""

    def __init__(
            self,
            config: TacGraspNetConfig,
    ):
        super().__init__()
        self._config = config

        # Initialize MLPs for all types of edge (mesh, contact, ...) feature updates
        self._edge_mlps = {}
        for edge_type in config.edge_types:
            self._edge_mlps[edge_type] = MLP(
                # Input dimension = encoded (sender (node) + receiver (node) + current edge) feature dimensions
                # Note: all features are encoded to the same latent space before
                input_dim=config.latent_dim * 3,
                output_dim=config.latent_dim,
                hidden_dims=config.hidden_dims,
                hidden_activation=nn.ReLU(),
                is_output_normalized=True,
            )

        # Initialize MLP for node feature update
        self._node_mlp = MLP(
            # Input dimension = encoded (aggregated all types of corresponding toward edges + current node) feature dimensions
            # Note: all features are encoded to the same latent space before
            input_dim=config.latent_dim * (len(config.edge_types) + 1),
            output_dim=config.latent_dim,
            hidden_dims=config.hidden_dims,
            hidden_activation=nn.ReLU(),
            is_output_normalized=True,
        )

        # Initialize MLP for tetrahedral feature update
        self._tetra_mlp = MLP(
            # Input dimension = encoded (4 vertex nodes + tetrahedral) feature dimensions
            # Note: all features are encoded to the same latent space before
            input_dim=config.latent_dim * 5,
            output_dim=config.latent_dim,
            hidden_dims=config.hidden_dims,
            hidden_activation=nn.ReLU(),
            is_output_normalized=True,
        )

    def _update_node_features(self, batch: Dict[str, Any]) -> torch.Tensor:
        V = batch["nodes.features"].shape[-2]  # Number of nodes

        # Edge features aggregation
        features = [batch["nodes.features"]]
        for edge_type in self._config.edge_types:
            edges = batch[edge_type]
            edge_features = batch[edge_type + ".features"]
            E = edges.shape[-2] # Number of edges
            D_E = edge_features.shape[-1] # Dimension of (encoded) edge feature (= latent dimension in TacGraspNet)
            aggregated_edge_features = torch.zeros(V, D_E) # Initialize with zero tensor
            aggregated_edge_features.scatter_add_(
                dim=-2,
                index=edges[..., 1, None].expand(E, D_E), # Consider only toward edges
                src=edge_features
            )
            features.append(aggregated_edge_features)

        # Update node features with MLP
        return self._node_mlp(torch.cat(features, dim=-1))

    def _update_edge_features(self, edge_type: str, batch: Dict[str, Any]) -> torch.Tensor:
        node_features = batch["nodes.features"]
        edges = batch[edge_type]
        edge_features = batch[edge_type + ".features"]
        E = edges.shape[-2]  # Number of edges
        D_V = node_features.shape[-1]  # Dimension of (encoded) node feature (= latent dimension in TacGraspNet)

        # Node features gathering
        sender_features = torch.gather(
            input=node_features,
            dim=-2,
            index=edges[..., 0, None].expand(E, D_V)
        )
        receiver_features = torch.gather(
            input=node_features,
            dim=-2,
            index=edges[..., 1, None].expand(E, D_V)
        )
        features = [sender_features, receiver_features, edge_features]

        # Update edge features with MLP
        return self._edge_mlps[edge_type](torch.cat(features, dim=-1))

    def _update_tetra_features(self, batch: Dict[str, Any]) -> torch.Tensor:
        node_features = batch["nodes.features"]
        tetras = batch["tetrahedra"]
        T = batch["tetrahedra.features"].shape[-2] # Number of tetrahedra
        D_V = node_features.shape[-1]  # Dimension of (encoded) node feature (= latent dimension in TacGraspNet)

        # Node features gathering
        features = [batch["tetrahedra.features"]]
        for i in range(4):
            features.append(torch.gather(
                input=node_features,
                dim=-2,
                index=tetras[..., i, None].expand(T, D_V)
            ))

        # Update edge features with MLP
        return self._tetra_mlp(torch.cat(features, dim=-1))

    # TODO
    # def _update_global_node_feature(self, batch: Dict[str, Any]):
    #     return batch["globa_node.features"]

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        new_batch = batch.copy()

        # Update edge features
        for edge_type in self._config.edge_types:
            new_batch[edge_type + ".features"] = self._update_edge_features(edge_type, batch)

        # Update node features
        new_batch["nodes.features"] = self._update_node_features(batch)

        # Update tetrahedral features
        new_batch["tetrahedra.features"] = self._update_tetra_features(batch)

        # Add residual connections
        new_batch["nodes.features"] += batch["nodes.features"]
        for edge_type in self._config.edge_types:
            new_batch[edge_type + ".features"] += batch[edge_type + ".features"]
        new_batch["tetrahedra.features"] += batch["tetrahedra.features"]

        return new_batch
