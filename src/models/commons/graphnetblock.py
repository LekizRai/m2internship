import torch

from torch import nn
from typing import Dict, Any

from models.tacgraspnet.tacgraspnet_config import TacGraspNetConfig
from models.commons.mlp import MLP


class GraphNetBlock(nn.Module):
    """Module for message passing"""

    def __init__(self, config: TacGraspNetConfig):
        super().__init__()
        self._config = config

        # Initialize MLP for node feature update
        self._node_mlp = MLP(
            # Input dimension = encoded (aggregated all types of corresponding toward edges + current node) or (aggregated
            # all types of corresponding toward edges + current node + global node) feature dimensions
            # Note: all features are encoded to the same latent space before
            input_dim=config.latent_dim * (len(config.edge_types) + 2) if config.use_global_node
                      else config.latent_dim * (len(config.edge_types) + 1),
            output_dim=config.latent_dim,
            hidden_dims=config.hidden_dims,
            hidden_activation=nn.ReLU(),
            is_output_normalized=config.use_final_layer_norm,
        ).to(config.device)

        # Initialize MLPs for all types of edge (mesh, contact, ...) feature updates
        self._edge_mlps = nn.ModuleDict()
        if config.use_separate_edge_mlps: # Separate MLPs are created for different types of edge feature updates
            for edge_type in config.edge_types:
                self._edge_mlps[edge_type] = MLP(
                    # Input dimension = encoded (sender (node) + receiver (node) + current edge) feature dimensions
                    # Note: all features are encoded to the same latent space before
                    input_dim=config.latent_dim * 3,
                    output_dim=config.latent_dim,
                    hidden_dims=config.hidden_dims,
                    hidden_activation=nn.ReLU(),
                    is_output_normalized=config.use_final_layer_norm,
                ).to(config.device)
        else: # Only one MLP is created for all types of edge feature updates
            edge_mlp = MLP(
                # Input dimension = encoded (sender (node) + receiver (node) + current edge) feature dimensions
                # Note: all features are encoded to the same latent space before
                input_dim=config.latent_dim * 3,
                output_dim=config.latent_dim,
                hidden_dims=config.hidden_dims,
                hidden_activation=nn.ReLU(),
                is_output_normalized=config.use_final_layer_norm,
            ).to(config.device)
            for edge_type in config.edge_types:
                self._edge_mlps[edge_type] = edge_mlp

        # Initialize MLP for tetrahedral feature update
        # Initialize only when flag is true
        if config.use_node_tetra_separate_decoders:
            self._tetra_mlp = MLP(
                # Input dimension = encoded (4 vertex nodes + tetrahedral) feature dimensions
                # Note: all features are encoded to the same latent space before
                input_dim=config.latent_dim * 5,
                output_dim=config.latent_dim,
                hidden_dims=config.hidden_dims,
                hidden_activation=nn.ReLU(),
                is_output_normalized=config.use_final_layer_norm,
            ).to(config.device)

        # Initialize MLP for global node feature update
        # Initialize only when flag is true
        if config.use_global_node:
            self._global_node_mlp = MLP(
                # Input dimension = encoded node feature dimensions
                # Note: all features are encoded to the same latent space before
                input_dim=config.latent_dim,
                output_dim=config.latent_dim,
                hidden_dims=config.hidden_dims,
                hidden_activation=nn.ReLU(),
                is_output_normalized=config.use_final_layer_norm,
            ).to(config.device)

    def _update_node_features(self, batch: Dict[str, Any]) -> torch.Tensor:
        V = batch["nodes.features"].shape[-2]  # Number of nodes

        # Edge features aggregation
        features = [batch["nodes.features"]]
        for edge_type in self._config.edge_types:
            edges = batch[edge_type]
            edge_features = batch[edge_type + ".features"]
            E = edges.shape[-2] # Number of edges
            D_E = edge_features.shape[-1] # Dimension of (encoded) edge feature (= latent dimension in TacGraspNet)
            aggregated_edge_features = torch.zeros(V, D_E, device=self._config.device) # Initialize with zero tensor
            aggregated_edge_features.scatter_add_(
                dim=-2,
                index=edges[..., 1, None].expand(E, D_E), # Consider only toward edges
                src=edge_features
            )
            features.append(aggregated_edge_features)

        # Append global node features if flag is true
        if self._config.use_global_node:
            global_node_features = [] # List to store all global node features
            for idx in torch.unique(batch["datapoints.indices"]): # Iterate over all data points
                cur_V = batch["nodes.features"][batch["datapoints.indices"] == idx].shape[0] # Number of currently considered nodes
                global_node_features.append(batch["global_node.features"][idx, ...].expand(cur_V, -1))

            # Append global node features to feature tensor
            features.append(torch.cat(global_node_features, dim=-2))

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

        # Update tetrahedral features with MLP
        return self._tetra_mlp(torch.cat(features, dim=-1))

    def _update_global_node_feature(self, batch: Dict[str, Any]):
        # Compute predicted and target values separately for each data point
        features = [] # List to store all global node features
        for idx in torch.unique(batch["datapoints.indices"]): # Iterate over all data points
            # Get node features from currently considered data point
            cur_node_features = batch["nodes.features"][batch["datapoints.indices"] == idx]

            # Aggregate all node features by taking the mean of them
            features.append(torch.mean(cur_node_features, dim=-2, keepdim=True))

        # Update global node feature with MLP
        return self._global_node_mlp(torch.cat(features, dim=-2))

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        # Store old features information which is used for residual connections later
        old_info = {"nodes.features": batch["nodes.features"].clone()}
        for edge_type in self._config.edge_types:
            old_info[edge_type + ".features"] = batch[edge_type + ".features"].clone()
        if self._config.use_node_tetra_separate_decoders:
            old_info["tetrahedra.features"] = batch["tetrahedra.features"].clone()
        if self._config.use_global_node:
            old_info["global_node.features"] = batch["global_node.features"].clone()

        # Update edge features
        for edge_type in self._config.edge_types:
            batch[edge_type + ".features"] = self._update_edge_features(edge_type, batch)

        # Update node features
        batch["nodes.features"] = self._update_node_features(batch)

        # Update tetrahedral features
        # Update only when flag is true
        if self._config.use_node_tetra_separate_decoders:
            batch["tetrahedra.features"] = self._update_tetra_features(batch)

        # Update global node feature
        # Update only when flag is true
        if self._config.use_global_node:
            batch["global_node.features"] = self._update_global_node_feature(batch)

        # Add residual connections
        batch["nodes.features"] = batch["nodes.features"] + old_info["nodes.features"]
        for edge_type in self._config.edge_types:
            batch[edge_type + ".features"] = batch[edge_type + ".features"] + old_info[edge_type + ".features"]
        if self._config.use_node_tetra_separate_decoders: # Add residual connections for tetrahedra if flag is true
            batch["tetrahedra.features"] = batch["tetrahedra.features"] + old_info["tetrahedra.features"]
        if self._config.use_global_node: # Add residual connections for global node if flag is true
            batch["global_node.features"] = batch["global_node.features"] + old_info["global_node.features"]

        return batch
