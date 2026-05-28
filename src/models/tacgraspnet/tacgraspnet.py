from typing import Dict, Any
import torch
from torch import nn

from models.commons.mlp import MLP
from models.commons.graphnetblock import GraphNetBlock
from models.commons.normalizer import Normalizer
from models.tacgraspnet.tacgraspnet_config import TacGraspNetConfig
from commons.datatype import Databatch, NodeType


class TacGraspNet(nn.Module):
    def __init__(self, config: TacGraspNetConfig):
        super().__init__()
        self._config = config

        # Initializations for encoding
        # Initialize MLP for node feature encoding
        self._node_encoder = MLP(
            input_dim=config.node_feature_dim,
            output_dim=config.latent_dim,
            hidden_dims=config.hidden_dims,
            hidden_activation=nn.ReLU(),
            is_output_normalized=config.use_final_layer_norm,
        ).to(config.device)

        # Initialize MLPs for edge feature encoding
        self._edge_encoders = {}
        for edge_type in config.edge_types:
            self._edge_encoders[edge_type] = MLP(
                input_dim=config.edge_feature_dims[edge_type],
                output_dim=config.latent_dim,
                hidden_dims=config.hidden_dims,
                hidden_activation=nn.ReLU(),
                is_output_normalized=config.use_final_layer_norm,
            ).to(config.device)

        # Initialize MLP for tetrahedral feature encoding
        # Initialize only when flag is true
        if config.use_node_tetra_separate_decoders:
            self._tetra_encoder = MLP(
                input_dim=config.tetra_feature_dim,
                output_dim=config.latent_dim,
                hidden_dims=config.hidden_dims,
                hidden_activation=nn.ReLU(),
                is_output_normalized=config.use_final_layer_norm,
            ).to(config.device)

        # Initialization for processing
        if config.use_message_passing_separate_mlps: # Multiple (separate) GraphNetBlocks are created
            self._graphnetblocks = []
            for _ in range(self._config.message_passing_steps):
                self._graphnetblocks.append(GraphNetBlock(
                    config=config,
                ))
        else: # Only one GraphNetBlock is created
            self._graphnetblock = GraphNetBlock(config=config)

        # Initializations for decoding
        # Initialize MLP for node decoding
        self._node_decoder = MLP(
            input_dim=config.latent_dim,
            output_dim=config.node_output_dim,
            hidden_dims=config.hidden_dims,
            hidden_activation=nn.ReLU(),
            is_hidden_normalized=config.normalize_features, # Here, instead, depend on whether features are normalized or not
        ).to(config.device)

        # Initialize MLP for tetrahedron decoding
        # Initialize only when flag is true
        if config.use_node_tetra_separate_decoders:
            self._tetra_decoder = MLP(
                input_dim=config.latent_dim,
                output_dim=config.tetra_output_dim,
                hidden_dims=config.hidden_dims,
                output_activation=nn.ReLU(),
                is_hidden_normalized=config.normalize_features, # Here, instead, depend on whether features are normalized or not
            ).to(config.device)

        # Initialize normalizers for feature normalization if flag is true
        if config.normalize_features:
            self._node_normalizer = Normalizer(
                config=config,
                feature_dim=config.node_feature_dim
            ).to(config.device)
            self._edge_normalizers= {}
            for edge_type in config.edge_types:
                self._edge_normalizers[edge_type] = Normalizer(
                    config=config,
                    feature_dim=config.edge_feature_dims[edge_type]
                ).to(config.device)
            self._node_output_normalizer = Normalizer(
                config=config,
                feature_dim=config.node_output_dim
            ).to(config.device)
            if config.use_node_tetra_separate_decoders: # Initialize normalizer for tetrahedral features if flag is true
                self._tetra_normalizer = Normalizer(
                    config=config,
                    feature_dim=config.tetra_feature_dim
                ).to(config.device)
                self._tetra_output_normalizer = Normalizer(
                    config=config,
                    feature_dim=config.tetra_output_dim
                ).to(config.device)

    def _encode(self, batch: Databatch) -> Databatch:
        new_batch = batch.copy()

        # Encode node features
        new_batch["nodes.features"] = self._node_encoder(batch["nodes.features"])

        # Encode edge features
        for edge_type in self._config.edge_types:
            new_batch[edge_type + ".features"] = self._edge_encoders[edge_type](batch[edge_type + ".features"])

        # Encode tetrahedral features if flag is true
        if self._config.use_node_tetra_separate_decoders:
            new_batch["tetrahedra.features"] = self._tetra_encoder(batch["tetrahedra.features"])

        return new_batch

    def _decode(self, batch: Databatch) -> Databatch:
        new_batch = batch.copy()

        # Decode node features
        new_batch["nodes.outputs"] = self._node_decoder(batch["nodes.features"])

        # Decode tetrahedral features if flag is true
        if self._config.use_node_tetra_separate_decoders:
            new_batch["tetrahedra.outputs"] = self._tetra_decoder(batch["tetrahedra.features"])

        return new_batch

    # TODO
    def update(self, batch: Databatch) -> Databatch:
        batch = self.forward(batch)
        return batch

    def forward(self, batch: Databatch) -> Databatch:
        # Normalize features if flag is true
        if self._config.normalize_features:
            batch["nodes.features"] = self._node_normalizer(
                batch["nodes.features"],
                is_training=self._config.is_training
            )
            for edge_type in self._config.edge_types:
                batch[edge_type + ".features"] = self._edge_normalizers[edge_type](
                    batch[edge_type + ".features"],
                    is_training=self._config.is_training
                )
            # Get starting positions to compute target displacements
            starting_pos = batch["template.vertices.positions"] if self._config.use_template_data \
                           else batch["2nd_frame.vertices.positions"]
            if self._config.use_node_tetra_separate_decoders:  # Normalize corresponding outputs and tetrahedral features if flag is true
                batch["tetrahedra.features"] = self._tetra_normalizer(
                    batch["tetrahedra.features"],
                    is_training=self._config.is_training
                )

                # Normalize target node outputs (target displacements)
                batch["targets.nodes.outputs"] = batch["vertices.positions"] - starting_pos # Target displacements
                # Store only normalized tactile sensor node outputs
                batch["targets.tactile_sensors.nodes.normalized_outputs"] = self._node_output_normalizer(
                    # Only normalize tactile sensor node outputs
                    batch["targets.nodes.outputs"][batch["nodes.types"] != NodeType.OBJECT]
                )

                # Normalize target tetrahedral outputs (target stresses)
                batch["targets.tetrahedra.outputs"] = batch["tetrahedra.stresses"]
                batch["targets.tetrahedra.normalized_outputs"] = self._tetra_output_normalizer(
                    batch["targets.tetrahedra.outputs"]
                )
            else: # Normalize corresponding outputs otherwise
                batch["targets.nodes.outputs"] = torch.cat([
                    batch["vertices.positions"] - starting_pos,
                    batch["vertices.stresses"]
                ], dim=-1)
                # Store only normalized tactile sensor node outputs
                batch["targets.tactile_sensors.nodes.normalized_outputs"] = self._node_output_normalizer([
                    # Only normalize tactile sensor node outputs
                    batch["targets.nodes.outputs"][batch["nodes.types"] != NodeType.OBJECT]
                ])

        # Encode
        batch = self._encode(batch)

        # Process
        if self._config.use_message_passing_separate_mlps:  # Different GraphNetBlock for each message passing step
            for graphnetblock in self._graphnetblocks:
                batch = graphnetblock(batch)
        else:  # The same GraphNetBlock for all message passing steps
            for _ in range(self._config.message_passing_steps):
                batch = self._graphnetblock(batch)

        # Decode
        batch = self._decode(batch)

        return batch
