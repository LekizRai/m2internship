from typing import Dict, Any
import torch
from torch import nn

from models.commons.mlp import MLP
from models.commons.graphnetblock import GraphNetBlock
from models.commons.normalizer import Normalizer
from models.tacgraspnet.tacgraspnet_config import TacGraspNetConfig
from commons.datatype import Databatch


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
            output_activation=nn.ReLU(),
            is_output_normalized=True,
        ).to(config.device)

        # Initialize MLPs for edge feature encoding
        self._edge_encoders = {}
        for edge_type in config.edge_types:
            self._edge_encoders[edge_type] = MLP(
                input_dim=config.edge_feature_dims[edge_type],
                output_dim=config.latent_dim,
                hidden_dims=config.hidden_dims,
                output_activation=nn.ReLU(),
                is_output_normalized=True,
            ).to(config.device)

        # Initialize MLP for tetrahedral feature encoding
        self._tetra_encoder = MLP(
            input_dim=config.tetra_feature_dim,
            output_dim=config.latent_dim,
            hidden_dims=config.hidden_dims,
            output_activation=nn.ReLU(),
            is_output_normalized=True,
        ).to(config.device)

        # Initialization for processing
        self._graphnetblocks = []
        for _ in range(self._config.message_passing_steps):
            self._graphnetblocks.append(GraphNetBlock(
                config=config,
            ))

        # Initializations for decoding
        # Initialize MLP for node decoding
        self._node_decoder = MLP(
            input_dim=config.latent_dim,
            output_dim=config.node_output_dim,
            hidden_dims=config.hidden_dims,
            output_activation=nn.ReLU(),
        ).to(config.device)

        # Initialize MLP for tetrahedron decoding
        self._tetra_decoder = MLP(
            input_dim=config.latent_dim,
            output_dim=config.tetra_output_dim,
            hidden_dims=config.hidden_dims,
            output_activation=nn.ReLU(),
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
            )
            if config.use_node_tetra_separate_decoders:
                self._tetra_normalizer = Normalizer(
                    config=config,
                    feature_dim=config.tetra_feature_dim
                )
                self._tetra_output_normalizer = Normalizer(
                    config=config,
                    feature_dim=config.tetra_output_dim
                )

    def _encode(self, batch: Databatch) -> Databatch:
        new_batch = batch.copy()

        # Encode node features
        new_batch["nodes.features"] = self._node_encoder(batch["nodes.features"])

        # Encode edge features
        for edge_type in self._config.edge_types:
            new_batch[edge_type + ".features"] = self._edge_encoders[edge_type](batch[edge_type + ".features"])

        # Encode tetrahedral features
        new_batch["tetrahedra.features"] = self._tetra_encoder(batch["tetrahedra.features"])

        return new_batch

    def _decode(self, batch: Databatch) -> Databatch:
        new_batch = batch.copy()

        # Decode node features
        new_batch["nodes.outputs"] = self._node_decoder(batch["nodes.features"])

        # Decode tetrahedral features
        new_batch["tetrahedra.outputs"] = self._tetra_decoder(batch["tetrahedra.features"])

        return new_batch

    def forward(self, batch: Databatch) -> Databatch:
        # Features normalization
        batch["nodes.features"] = self._node_normalizer(
            batch["nodes.features"],
            is_training=self._config.is_training
        )
        for edge_type in self._config.edge_types:
            batch[edge_type + ".features"] = self._edge_normalizers[edge_type](
                batch[edge_type + ".features"],
                is_training=self._config.is_training
            )
        if self._config.use_node_tetra_separate_decoders:
            batch["tetrahedra.features"] = self._tetra_normalizer(
                batch["tetrahedra.features"],
                is_training=self._config.is_training
            )

        # Encode
        batch = self._encode(batch)

        # Process
        for graphnetblock in self._graphnetblocks:
            batch = graphnetblock(batch)

        # Decode
        batch = self._decode(batch)

        return batch
