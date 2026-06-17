import torch

from torch import nn
import torch.nn.functional as F

from models.commons.mlp import MLP
from models.commons.graphnetblock import GraphNetBlock
from models.commons.normalizer import Normalizer
from models.tacgraspnet.tacgraspnet_config import TacGraspNetConfig
from commons.datatype import Databatch, NodeType
from utils.transform import split_two_fingers, kabsch


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
            print("#############")
            print("Use the same GraphNetBlock")
            print("#############")
            self._graphnetblock = GraphNetBlock(config=config)

        # Initializations for decoding
        # Initialize MLP for node decoding
        self._node_decoder = MLP(
            input_dim=config.latent_dim,
            output_dim=config.node_output_dim,
            hidden_dims=config.hidden_dims,
            hidden_activation=nn.ReLU(),
            # is_output_normalized=config.normalize_outputs, # Here, instead, depend on the normalization flag for outputs
        ).to(config.device)

        # Initialize MLP for tetrahedron decoding
        # Initialize only when flag is true
        if config.use_node_tetra_separate_decoders:
            self._tetra_decoder = MLP(
                input_dim=config.latent_dim,
                output_dim=config.tetra_output_dim,
                hidden_dims=config.hidden_dims,
                hidden_activation=nn.ReLU(),
                # is_output_normalized=config.normalize_outputs, # Here, instead, depend on the normalization flag for outputs
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
        # Encode node features
        batch["nodes.features"] = self._node_encoder(batch["nodes.features"])

        # Encode edge features
        for edge_type in self._config.edge_types:
            batch[edge_type + ".features"] = self._edge_encoders[edge_type](batch[edge_type + ".features"])

        # Encode tetrahedral features if flag is true
        if self._config.use_node_tetra_separate_decoders:
            batch["tetrahedra.features"] = self._tetra_encoder(batch["tetrahedra.features"])

        return batch

    def _decode(self, batch: Databatch) -> Databatch:
        # Decode node features
        batch["nodes.outputs"] = self._node_decoder(batch["nodes.features"])

        # Decode tetrahedral features if flag is true
        if self._config.use_node_tetra_separate_decoders:
            batch["tetrahedra.outputs"] = self._tetra_decoder(batch["tetrahedra.features"])

        print("--- DECODE DIAGNOSTIC ---")
        print("Output Disps Min/Max:", batch["nodes.outputs"].min().item(), batch["nodes.outputs"].max().item())
        print("Output Stress Min/Max:", batch["tetrahedra.outputs"].min().item(), batch["tetrahedra.outputs"].max().item())

        return batch

    def _update(self, batch: Databatch) -> Databatch:
        # TODO
        # Unnormalize node (and tetrahedral) outputs to compute next positions and scores
        if self._config.normalize_outputs: # If flag is true, then using normalizers
            if self._config.use_node_tetra_separate_decoders:
                unnormalized_pred_disps = self._node_output_normalizer.inverse(batch["nodes.outputs"])
                unnormalized_pred_stresses = self._tetra_output_normalizer.inverse(batch["tetrahedra.outputs"])
            else:
                unnormalized_outputs = self._node_output_normalizer.inverse(batch["nodes.outputs"])
                unnormalized_pred_disps, unnormalized_pred_stresses = torch.split(
                    unnormalized_outputs,
                    [3, 1],
                    dim=-1
                )
        else: # Otherwise, the outputs themselves are unnormalized
            if self._config.use_node_tetra_separate_decoders:
                unnormalized_pred_disps = batch["nodes.outputs"]
                unnormalized_pred_stresses = batch["tetrahedra.outputs"]
            else:
                unnormalized_pred_disps, unnormalized_pred_stresses = torch.split(
                    batch["nodes.outputs"],
                    [3, 1],
                    dim=-1
                )

        # Extract vertice positions of objects (since it is unchanged)
        pred_obj_pos = batch["vertices.positions"][batch["nodes.types"] == NodeType.OBJECT]
        # Extract displacements of tactile sensor nodes only
        unnormalized_pred_ts_disps = unnormalized_pred_disps[batch["nodes.types"] != NodeType.OBJECT]

        if self._config.use_translation_inductive_bias: # Carry out adding translation
            if self._config.use_template_data:
                A = batch["template.vertices.positions"][batch["nodes.types"] != NodeType.OBJECT]
            else:
                A = batch["2nd_frame.vertices.positions"][batch["nodes.types"] != NodeType.OBJECT]
            B = batch["vertices.positions"][batch["nodes.types"] != NodeType.OBJECT]

            left_idx, right_idx = split_two_fingers(A)
            Al, Bl = A[left_idx], B[left_idx]
            Ar, Br = A[right_idx], B[right_idx]

            Rl, tl = kabsch(Al, Bl)
            Rr, tr = kabsch(Ar, Br)

            baseline = torch.zeros_like(A)
            baseline[left_idx] = Al @ Rl.T + tl
            baseline[right_idx] = Ar @ Rr.T + tr

            resid = unnormalized_pred_ts_disps
            resid_world = torch.zeros_like(resid)
            resid_world[left_idx] = resid[left_idx] @ Rl.T
            resid_world[right_idx] = resid[right_idx] @ Rr.T

            pred_ts_pos = baseline.clone()
            pred_ts_pos[left_idx] = baseline[left_idx] + resid_world[left_idx]
            pred_ts_pos[right_idx] = baseline[right_idx] + resid_world[right_idx]

            # world residual → local residual per finger
            world_resid = B - baseline
            target_ts_disps = torch.zeros_like(world_resid)
            target_ts_disps[left_idx] = world_resid[left_idx] @ Rl
            target_ts_disps[right_idx] = world_resid[right_idx] @ Rr
        else:
            if self._config.use_template_data:
                pred_ts_pos = (batch["template.vertices.positions"][batch["nodes.types"] != NodeType.OBJECT]
                               + unnormalized_pred_ts_disps)
                target_ts_disps = (batch["vertices.positions"]
                                   - batch["template.vertices.positions"])[batch["nodes.types"] != NodeType.OBJECT]
            else:
                pred_ts_pos = (batch["2nd_frame.vertices.positions"][batch["nodes.types"] != NodeType.OBJECT]
                               + unnormalized_pred_ts_disps)
                target_ts_disps = (batch["vertices.positions"]
                                   - batch["2nd_frame.vertices.positions"])[batch["nodes.types"] != NodeType.OBJECT]

        # Concatenate in the same order as Datapoint (object first, then gripper)
        pred_pos = torch.cat((pred_ts_pos, pred_obj_pos), dim=-2)

        # Stress: zero for object nodes, keep predicted for gripper nodes
        if not self._config.use_node_tetra_separate_decoders:
            unnormalized_pred_stresses[batch["nodes.types"] == NodeType.OBJECT] = 0.0
        unnormalized_pred_stresses = F.relu(unnormalized_pred_stresses)

        # Add predicted value
        batch["predictions.vertices.positions"] = pred_pos
        batch["predictions.vertices.displacements"] = unnormalized_pred_disps
        if self._config.use_node_tetra_separate_decoders:
            batch["predictions.tetrahedra.stresses"] = unnormalized_pred_stresses
        else:
            batch["predictions.vertices.stresses"] = unnormalized_pred_stresses
        ######

        if self._config.use_node_tetra_separate_decoders:  # Extract both node and tetrahedral outputs if flag is true
            # Extract only tactile sensor node outputs (target displacements)
            batch["targets.nodes.outputs"] = target_ts_disps  # Target displacements
            # Extract tetrahedral outputs
            batch["targets.tetrahedra.outputs"] = batch["tetrahedra.stresses"] # Target stresses
        else:
            # Extract only tactile sensor node outputs
            batch["targets.nodes.outputs"] = torch.cat([
                target_ts_disps,
                batch["vertices.stresses"][batch["nodes.types"] != NodeType.OBJECT]
            ], dim=-1)

        # Normalize target outputs if flat is true
        if self._config.normalize_outputs:
            if self._config.use_node_tetra_separate_decoders:  # Normalize both node and tetrahedral outputs if flag is true
                batch["targets.nodes.normalized_outputs"] = self._node_output_normalizer(
                    batch["targets.nodes.outputs"],
                    is_training=self._config.is_training
                ) # Normalize node outputs (Target displacements)
                batch["targets.tetrahedra.normalized_outputs"] = self._tetra_output_normalizer(
                    batch["targets.tetrahedra.outputs"],
                    is_training=self._config.is_training
                ) # Normalize tetrahedral outputs (target stresses)
            else:  # Normalize only node outputs otherwise
                batch["targets.nodes.normalized_outputs"] = self._node_output_normalizer(
                    batch["targets.nodes.outputs"],
                    is_training=self._config.is_training
                ) # Normalize node outputs (Target displacements and stresses)

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
            if self._config.use_node_tetra_separate_decoders:  # Normalize tetrahedral features if flag is true
                pass
                # batch["tetrahedra.features"] = self._tetra_normalizer(
                #     batch["tetrahedra.features"],
                #     is_training=self._config.is_training
                # )

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

        return self._update(batch)

    def accumulate(self, batch: Databatch):
        if self._config.normalize_features:
            self._node_normalizer(
                batch["nodes.features"],
                is_training=True
            )
            for edge_type in self._config.edge_types:
                self._edge_normalizers[edge_type](
                    batch[edge_type + ".features"],
                    is_training=True
                )
            if self._config.use_node_tetra_separate_decoders:  # Normalize tetrahedral features if flag is true
                pass
                # batch["tetrahedra.features"] = self._tetra_normalizer(
                #     batch["tetrahedra.features"],
                #     is_training=self._config.is_training
                # )
                # Normalize target outputs if flat is true
        if self._config.normalize_outputs:
            if self._config.use_template_data:
                target_ts_disps = (batch["vertices.positions"]
                                   - batch["template.vertices.positions"])[batch["nodes.types"] != NodeType.OBJECT]
            else:
                target_ts_disps = (batch["vertices.positions"]
                                   - batch["2nd_frame.vertices.positions"])[batch["nodes.types"] != NodeType.OBJECT]
            if self._config.use_node_tetra_separate_decoders:  # Normalize both node and tetrahedral outputs if flag is true
                self._node_output_normalizer(
                    target_ts_disps,
                    is_training=True
                )  # Normalize node outputs (Target displacements)
                self._tetra_output_normalizer(
                    batch["tetrahedra.stresses"],
                    is_training=True
                )  # Normalize tetrahedral outputs (target stresses)
            else:  # Normalize only node outputs otherwise
                self._node_output_normalizer(
                    torch.cat([
                        target_ts_disps,
                        batch["vertices.stresses"][batch["nodes.types"] != NodeType.OBJECT]
                    ], dim=-1),
                    is_training=True
                )  # Normalize node outputs (Target displacements and stresses)

    def set_is_training(self, is_training: bool):
        self._config.is_training = is_training
