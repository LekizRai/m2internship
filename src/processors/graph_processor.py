import torch

from typing import Dict, Any
from itertools import permutations
from torch.nn import functional as F

from models.tacgraspnet.tacgraspnet_config import TacGraspNetConfig
from commons.processor import Processor
from commons.datatype import NodeType, Databatch
from utils.graph import construct_contact_edges


class GraphBuildingProcessor(Processor):
    def __init__(self, config: TacGraspNetConfig):
        self._config = config

    def __call__(self, batch: Databatch) -> Databatch:
        # Build mesh edges
        mesh_edges, mesh_edge_features = self._build_mesh_edges(batch)
        batch["mesh_edges"] = mesh_edges
        batch["mesh_edges.features"] = mesh_edge_features

        # Build contact edges (world edges)
        contact_edges, contact_edge_features = self._build_contact_edges(batch)
        batch["contact_edges"] = contact_edges
        batch["contact_edges.features"] = contact_edge_features

        # Build node features
        node_features = self._build_node_features(batch)
        batch["nodes.features"] = node_features

        # Build tetrahedral features
        if self._config.use_node_tetra_separate_decoders:
            tetra_features = torch.zeros(batch["tetrahedra"].shape[-2], 1)
            batch["tetrahedra.features"] = tetra_features

        # Build global node feature
        if self._config.use_global_node:
            global_node_features = torch.zeros(1, self._config.global_node_feature_dim)
            batch["global_node.features"] = global_node_features

        return batch

    def _build_mesh_edges(self, batch: Databatch) -> tuple[torch.Tensor, torch.Tensor]:
        ########################################
        ## Tactile sensor mesh edges
        ########################################
        # Get all tactile sensor tetrahedra
        ts_tetras = batch["tetrahedra"]
        # List of permutation indices to retrieve edges from (tactile sensor) tetrahedra
        ts_perm_idx = [0, 1, 0, 2, 0, 3, 1, 0, 1, 2, 1, 3, 2, 0, 2, 1, 2, 3, 3, 0, 3, 1, 3, 2]
        # Gather all permutations and reshape to obtain tactile sensor mesh edges
        ts_mesh_edges = ts_tetras[:, ts_perm_idx].reshape(-1, 2)
        # Eliminate duplicates
        ts_mesh_edges = torch.unique(ts_mesh_edges, dim=-2)

        ########################################
        ## Object mesh edges
        ########################################
        # Get all object faces
        obj_faces = batch["faces"]
        # List of permutation indices to retrieve edges from (object) faces
        obj_perm_idx = [0, 1, 0, 2, 1, 0, 1, 2, 2, 0, 2, 1]
        # Gather all permutations and reshape to obtain object mesh edges
        obj_mesh_edges = obj_faces[:, obj_perm_idx].reshape(-1, 2)
        obj_mesh_edges = torch.unique(obj_mesh_edges, dim=-2)

        ########################################
        ## Mesh edge features
        ########################################
        # Combine two mesh edge tensors to form the complete edge tensor
        mesh_edges = torch.cat([ts_mesh_edges, obj_mesh_edges], dim=-2)

        # Retrieve senders and receivers from mesh edges
        senders = mesh_edges[..., 0]  # Get mesh edge sender indices
        receivers = mesh_edges[..., 1]  # Get mesh edge receiver indices

        # Compute mesh edge features
        if self._config.use_template_data: # Use template data if flag is true
            # Compute template relative displacement for each mesh edge
            relative_template_disps = batch["template.vertices.positions"][receivers] \
                                      - batch["template.vertices.positions"][senders]
            mesh_edge_features = torch.cat([
                relative_template_disps, # Template relative displacements
                torch.norm(relative_template_disps, dim=-1, keepdim=True) # Template relative displacement norms
            ], dim=-1)
        else: # Otherwise, use first and second frame data
            # Compute first frame relative displacement for each mesh edge
            relative_1st_frame_disps = batch["1st_frame.vertices.positions"][receivers] \
                                       - batch["1st_frame.vertices.positions"][senders]
            # Compute second frame relative displacement for each mesh edge
            relative_2nd_frame_disps = batch["2nd_frame.vertices.positions"][receivers] \
                                       - batch["2nd_frame.vertices.positions"][senders]
            mesh_edge_features = torch.cat([
                relative_2nd_frame_disps,  # Second frame relative displacements
                torch.norm(relative_2nd_frame_disps, dim=-1, keepdim=True),  # Second frame relative displacement norms
                relative_1st_frame_disps,  # First frame relative displacements
                torch.norm(relative_1st_frame_disps, dim=-1, keepdim=True),  # First frame relative displacement norms
            ], dim=-1)

        return mesh_edges, mesh_edge_features

    def _build_contact_edges(self, batch: Databatch) -> tuple[torch.Tensor, torch.Tensor]:
        # To compute contact edges and their features, we need to work individually on each data point
        # of the batch. We can use data point indices in batch to separate all data points
        datapoint_indices = batch["datapoints.indices"] # Extract data point indices from batch
        contact_edge_lst = [] # Initialize list of contact edges
        contact_edge_feature_lst = [] # Initialize list of contact edge features
        # This value is used to compute cumulative node indices when processing data points individually
        current_node_index_cumul = 0
        for idx in torch.unique(datapoint_indices):
            ########################################
            ## Contact edges
            ########################################
            # Retrieve from each data point necessary data to construct contact edges
            vert_2nd_pos = batch["2nd_frame.vertices.positions"][datapoint_indices == idx, ...]
            node_types = batch["nodes.types"][datapoint_indices == idx]

            # Construct contact edges from second frame vertice positions
            # Accumulative values are added despite previous collation because construction of contact edges is
            # based on vertice positions, not from data. Therefore, it is a from-scratch process and needs common separations
            contact_edges = construct_contact_edges(
                vert_2nd_pos,
                node_types,
                self._config.radius
            ) + current_node_index_cumul # Add cumulative value to separate sets of contact edges

            ########################################
            ## Contact edge features
            ########################################
            # Retrieve senders and receivers from contact edges
            # Note that cumulative node index value is not needed here
            senders = contact_edges[..., 0]  # Get contact edge sender indices
            receivers = contact_edges[..., 1]  # Get contact edge receiver indices

            # Compute force feature for each contact edge at current (considered) data point
            force = batch["forces"][idx].item()
            # Number of contact edges are divided by two because they are all bidirectional edges
            # Also need to ensure that there is no divided-by-zero error
            force_per_contact_edge = force / (contact_edges.shape[-2] / 2 + 1e-8)
            force_features = torch.full((contact_edges.shape[-2], 1), force_per_contact_edge)

            # Compute contact edge features
            if self._config.use_template_data:  # Use template data if flag is true
                # Compute template relative displacement for each mesh edge
                relative_template_disps = batch["template.vertices.positions"][receivers] \
                                          - batch["template.vertices.positions"][senders]
                contact_edge_features = torch.cat([
                    relative_template_disps,  # Template relative displacements
                    torch.norm(relative_template_disps, dim=-1, keepdim=True),  # Template relative displacement norms
                    force_features  # Force features
                ], dim=-1)
            else:  # Otherwise, use first and second frame data
                # Compute second frame relative displacement for each mesh edge
                relative_2nd_frame_disps = batch["2nd_frame.vertices.positions"][receivers] \
                                           - batch["2nd_frame.vertices.positions"][senders]
                contact_edge_features = torch.cat([
                    relative_2nd_frame_disps,  # Second frame relative displacements
                    torch.norm(relative_2nd_frame_disps, dim=-1, keepdim=True),  # Second frame relative displacement norms
                    force_features  # Force features
                ], dim=-1)

            # Add contact edges and contact edge features to lists
            contact_edge_lst.append(contact_edges)
            contact_edge_feature_lst.append(contact_edge_features)

            # Update cumulative node index value
            current_node_index_cumul += len(node_types)

        # Gather all information from all data points together
        contact_edges = torch.cat(contact_edge_lst, dim=-2)
        contact_edge_features = torch.cat(contact_edge_feature_lst, dim=-2)

        return contact_edges, contact_edge_features

    def _build_node_features(self, batch: Databatch) -> torch.Tensor:
        # Update: velocity in this context is not common velocity. In fact, it is closing direction
        # Get node velocities of tactile sensors and object
        # Note that velocities are computed using normal vectors of tactile sensors (left and right)
        node_velocities = [] # Initialize node velocity list
        for idx in torch.unique(batch["datapoints.indices"]): # Iterate over all data points
            node_types = batch["nodes.types"][batch["datapoints.indices"] == idx] # Get node types of current (considered) data point
            n_obj_nodes = int((node_types == NodeType.OBJECT).sum().item())  # Get number of object nodes (vertices)
            n_ts_nodes = len(node_types) - n_obj_nodes  # Get number of tactile sensor nodes
            n_ts_comp_nodes = n_ts_nodes // 2  # Get number of each tactile sensor (component) nodes (left and right)

            # Compute all node velocities (both tactile sensors and object)
            ts_left_node_velocities = batch["tactile_sensors.normals"][idx, 0, ...].expand(n_ts_comp_nodes, -1)
            ts_right_node_velocities = batch["tactile_sensors.normals"][idx, 1, ...].expand(n_ts_comp_nodes, -1)
            obj_node_velocities = torch.zeros((n_obj_nodes, 3)).to(self._config.device)

            # Add velocities to node velocity list
            node_velocities.extend([
                ts_left_node_velocities,
                ts_right_node_velocities,
                obj_node_velocities
            ])

        node_velocities = torch.cat(node_velocities, dim=-2)

        # Get node one-hot encodings of tactile sensors and object
        node_one_hot_encodings = F.one_hot(batch["nodes.types"], NodeType.NUM)

        # Build node features
        node_features = torch.cat([
            node_velocities,
            node_one_hot_encodings
        ], dim=-1)

        return node_features
