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
        tetra_features = torch.zeros(batch["tetrahedra"].shape[-2], 1)
        batch["tetrahedra.features"] = tetra_features

        return batch

    def _build_mesh_edges(self, batch: Databatch) -> tuple[torch.Tensor, torch.Tensor]:
        ########################################
        ## Tactile sensor mesh edges
        ########################################
        # Build tactile sensor mesh edges
        ts_tetras = batch["tetrahedra"]
        ts_mesh_edges = []  # Initialize tactile sensor mesh edges

        for ts_tetra in ts_tetras:  # Iterate over all tactile sensor tetrahedra
            for permutation in permutations(ts_tetra,
                                            2):  # Iterate over all size-2 permutations of indices in tetrahedron
                ts_mesh_edges.append(permutation)  # Add to list of tactile sensor mesh edges

        ts_mesh_edges = torch.tensor(ts_mesh_edges).long()  # Long type for indices
        ts_mesh_edges = ts_mesh_edges.unique(dim=-2)  # Eliminate duplicate edges
        # ts_tetras = batch["tetrahedra"]  # Shape: (N_tetras, 4) #TODO
        #
        # # A tetrahedron has 12 directed permutations of edges:
        # # (0,1), (0,2), (0,3), (1,0), (1,2), (1,3), (2,0), (2,1), (2,3), (3,0), (3,1), (3,2)
        # perm_idx = [0, 1, 0, 2, 0, 3, 1, 0, 1, 2, 1, 3, 2, 0, 2, 1, 2, 3, 3, 0, 3, 1, 3, 2]
        # # Gather all permutation options across all elements instantly
        # ts_all_edges = ts_tetras[:, perm_idx].reshape(-1, 2)
        # # Eliminate duplicates globally in C++
        # ts_mesh_edges = torch.unique(ts_all_edges, dim=0)

        ########################################
        ## Object mesh edges
        ########################################
        # Build object mesh edges
        obj_faces = batch["faces"]
        obj_mesh_edges = [] # Initialize object mesh edges

        for obj_face in obj_faces: # Iterate over all object mesh faces
            for permutation in permutations(obj_face, 2): # Iterate over all size-2 permutations of indices in face
                obj_mesh_edges.append(permutation) # Add to list of object mesh edges

        obj_mesh_edges = torch.tensor(obj_mesh_edges).long() # Long type for indices
        obj_mesh_edges = obj_mesh_edges.unique(dim=-2) # Eliminate duplicate edges
        # obj_faces = batch["faces"]  # Shape: (N_faces, 3) #TODO
        #
        # # A triangle face has 6 directed permutations of edges:
        # # (0,1), (0,2), (1,0), (1,2), (2,0), (2,1)
        # face_perm_idx = [0, 1, 0, 2, 1, 0, 1, 2, 2, 0, 2, 1]
        # obj_all_edges = obj_faces[:, face_perm_idx].reshape(-1, 2)
        # obj_mesh_edges = torch.unique(obj_all_edges, dim=0)

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
            relative_template_disps = (batch["template.vertices.positions"][receivers]
                                       - batch["template.vertices.positions"][senders])
            mesh_edge_features = torch.cat([
                relative_template_disps, # Relative displacements
                torch.norm(relative_template_disps, dim=-1, keepdim=True) # Relative displacement norms
            ], dim=-1) # Build mesh edge features
        else: # Otherwise, use first and second frame data
            # Compute first frame relative displacement for each mesh edge
            relative_1st_frame_disps = (batch["1st_frame.vertices.positions"][receivers]
                                       - batch["1st_frame.vertices.positions"][senders])
            # Compute second frame relative displacement for each mesh edge
            relative_2nd_frame_disps = (batch["2nd_frame.vertices.positions"][receivers]
                                        - batch["2nd_frame.vertices.positions"][senders])
            mesh_edge_features = torch.cat([
                relative_1st_frame_disps,  # First frame relative displacements
                torch.norm(relative_1st_frame_disps, dim=-1, keepdim=True),  # First frame relative displacement norms
                relative_2nd_frame_disps,  # First frame relative displacements
                torch.norm(relative_2nd_frame_disps, dim=-1, keepdim=True)  # First frame relative displacement norms
            ], dim=-1)  # Build mesh edge features

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
            # TODO
            # Compute relative displacement for each contact edge at second frame
            relative_2nd_frame_disps = (batch["2nd_frame.vertices.positions"][receivers]
                                        - batch["2nd_frame.vertices.positions"][senders])

            # Compute force feature for each contact edge at current (considered) data point
            force = batch["forces"][idx].item()
            force_per_contact_edge = force / (contact_edges.shape[-2] + 1e-8) # Ensure that there is no divided-by-zero error
            force_features = torch.full(
                (contact_edges.shape[-2], 1),
                force_per_contact_edge,
                device="cuda" #TODO: do not need
            )

            # Build contact edge features
            contact_edge_features = torch.cat([
                relative_2nd_frame_disps,  # Relative displacements
                torch.norm(relative_2nd_frame_disps, dim=-1, keepdim=True),  # Relative displacement norms
                force_features # Force features
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

    @staticmethod
    def _build_node_features(batch: Databatch) -> torch.Tensor:
        # Get node velocities of tactile sensors and object
        # Note that velocities are computed using normal vectors of tactile sensors (left and right)
        node_velocities = [] # Initialize node velocity list
        for idx in torch.unique(batch["datapoints.indices"]): # Iterate over all data points
            node_types = batch["nodes.types"][batch["datapoints.indices"] == idx] # Get node types of current (considered) data point
            n_obj_nodes = int((node_types == NodeType.OBJECT).sum().item())  # Get number of object nodes (vertices)
            n_ts_nodes = len(node_types) - n_obj_nodes  # Get number of tactile sensor nodes
            n_ts_comp_nodes = n_ts_nodes // 2  # Get number of each tactile sensor (component) nodes (left and right)

            ts_left_node_velocities = torch.tile(
                batch["tactile_sensors.normals"][idx, 0, ...],
                (n_ts_comp_nodes, 1)
            ) # Set left tactile sensor node velocities as left tactile sensor normal
            ts_right_node_velocities = torch.tile(
                batch["tactile_sensors.normals"][idx, 1, ...],
                (n_ts_comp_nodes, 1)
            ) # Set right tactile sensor node velocities as right tactile sensor normal
            obj_node_velocities = torch.zeros((n_obj_nodes, 3), device="cuda") # Set object node velocities as zeros

            # Add velocities to node velocity list
            node_velocities.extend([
                ts_left_node_velocities,
                ts_right_node_velocities,
                obj_node_velocities
            ])
            # # Use repeat instead of tiles for faster memory allocation sequences #TODO
            # left_vel = batch["tactile_sensors.normals"][idx, 0, ...].expand(n_ts_comp_nodes, -1)
            # right_vel = batch["tactile_sensors.normals"][idx, 1, ...].expand(n_ts_comp_nodes, -1)
            # obj_vel = torch.zeros((n_obj_nodes, 3), dtype=torch.float32, device="cuda")
            #
            # node_velocities.extend([left_vel, right_vel, obj_vel])

        node_velocities = torch.cat(node_velocities, dim=-2)

        # Get node one-hot encodings of tactile sensors and object
        node_one_hot_encodings = F.one_hot(batch["nodes.types"], NodeType.NUM)

        # Build node features
        node_features = torch.cat([
            node_velocities,
            node_one_hot_encodings
        ], dim=-1)

        return node_features
