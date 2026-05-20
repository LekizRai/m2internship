import torch

from typing import Dict, Any
from itertools import permutations

from commons.config import Config
from commons.processor import Processor
from utils.graph import construct_contact_edges


class GraphBuildingProcessor(Processor):
    def __init__(self, config: Config):
        self._config = config

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        # Build mesh edges
        mesh_edges, mesh_edge_features = self._build_mesh_edges(batch)

        # Build contact edges (world edges)
        contact_edges, contact_edge_features = self._build_contact_edges(batch)

        # Build node features
        return {}

    @staticmethod
    def _build_mesh_edges(batch: Dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
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

        ########################################
        ## Mesh edge features
        ########################################
        # Combine two mesh edge tensors to form the complete edge tensor
        mesh_edges = torch.cat([ts_mesh_edges, obj_mesh_edges], dim=-2)

        # Build tactile sensor mesh edge features
        senders = mesh_edges[..., 0]  # Get mesh edge sender indices
        receivers = mesh_edges[..., 1]  # Get mesh edge receiver indices

        # Compute mesh edge features
        # Compute template relative displacement for each mesh edge
        template_relative_disps = (batch["template.vertices.positions"][receivers]
                               - batch["template.vertices.positions"][senders])
        mesh_edge_features = torch.cat([
            template_relative_disps, # Relative displacements
            torch.norm(template_relative_disps, dim=-1, keepdim=True) # Relative displacement norms
        ], dim=-1) # Build mesh edge features

        return mesh_edges, mesh_edge_features

    def _build_contact_edges(self, batch: Dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        ########################################
        ## Contact edges
        ########################################
        contact_edges = construct_contact_edges()

        ########################################
        ## Contact edge features
        ########################################

        return torch.empty(0), torch.empty(0)


