import torch

from commons.datatype import NodeType


def construct_contact_edges(
        verts: torch.Tensor,
        node_types: torch.Tensor,
        radius: float
) -> torch.Tensor:
    ts_surface_vert_mapping = torch.where(node_types == NodeType.SURFACE)[0] # Get tactile sensor surface vertice indices
    obj_verts_mapping = torch.where(node_types == NodeType.OBJECT)[0] # Get object vertice indices

    ts_surface_verts = verts[ts_surface_vert_mapping] # Extract tactile sensor surface vertices
    obj_verts = verts[obj_verts_mapping] # Extract object vertices

    # Compute pairwise distances between tactile sensor surface vertices and object vertices
    pairwise_dist = ((ts_surface_verts[:, None, ...] - obj_verts[None, ...]) ** 2).sum(dim=-1)

    # Get index pairs of vertices inside the same ball (neighborhood) of given radius
    ts_surface_indices, obj_indices = torch.where(pairwise_dist <= radius ** 2)
    ts_surface_vert_indices = ts_surface_vert_mapping[ts_surface_indices].unsqueeze(-1)
    obj_vert_indices = obj_verts_mapping[obj_indices].unsqueeze(-1)

    # Construct contact edges
    first_contact_edges = torch.cat([
        ts_surface_vert_indices,
        obj_vert_indices
    ], dim=-1) # Construct contact edge from tactile sensor surface to object
    second_contact_edges = torch.cat([
        obj_vert_indices,
        ts_surface_vert_indices
    ], dim=-1) # Construct contact edge from object to tactile sensor surface
    contact_edges = torch.cat([
        first_contact_edges,
        second_contact_edges
    ], dim=-2)

    return contact_edges
