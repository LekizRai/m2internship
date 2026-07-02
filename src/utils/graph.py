import torch

from commons.datatype import NodeType


def construct_contact_edges(
        verts: torch.Tensor,
        node_types: torch.Tensor,
        radius: float
) -> torch.Tensor:
    # Separate tactile sensor surface vertices and object vertices using boolean masks
    ts_surface_mask = (node_types == NodeType.SURFACE)
    obj_mask = (node_types == NodeType.OBJECT)

    ts_surface_vert_mapping = torch.where(ts_surface_mask)[0]
    obj_verts_mapping = torch.where(obj_mask)[0]

    # If either group has no nodes, return an empty edge tensor safely
    if len(ts_surface_vert_mapping) == 0 or len(obj_verts_mapping) == 0:
        return torch.empty((0, 2), dtype=torch.long, device=verts.device)

    # Extract vertice positions based on the computed index mappings
    ts_surface_verts = verts[ts_surface_vert_mapping]
    obj_verts = verts[obj_verts_mapping]

    # Compute pairwise distances and determine neighborhood formed by a small ball of a given radius
    pairwise_dist = torch.cdist(ts_surface_verts, obj_verts, p=2.0)
    ts_surface_indices, obj_indices = torch.where(pairwise_dist <= radius)

    # Compute corresponding index pairs of edges
    ts_surface_vert_indices = ts_surface_vert_mapping[ts_surface_indices]
    obj_vert_indices = obj_verts_mapping[obj_indices]

    # Construct bidirectional contact edges
    first_contact_edges = torch.stack([ts_surface_vert_indices, obj_vert_indices], dim=-1)
    second_contact_edges = torch.stack([obj_vert_indices, ts_surface_vert_indices], dim=-1)
    contact_edges = torch.cat([first_contact_edges, second_contact_edges], dim=-2)

    return contact_edges
