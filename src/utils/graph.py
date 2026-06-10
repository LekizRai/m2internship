import torch

from torch_cluster import radius as R

from commons.datatype import NodeType


def construct_contact_edges(
        verts: torch.Tensor,
        node_types: torch.Tensor,
        radius: float
) -> torch.Tensor:
    # ts_surface_vert_mapping = torch.where(node_types == NodeType.SURFACE)[0] # Get tactile sensor surface vertice indices
    # obj_verts_mapping = torch.where(node_types == NodeType.OBJECT)[0] # Get object vertice indices
    #
    # ts_surface_verts = verts[ts_surface_vert_mapping] # Extract tactile sensor surface vertices
    # obj_verts = verts[obj_verts_mapping] # Extract object vertices
    #
    # # Compute pairwise distances between tactile sensor surface vertices and object vertices
    # pairwise_dist = ((ts_surface_verts[:, None, ...] - obj_verts[None, ...]) ** 2).sum(dim=-1)
    #
    # # Get index pairs of vertices inside the same ball (neighborhood) of given radius
    # ts_surface_indices, obj_indices = torch.where(pairwise_dist <= radius ** 2)
    # ts_surface_vert_indices = ts_surface_vert_mapping[ts_surface_indices].unsqueeze(-1)
    # obj_vert_indices = obj_verts_mapping[obj_indices].unsqueeze(-1)
    #
    # # Construct contact edges
    # first_contact_edges = torch.cat([
    #     ts_surface_vert_indices,
    #     obj_vert_indices
    # ], dim=-1) # Construct contact edge from tactile sensor surface to object
    # second_contact_edges = torch.cat([
    #     obj_vert_indices,
    #     ts_surface_vert_indices
    # ], dim=-1) # Construct contact edge from object to tactile sensor surface
    # contact_edges = torch.cat([
    #     first_contact_edges,
    #     second_contact_edges
    # ], dim=-2)
    #
    # return contact_edges
    print(verts.device) # TODO
    ts_idx = torch.where(node_types == NodeType.SURFACE)[0]
    obj_idx = torch.where(node_types == NodeType.OBJECT)[0]

    # Spatial tree bucket query runs in O(N log N) instead of dense O(N^2) search!
    # Returns a (2, N_edges) tensor containing exact connected indices
    edge_index = R(verts[obj_idx], verts[ts_idx], r=radius)

    # Remap back to global block tags and format as (N_edges, 2)
    fwd = torch.stack([ts_idx[edge_index[0]], obj_idx[edge_index[1]]], dim=-1)
    bwd = torch.stack([obj_idx[edge_index[1]], ts_idx[edge_index[0]]], dim=-1)
    return torch.cat([fwd, bwd], dim=0)
