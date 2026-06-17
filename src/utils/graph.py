import torch

# from torch_cluster import radius as R

from commons.datatype import NodeType


def construct_contact_edges(
        verts: torch.Tensor,
        node_types: torch.Tensor,
        radius: float
) -> torch.Tensor:
    # print(verts.device)  # TODO
    # print(node_types.device)  # TODO
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
    # 1. Separate surface vertices and object vertices using boolean masks
    ts_surface_mask = (node_types == NodeType.SURFACE)
    obj_mask = (node_types == NodeType.OBJECT)

    ts_surface_vert_mapping = torch.where(ts_surface_mask)[0]
    obj_verts_mapping = torch.where(obj_mask)[0]

    # Quick exit check: if either group has no nodes, return an empty edge tensor safely
    if len(ts_surface_vert_mapping) == 0 or len(obj_verts_mapping) == 0:
        return torch.empty((0, 2), dtype=torch.long, device=verts.device)

    # 2. Extract positions based on the index mappings
    ts_surface_verts = verts[ts_surface_vert_mapping]
    obj_verts = verts[obj_verts_mapping]

    # 3. OPTIMIZATION FOR PASCAL HARDWARE:
    # We use standard cdist without restrictive flags, allowing the runtime to leverage
    # optimized BLAS matrix-multiplication paths that fit the P4000 architecture best.
    pairwise_dist = torch.cdist(ts_surface_verts, obj_verts, p=2.0)

    # 4. Filter indices that fall within the threshold radius ball
    ts_surface_indices, obj_indices = torch.where(pairwise_dist < radius)

    # Map back to global graph layout identifiers
    ts_surface_vert_indices = ts_surface_vert_mapping[ts_surface_indices]
    obj_vert_indices = obj_verts_mapping[obj_indices]

    # 5. Direct structural stack instead of multiple unsqueeze and cat copies
    # Directed forward links (Tactile Sensor -> Object)
    first_contact_edges = torch.stack([ts_surface_vert_indices, obj_vert_indices], dim=-1)

    # Directed backward links (Object -> Tactile Sensor)
    second_contact_edges = torch.stack([obj_vert_indices, ts_surface_vert_indices], dim=-1)

    print("############## Contact edges")
    print(first_contact_edges.shape)
    print(second_contact_edges.shape)
    print("############## Contact edges")

    # Combine into a single unified interaction edge index matrix
    return torch.cat([first_contact_edges, second_contact_edges], dim=0)
