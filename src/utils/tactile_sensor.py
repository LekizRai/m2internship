import torch

import torch.nn.functional as F

# Compute the moving directions of left and right tactile sensors on the gripper (gripper closing directions)
# We use a trick here, which is that the nodes of tactile sensors are always arranged in order from left to right
def compute_gripper_closing_directions(
        ts_verts: torch.Tensor # Vertice positions of tactile sensors
) -> torch.Tensor:
    # Get indices of nodes of the left and right tactile sensors
    n_ts_nodes = ts_verts.shape[0]
    left_idx = list(range(n_ts_nodes // 2))
    right_idx = list(range(n_ts_nodes // 2, n_ts_nodes))

    # We use a trick here, which is that the two sensors are symmetric to each other and grasping process is in parallel style
    # Therefore, the closing directions can be computed using two centroids of two sensors
    left_centroid = torch.mean(ts_verts[left_idx], dim=-2)
    right_centroid = torch.mean(ts_verts[right_idx], dim=-2)
    left_gripper_closing_direction = right_centroid - left_centroid
    right_gripper_closing_direction = left_centroid - right_centroid

    return torch.stack([
        F.normalize(left_gripper_closing_direction, dim=-1),
        F.normalize(right_gripper_closing_direction, dim=-1)
    ])
