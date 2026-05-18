import math
import torch


def do_rotation(
        x: torch.Tensor, # Need-rotating input tensor (tensor of column vectors)
        r: float, # Rolling angle (for rotation around x-axis)
        p: float, # Pitching angle (for rotation around y-axis)
        y: float # Yawing angle (for rotation around z-axis)
) -> torch.Tensor:
    Rx = torch.tensor([[1, 0, 0],
                       [0, math.cos(r), -math.sin(r)],
                       [0, math.sin(r), math.cos(r)]]).float()
    Ry = torch.tensor([[math.cos(p), 0, math.sin(p)],
                       [0, 1, 0],
                       [-math.sin(p), 0, math.cos(p)]]).float()
    Rz = torch.tensor([[math.cos(y), -math.sin(y), 0],
                       [math.sin(y), math.cos(y), 0],
                       [0, 0, 1]]).float()
    return Rz @ Ry @ Rx @ x.float()

def do_translation(
        x: torch.Tensor, # Need-rotating input tensor (tensor of column vectors)
        t: torch.Tensor # Translation vector
) -> torch.Tensor:
    return x.float() + t.float()

def do_scale(
        x: torch.Tensor, # Need-scaling input tensor (tensor of column vectors)
        s: torch.Tensor # Scale vector
) -> torch.Tensor:
    return x.float() * s.float()

def transform(
        x: torch.Tensor, # Need-transforming input tensor (tensor of column vectors)
        trans_matrix: torch.Tensor # Transformation matrix
) -> torch.Tensor:
    ones = torch.ones(1, x.shape[-1]).float() # Create tensor of ones to form new tensor in homogeneous coordinate
    homo_x = torch.cat([x, ones], dim=-2) # New tensor in homogeneous coordinate
    return trans_matrix[:3, ...].float() @ homo_x.float()

# Assume that object normal is its local x-axis
# and equivalent to how to compute tactile sensor normals
def extract_normal_from_trans_matrix(
        trans_matrix: torch.Tensor # Transformation matrix
) -> torch.Tensor:
    x_normal = torch.tensor([1, 0, 0]).reshape(3, 1)
    trans_matrix_no_translation = trans_matrix[:3, :3]
    return trans_matrix_no_translation.float() @ x_normal.float()
