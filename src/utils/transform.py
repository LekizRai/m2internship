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
    x_normal = torch.tensor([1, 0, 0]) # A row vector
    return torch.matmul(trans_matrix[:3, :3].float(), x_normal.float()) # Return a row vector corresponding to x-axis

##################################################

def split_two_fingers(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Split gripper nodes into two fingers using 1D k-means on the best axis.
    Returns absolute indices (left_idx, right_idx) in ORIGINAL order.
    """
    M = A.shape[0]
    device, dtype = A.device, A.dtype

    best_axis, best_score = 0, -1.0
    best_labels = None
    for axis in (0, 1, 2):
        s = A[:, axis]

        # k-means init: farthest scalars
        c0 = s.min()
        c1 = s.max()

        # few iterations are enough in 1D
        for _ in range(10):
            d0 = (s - c0).abs()
            d1 = (s - c1).abs()
            labels = d1 < d0
            # update centroids
            if (~labels).any():
                c0 = s[~labels].mean()
            # if empty, keep previous c0
            if (labels).any():
                c1 = s[labels].mean()
            # if empty, keep previous c1

        n0 = int((~labels).sum().item())
        n1 = int((labels).sum().item())
        if n0 == 0 or n1 == 0:
            continue

        m0 = s[~labels].mean()
        m1 = s[labels].mean()
        s0 = s[~labels].std().clamp_min(1e-12)
        s1 = s[labels].std().clamp_min(1e-12)
        score = (m0 - m1).abs() / (s0 + s1)

        if score.item() > best_score:
            best_score = score.item()
            best_axis = axis
            best_labels = labels

    # Fallback: median split on the axis with largest range
    if best_labels is None:
        ranges = (A.max(0).values - A.min(0).values)
        axis = int(torch.argmax(ranges).item())
        s = A[:, axis]
        thr = s.median()
        best_labels = s > thr
        best_axis = axis

    s = A[:, best_axis]
    m_left = s[~best_labels].mean()
    m_right = s[best_labels].mean()
    # Choose which side is "left" by centroid order along chosen axis
    if m_left <= m_right:
        left_mask = ~best_labels
        right_mask = best_labels
    else:
        left_mask = best_labels
        right_mask = ~best_labels

    left_idx = torch.nonzero(left_mask, as_tuple=False).flatten()
    right_idx = torch.nonzero(right_mask, as_tuple=False).flatten()

    left_idx = torch.sort(left_idx).values.to(device)
    right_idx = torch.sort(right_idx).values.to(device)

    return left_idx, right_idx

def kabsch(A: torch.Tensor, B: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Classic Kabsch: find R,t to align A -> B (Nx3). Returns R (3x3), t (3,).
    Works on the same device/dtype as A.
    """
    assert A.ndim == 2 and B.ndim == 2 and A.shape == B.shape and A.shape[1] == 3
    device, dtype = A.device, A.dtype

    muA = A.mean(dim=0, keepdim=True)
    muB = B.mean(dim=0, keepdim=True)
    Ac = A - muA
    Bc = B - muB

    H = Ac.T @ Bc
    U, S, Vt = torch.linalg.svd(H)
    R = Vt.T @ U.T

    if torch.linalg.det(R) < 0:
        Vt = Vt.clone()
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = (muB - muA @ R.T).squeeze(0)

    return R.to(dtype=dtype, device=device), t.to(dtype=dtype, device=device)