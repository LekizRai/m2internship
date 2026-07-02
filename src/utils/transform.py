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

def pointcloud_transform(A, B):
    """
    Computes best-fit rigid body transformation between point clouds A and B, that is
    the 3D translation and rotation that minimizes least squares error between
    transformed A and B using singular value decomposition.

    Args:
            A
            B

    Returns:
            translation: 3D translation tensor
            rotation: 3x3 rotation matrix tensor
    """
    assert A.shape == B.shape

    A_centroid = torch.mean(A, dim=0)
    B_centroid = torch.mean(B, dim=0)
    A_zero_mean = A - A_centroid
    B_zero_mean = B - B_centroid

    cov = A_zero_mean.T @ B_zero_mean
    U, S, Vh = torch.linalg.svd(cov)

    rotation = Vh.T @ U.T

    if torch.linalg.det(rotation) < 0:
        Vh[-1, :] *= -1

    rotation = Vh.T @ U.T
    translation = B_centroid - rotation @ A_centroid

    return translation, rotation

def pointcloud_tf_feature(
    pointcloud_A: torch.Tensor, pointcloud_B: torch.Tensor
) -> torch.Tensor:
    """
    Constructs a 9D feature for the best-fit rigid transformation between two point
    clouds. Computes the best-fit rigid translation and rotation between the point
    clouds using SVD, then converts the rotation matrix into a continuous 6D
    representation. Returns the concatenation of 3D translation and 6D rotation.

    Args:
            pointcloud_A: Pointcloud to transform from
            pointcloud_B: Pointcloud to transform to

    Returns:
            tf_feature: 9D rigid transformation feature
    """
    translation, rot_matrix = pointcloud_transform(pointcloud_A, pointcloud_B)
    rot_feature = torch_rotation_matrix_to_feature(rot_matrix)
    tf_feature = torch.cat([translation, rot_feature], dim=0)
    assert tf_feature.shape == (9,)
    return tf_feature


def torch_rotation_matrix_to_feature(rot_matrix: torch.Tensor) -> torch.Tensor:
    """
    Extract rotation features according to this paper:
    https://openaccess.thecvf.com/content_CVPR_2019/html/Zhou_On_the_Continuity_of_Rotation_Representations_in_Neural_Networks_CVPR_2019_paper.html
    Unlike the paper though, we include the second and third column instead of the first and second,

    Args:
            rot_matrix ((3,3) Tensor): Rotation matrix to extract feature from
    Returns:
            feature ((6,) Tensor): Extracted 6D rotation feature
    """
    return rot_matrix.T.reshape(-1)[3:]


def torch_rotation_feature_to_matrix(rot_feature: torch.Tensor) -> torch.Tensor:
    """
    Reconstruct valid, orthonormal rotation matrix from a 6D rotation feature.

    Args:
            rot_feature ((6,) Tensor): Input 6D rotation feature
    Returns:
            rot_matrix ((3,3) Tensor): Reconstructed 3x3 rotation matrix
    """
    z_axis_unnorm = rot_feature[3:]
    assert torch.linalg.norm(z_axis_unnorm) > 0
    z_axis = z_axis_unnorm / torch.linalg.norm(z_axis_unnorm)
    y_axis_unnorm = rot_feature[:3] - torch.dot(z_axis, rot_feature[:3]) * z_axis
    assert torch.linalg.norm(y_axis_unnorm) > 0
    y_axis = y_axis_unnorm / torch.linalg.norm(y_axis_unnorm)
    x_axis = torch.linalg.cross(y_axis, z_axis, dim=0)

    return torch.stack([x_axis, y_axis, z_axis], dim=1)