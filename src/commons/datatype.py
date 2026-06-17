import enum
import torch

from typing import Dict, Any


# Define Datapoint type representing one data point
# In the sense of TacGraspNet, it corresponds to a graph
Datapoint = Dict[str, torch.Tensor]

# Define Databatch type representing one batch of data points
# In the sense of TacGraspNet, it corresponds to the big graph
# composed of all graphs (data points)
Databatch = Dict[str, torch.Tensor]

# Define DatapointInfo type representing information of one data point
DatapointInfo = Dict[str, Any]

# Define NodeType type representing type of nodes in graphs
class NodeType(enum.IntEnum):
    INTERIOR = 0  # Interior nodes of tactile sensors
    OBJECT = 1  # Nodes of the undeformable object
    SURFACE = 2  # Surface nodes of tactile sensors
    NUM = 3  # Number of node types

# Universal boolean data type for running arguments
# Used to convert both strings and boolean values to boolean values
def universal_bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "yes", "1"):
        return True
    if v.lower() in ("false", "no", "0"):
        return False
    raise ValueError('Boolean value are expected (e.g. true, False, Yes, no, 0, 1)')
