import enum
import torch

from typing import Dict


# Define Datapoint type representing one data point
# In the sense of TacGraspNet, it corresponds to a graph
Datapoint = Dict[str, torch.Tensor]

# Define Databatch type representing one batch of data points
# In the sense of TacGraspNet, it corresponds to the big graph
# composed of all graphs (data points)
Databatch = Dict[str, torch.Tensor]

# Define NodeType type representing type of nodes in graphs
class NodeType(enum.IntEnum):
    INTERIOR = 0  # Interior nodes of tactile sensors
    OBJECT = 1  # Nodes of the undeformable object
    SURFACE = 2  # Surface nodes of tactile sensors
    # SIZE = 3  # in one-hot encoding

