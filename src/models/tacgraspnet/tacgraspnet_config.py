from typing import Dict, List
from dataclasses import dataclass, field
import torch

from commons.config import Config

# TODO:
#  "Sample transform" for undeformed tactile sensor vertices
#  Use second frame to compute relative displacements and construct world edges (check)
#  Use first and second frames for at-rest data
#  Last decoder layer having normalization of not
#  Normalization or not
#  Separate decoders for node and tetrahedron or not
#  Examine each data point (whether it is valid or not)

@dataclass
class TacGraspNetConfig(Config):
    ########################################
    ## Important flags
    ########################################
    # Indicate whether the model is training or not
    # It is used mainly for normalizer. Normalization is conducted only during training process
    is_training: bool = True

    # Indicate whether template data (e.g. vertice positions, ...) are used instead of first frame data or not
    use_template_data: bool = True

    # Indicate whether features are normalized or not
    normalize_features: bool = True

    # Indicate whether we use two separate or only one (combining) decoder for node and tetrahedral features
    use_node_tetra_separate_decoders: bool = True

    # Indicate whether we use separate MLPs for different edge types in message passing or not
    use_separate_edge_mlps: bool = True

    # Indicate whether each message passing step has its own set of MLPs or not
    # This also means whether we create multiple or only one GraphNetBlock for (performing) each message passing step
    use_message_passing_separate_mlps: bool = True

    ########################################
    ## Modeling configuration
    ########################################

    # Node configuration
    node_feature_dim: int = 6 # Node position (3) + number of node types (3)

    # Edge configuration
    edge_feature_dims: Dict[str, int] = field(default_factory=lambda: {
        "mesh_edges": 4, # Relative displacement in template (at-rest object) (3) + its norm (1)
        # "mesh_edges": 8, # Relative displacement in first and second frame (6) + their norm (2)
        "contact_edges": 5, # Relative displacement in second frame (3) + its norm (1) + applied force in contact (1)
    })
    edge_types: List[str] = field(default_factory=lambda: ["mesh_edges", "contact_edges"])

    # Tetrahedron configuration
    tetra_feature_dim: int = 1 # Stress (1)

    # Global node configuration
    global_node_feature_dim: int = 0 # TODO

    # Output configuration
    node_output_dim: int = 3 # Deformation (displacement) (3)
    tetra_output_dim: int = 1 # Stress (1)
    # self.output_dim = 4 # Deformation (displacement) (3) + stress (1)

    # MLPs configuration
    # The same hidden configuration for all updating MLPs in GraphNetBlock
    n_hidden_layers: int = 2
    hidden_dims: List[int] = field(default_factory=lambda: []) # To be defined later in __post_init__
    # The same latent dimension for all (node, edge, tetrahedral) features in (MLP) encoder
    latent_dim: int = 128

    # Message passing configuration
    message_passing_steps: int = 15

    ########################################
    ## Graph building configuration
    ########################################

    radius: float = 0.005 # Radius of the ball (neighborhood) for construction of contact edges (world edges)

    ########################################
    ## Device configuration
    ########################################

    # Set up available device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    ########################################
    ## Training configuration
    ########################################
    # Batch size
    batch_size: int = 1

    # Number of epochs
    n_epochs: int = 1

    # Optimizer

    def __post_init__(self):
        self.hidden_dims = [128] * self.n_hidden_layers
    