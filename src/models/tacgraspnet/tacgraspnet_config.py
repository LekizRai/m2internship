import torch

from typing import Dict, List
from dataclasses import dataclass, field

from commons.config import Config
from commons.datatype import NodeType

# TODO:
#  Predict normalized outputs
#  Examine each data point (whether it is valid or not)

@dataclass
class TacGraspNetConfig(Config):
    ########################################
    ## Important flags
    ########################################
    # Indicate whether the model is run.py or not
    # It is used mainly for normalizer. Normalization is conducted only during run.py process
    is_training: bool = True

    # Indicate whether template data (e.g. vertice positions, ...) are used instead of first frame data or not
    use_template_data: bool = False

    # Indicate whether layer normalization is used for all MLP final layers (except decoder) or not
    use_final_layer_norm: bool = True

    # Indicate whether node and edge features are normalized or not
    normalize_features: bool = True

    # Indicate whether the predicted outputs are normalized or not for loss calculation
    # It is also used to indicate whether the layer normalization is used for decoder final layer or not
    normalize_outputs: bool = True

    # Indicate whether we use two separate or only one (combining) decoder for node and tetrahedral features
    use_node_tetra_separate_decoders: bool = True

    # Indicate whether we use separate MLPs for different edge types in message passing or not
    use_separate_edge_mlps: bool = True

    # Indicate whether each message passing step has its own set of MLPs or not
    # This also means whether we create multiple or only one GraphNetBlock for (performing) each message passing step
    use_message_passing_separate_mlps: bool = True

    # Indicate whether translation inductive bias is used for training and evaluation or not
    use_translation_inductive_bias: bool = False

    ########################################
    ## Modeling configuration
    ########################################

    # Node configuration
    node_feature_dim: int = 3 + NodeType.NUM # Node velocity (3) + number of node types (3) (INTERIOR, SURFACE and OBJECT)

    # Edge configuration
    if use_template_data:
        edge_feature_dims: Dict[str, int] = field(default_factory=lambda: {
            "mesh_edges": 4, # Relative displacement in template (at-rest object) (3) + its norm (1)
            "contact_edges": 5, # Relative displacement in second frame (3) + its norm (1) + applied force in contact (1)
        })
    else:
        edge_feature_dims: Dict[str, int] = field(default_factory=lambda: {
            "mesh_edges": 8, # Relative displacement in first and second frame (6) + their norm (2)
            "contact_edges": 5, # Relative displacement in second frame (3) + its norm (1) + applied force in contact (1)
        })
    edge_types: List[str] = field(default_factory=lambda: ["mesh_edges", "contact_edges"])

    # Tetrahedron configuration
    tetra_feature_dim: int = 1 # Stress (1)

    # Global node configuration
    global_node_feature_dim: int = 0 # TODO

    # Output configuration
    if use_node_tetra_separate_decoders: # Displacement predictions on nodes, stress predictions on tetrahedra
        node_output_dim: int = 3 # Deformation (displacement) (3)
        tetra_output_dim: int = 1 # Stress (1)
    else: # Prediction at each node including displacement and stress
        node_output_dim: int = 4 # Deformation (displacement) (3) + stress (1)

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
        device: str = "cuda"
    else:
        device: str = "cpu"

    ########################################
    ## Training configuration
    ########################################
    # Validation ratio (for training-validation splitting)
    validation_ratio: float = 0.2

    # Batch size
    batch_size: int = 1

    # Number of epochs
    n_epochs: int = 40

    # Optimizer
    optimizer_params: Dict = field(default_factory=lambda: {
        "lr": 1e-4,
    })

    ########################################
    ## Running configuration
    ########################################
    # Running mode (training or evaluation)
    mode: str = "training"

    # Data strategy (single object, multiple objects 1 or multiple objects 2) # TODO
    data_strategy: str = "single_obj"

    # Focused objects (for training or evaluation)
    objs: List[str] = field(default_factory=lambda: ["sphere01"])

    # Validation objects (only for training with multiple objects)
    validation_objs: List[str] = field(default_factory=lambda: ["sphere01"])

    # This attribute is used to store arguments from keyboard
    args = None

    def __post_init__(self):
        self.hidden_dims = [128] * self.n_hidden_layers

    def update(self, args):
        # Important flags
        self.is_training = args.mode == "training"
        self.use_template_data = args.use_template_data
        self.use_final_layer_norm = args.use_final_layer_norm
        self.normalize_features = args.normalize_features
        self.normalize_outputs = args.normalize_outputs
        self.use_node_tetra_separate_decoders = args.use_node_tetra_separate_decoders
        self.use_separate_edge_mlps = args.use_separate_edge_mlps
        self.use_message_passing_separate_mlps = args.use_message_passing_separate_mlps
        self.use_translation_inductive_bias = args.use_translation_inductive_bias

        # Modeling configuration
        # Edge update
        if self.use_template_data:
            self.edge_feature_dims = {
                "mesh_edges": 4,
                "contact_edges": 5,
            }
        else:
            self.edge_feature_dims = {
                "mesh_edges": 8,
                "contact_edges": 5,
            }
        # Node update
        if self.use_node_tetra_separate_decoders:
            self.node_output_dim = 3
            self.tetra_output_dim = 1
        else:
            self.node_output_dim = 4

        # MLP update
        self.n_hidden_layers = args.n_hidden_layers
        self.hidden_dims = [128] * self.n_hidden_layers
        self.latent_dim = args.latent_dim

        # Message passing update
        self.message_passing_steps = args.message_passing_steps

        # Graph building update
        self.radius = args.radius

        # Training update
        self.validation_ratio = args.validation_ratio
        self.batch_size = args.batch_size
        self.n_epochs = args.n_epochs
        self.optimizer_params = {
            "lr": args.learning_rate,
        }

        # Running update
        self.mode = args.mode
        self.data_strategy = args.data_strategy
        self.objs = args.objs

        # Eliminate objects for training out of objects for validation
        self.validation_objs = []
        for obj in args.validation_objs:
            if obj not in self.objs:
                self.validation_objs.append(obj)

        # Store all those from keyboard arguments also
        self.args = args
    