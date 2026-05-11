import torch

# TODO: Last decoder layer having normalization of not, normalization or not, separate decoders for node and tetrahedron or not

class TacGraspNetConfig:
    def __init__(self):
        ########################################
        ## General configuration
        ########################################

        # Node configuration
        self.node_feature_dim = 6 # Node velocity (3) + number of node types (3)

        # Edge configuration
        self.edge_feature_dims = {
            "mesh_edges": 4, # Relative displacement in template (rest object) (3) + its norm (1)
            # "mesh": 4, # Relative displacement in first and second frame (6) + their norm (2)
            "contact_edges": 5, # Relative displacement in current frame (3) + its norm (1) + applied force in contact (1)
        }
        self.edge_types = ["mesh_edges", "contact_edges"]

        # Tetrahedron configuration
        self.is_tetra_used = True
        self.tetra_feature_dim = 1 # Stress (1)

        # Global node configuration
        self.global_node_feature_dim = None # TODO

        # Output configuration
        self.node_output_dim = 3 # Deformation (acceleration?) (3)
        self.tetra_output_dim = 1 # Stress (1)
        # self.output_dim = 4 # Deformation (acceleration?) (3) + stress (1)

        # MLPs configuration
        # The same hidden configuration for all updating MLPs in GraphNetBlock
        self.n_hidden_layers = 2
        self.hidden_dims = [128] * self.n_hidden_layers
        # The same latent dimension for all (node, edge, tetrahedral) features in (MLP) encoder
        self.latent_dim = 128

        # Message passing configuration
        self.message_passing_steps = 15

        ########################################
        ## Graph building configuration
        ########################################



        ########################################
        ## Device configuration
        ########################################

        # Set up available device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        ########################################
        ## Training configuration
        ########################################
    