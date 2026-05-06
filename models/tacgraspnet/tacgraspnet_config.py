class TacGraspNetConfig:
    def __init__(self):
        # Node configuration
        self.node_feature_dim = 6 # Node velocity (3) + number of node types (3)

        # Edge configuration
        self.mesh_edge_feature_dim = 4 # Relative displacement in template (rest object) (3) + its norm (1)
        # self.mesh_edge_feature_dim = 8 # Relative displacement in first and second frame (6) + their norm (2)
        self.contact_edge_feature_dim = 5 # Relative displacement in current frame (3) + its norm (1) + applied force in contact (1)

        # Tetrahedron configuration
        self.tetrahedral_feature_dim = 1 # Stress (1)

        # Output configuration
        self.output_dim = 3 # Deformation (acceleration?) (3)
        # self.output_dim = 4 # Deformation (acceleration?) (3) + stress (1)
    