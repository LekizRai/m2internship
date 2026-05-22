import torch

from torch.utils.data import DataLoader

from commons.datatype import NodeType
from models.tacgraspnet.tacgraspnet import TacGraspNet
from models.tacgraspnet.tacgraspnet_config import TacGraspNetConfig
from models.tacgraspnet.tacgraspnet_processor import make_tacgraspnet_processors
from data.dgs_dataset.dgs_dataset import DGSDataset
from data.dgs_dataset.dgs_dataset_config import DGSDatasetConfig


def test_make_tacgraspnet_processors_with_self_defined_data():
    # Initialize data batch
    B = 2  # Batch size
    N_TETRAS = 4  # Number of tetrahedra
    TS_V = 8  # Number of tactile sensor vertices
    OBJ_V = 3  # Number of object vertices

    batch = {
        # Template (at-rest) data
        "template.vertices.positions": torch.rand(B * (TS_V + OBJ_V), 3),

        # First frame
        "1st_frame.vertices.positions": torch.rand(B * (TS_V + OBJ_V), 3),

        # Second frame
        "2nd_frame.vertices.positions": torch.rand(B * (TS_V + OBJ_V), 3),

        # Forces
        "forces": torch.tensor([1., 2.]),

        # Tactile sensor normals
        "tactile_sensors.normals": torch.rand(B, 2, 3),

        # Vertices
        "vertices.positions": torch.rand(B * (TS_V + OBJ_V), 3),
        "vertices.stresses": torch.rand(B * (TS_V + OBJ_V), 1),

        # (Tactile sensor) tetrahedra
        "tetrahedra": torch.tensor([
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [11, 12, 13, 14],
            [15, 16, 17, 18]
        ]).long(),
        "tetrahedra.stresses": torch.rand(N_TETRAS, 1),

        # (Object) faces
        # Adding elementwise the number of vertices of tactile sensor to separate two sets
        # of indices (object face indices and tactile sensor tetrahedral indices)
        "faces": torch.tensor([
            [8, 9, 10],
            [19, 20, 21]
        ]).long(),

        # Node types
        "nodes.types": torch.tensor([
            NodeType.SURFACE, NodeType.SURFACE, NodeType.SURFACE, NodeType.SURFACE,
            NodeType.SURFACE, NodeType.SURFACE, NodeType.SURFACE, NodeType.SURFACE,
            NodeType.OBJECT, NodeType.OBJECT, NodeType.OBJECT,
            NodeType.SURFACE, NodeType.SURFACE, NodeType.SURFACE, NodeType.SURFACE,
            NodeType.SURFACE, NodeType.SURFACE, NodeType.SURFACE, NodeType.SURFACE,
            NodeType.OBJECT, NodeType.OBJECT, NodeType.OBJECT,
        ]).long(),

        # Datapoint index (default value is zero, used for distinguish among datapoints
        # after combining to form data batch
        "datapoints.indices": torch.tensor([
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        ]).long()
    }

    # Initialize processors
    config = TacGraspNetConfig()
    preprocessor, postprocessor = make_tacgraspnet_processors(config)

    # Initialize model
    model = TacGraspNet(config)

    # Test preprocessor __call__
    batch = preprocessor(batch)
    assert batch is not None

    # Test TacGraspNet forward
    batch = model(batch)
    assert batch is not None

    # Test postprocessor __call__
    batch = postprocessor(batch)
    assert batch is not None


def test_tacgraspnet_make_processors_with_dgs_dataset():
    # Initialize dataset and data loader
    model_config = TacGraspNetConfig()
    dataset_config = DGSDatasetConfig()
    dataset = DGSDataset(dataset_config)
    data_loader = DataLoader(
        dataset,
        batch_size=model_config.batch_size,
        shuffle=True,
        collate_fn=dataset.collate
    )

    # Initialize processors
    preprocessor, postprocessor = make_tacgraspnet_processors(model_config)

    # Initialize model
    model = TacGraspNet(model_config)

    # Test preprocessor __call__
    batch = next(iter(data_loader))
    batch = preprocessor(batch)
    assert batch is not None

    # Test TacGraspNet forward
    batch = model(batch)
    assert batch is not None

    # Test postprocessor __call__
    batch = postprocessor(batch)
    assert batch is not None


if __name__ == "__main__":
    test_make_tacgraspnet_processors_with_self_defined_data()
    test_tacgraspnet_make_processors_with_dgs_dataset()
    print("Test passed.")