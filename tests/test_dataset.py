from src.datasets.dgs_dataset.dgs_dataset import DGSDataset
from src.datasets.dgs_dataset.dgs_dataset_config import DGSDatasetConfig


def test_dataset():
    # Initialize dataset
    config = DGSDatasetConfig()
    dataset = DGSDataset(config)
