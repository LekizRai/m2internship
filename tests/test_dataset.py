from all_datasets.dgs_dataset.dgs_dataset import DGSDataset
from all_datasets.dgs_dataset.dgs_dataset_config import DGSDatasetConfig


def test_dataset():
    # Initialize dataset
    config = DGSDatasetConfig()
    dataset = DGSDataset(config)

if __name__ == "__main__":
    test_dataset()
    print("Test passed.")