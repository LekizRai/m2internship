import torch

from torch.utils.data import DataLoader

from models.tacgraspnet.tacgraspnet import TacGraspNet
from models.tacgraspnet.tacgraspnet_config import TacGraspNetConfig
from models.tacgraspnet.tacgraspnet_processor import make_tacgraspnet_processors
from data.dgs_dataset.dgs_dataset import DGSDataset
from data.dgs_dataset.dgs_dataset_config import DGSDatasetConfig
from losses.tacgraspnet.mse import MSE


def test_tacgraspnet_mse_with_dgs_dataset():
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

    # Test MSE loss function
    batch = next(iter(data_loader))
    batch = preprocessor(batch)
    batch = model(batch)
    loss_fn = MSE(model_config)
    loss = loss_fn(batch)
    assert isinstance(loss, torch.Tensor)


if __name__ == "__main__":
    test_tacgraspnet_mse_with_dgs_dataset()
    print("Test passed.")
