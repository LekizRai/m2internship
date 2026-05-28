import torch

from torch.utils.data import DataLoader

from models.tacgraspnet.tacgraspnet import TacGraspNet
from models.tacgraspnet.tacgraspnet_config import TacGraspNetConfig
from models.tacgraspnet.tacgraspnet_processor import make_tacgraspnet_processors
from data.dgs_dataset.dgs_dataset import DGSDataset
from data.dgs_dataset.dgs_dataset_config import DGSDatasetConfig
from scores.tacgraspnet.mae import DisplacementMAE, StressMAE
from scores.tacgraspnet.r2 import DisplacementR2PerSample, StressR2PerSample


def test_tacgraspnet_r2_with_dgs_dataset():
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

    # Do TacGraspNet forward
    batch = next(iter(data_loader))
    batch = preprocessor(batch)
    batch = model(batch)

    # Test R2 score functions
    disp_score_fn = DisplacementR2PerSample(model_config) # Displacement score function
    stress_score_fn = StressR2PerSample(model_config)  # Stress score function
    disp_score = disp_score_fn(batch)
    stress_score = stress_score_fn(batch)
    assert isinstance(disp_score, torch.Tensor)
    assert isinstance(stress_score, torch.Tensor)

def test_tacgraspnet_mae_with_dgs_dataset():
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

    # Do TacGraspNet forward
    batch = next(iter(data_loader))
    batch = preprocessor(batch)
    batch = model(batch)

    # Test MAE score functions
    disp_score_fn = DisplacementMAE(model_config)  # Displacement score function
    stress_score_fn = StressMAE(model_config)  # Stress score function
    disp_score = disp_score_fn(batch)
    stress_score = stress_score_fn(batch)
    assert isinstance(disp_score, torch.Tensor)
    assert isinstance(stress_score, torch.Tensor)


if __name__ == "__main__":
    test_tacgraspnet_r2_with_dgs_dataset()
    print("Test passed.")
