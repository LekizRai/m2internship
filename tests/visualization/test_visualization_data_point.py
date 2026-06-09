from torch.utils.data import DataLoader

from data.dgs_dataset.dgs_dataset import DGSDataset
from data.dgs_dataset.dgs_dataset_config import DGSDatasetConfig
from visualization.visualize_data_point import visualize_data_point


def test_visualization_data_point():
    # Initialize dataset and data loader
    dataset_config = DGSDatasetConfig()
    dataset_config.focused_objs = ["sphere01"] # Focus on one object only
    dataset_config.focused_trajs = [45] # Focus on one grasping pose only (i.e. one trajectory)
    dataset_config.focused_frames = [49] # Focus on one force value only (i.e. one frame)
    dataset = DGSDataset(dataset_config)
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=dataset.collate
    )

    # Test visualization
    data_point = next(iter(data_loader))
    visualize_data_point(data_point, opts=0)
