from data.dgs_dataset.dgs_dataset import DGSDataset
from data.dgs_dataset.dgs_dataset_config import DGSDatasetConfig


# To run this test, use must replace the current path to the path
# pointing to the folder of DGS dataset in your local machine.
# Moreover, your DGS dataset folder must be structured as follows
# <dgs_dataset>
# |-- tactile_sensor.tet
# |-- <dgn_dataset>
# |   |-- <sphere01>
# |   |    |-- sphere01_processed.stl
# |   |    |-- ...
# |   |-- <lemon01>
# |   |    |-- ...
# |   |-- <...>
# |-- <dgs_output>
#      |-- sphere01_....h5
#      |-- lemon01_....h5
#      |-- ...
# where <a> denotes folder "a", a.b denotes file "a.b"

def test_dataset():
    # Initialize dataset configuration
    config = DGSDatasetConfig()

    #################################################
    # Replace this path with your path of DGS dataset
    config.dataset_path = "../dgs_dataset"
    #################################################

    # Test dataset with "sphere01" object
    config.focused_objs = ["sphere01"]
    config.focused_trajs = []
    config.focused_frames = []
    dataset = DGSDataset(config)
    assert len(dataset) == 5000

    # Test dataset with first trajectory (including all frames) of "sphere01" object
    config.focused_objs = ["sphere01"]
    config.focused_trajs = [0]
    config.focused_frames = []
    dataset = DGSDataset(config)
    assert len(dataset) == 50

    # Test dataset with first frame (corresponding to all trajectories) of "sphere01" object
    config.focused_objs = ["sphere01"]
    config.focused_trajs = []
    config.focused_frames = [0]
    dataset = DGSDataset(config)
    assert len(dataset) == 100

    # Test retrieving first data point
    data_point = dataset[0]
    print(data_point["forces"].shape)
    assert data_point is not None


if __name__ == "__main__":
    test_dataset()
    print("Test passed.")