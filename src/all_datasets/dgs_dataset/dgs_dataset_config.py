import os

from typing import List
from dataclasses import dataclass, field

from commons.config import Config


"""
The DefGraspSim (DGS) dataset records deformations and stresses
in simulation of tactile sensor thick covering layers when robot
arms holds rigid objects whose different shapes (e.g. sphere, lemon,
polygons, ...). The simulation is conducted using 100 different
grasping poses on each object (i.e. trajectories) and each trajectory
is divided into 50 time slices performing temporal increasing of force.
"""

@dataclass
class DGSDatasetConfig(Config):
    ########################################
    ## Important flags
    ########################################
    # We consider all flags following priority order: object -> trajectory -> frame.
    # If all flags are None, we consider the whole dataset

    # Indicate whether we consider on specified list of objects or not and which objects
    # Empty list means this option is deprecated
    focused_objs: List[str] = field(default_factory=lambda: [])

    # Indicate whether we consider on specified list of trajectories or not and which trajectories
    # Empty list means this option is deprecated
    focused_trajs: List[int] = field(default_factory=lambda: [])

    # Indicate whether we consider on specified list of frames or not and which frames
    # Empty list means this option is deprecated
    focused_frames: List[int] = field(default_factory=lambda: [])

    ########################################
    ## Path configuration
    ########################################
    # Directory of the whole dataset
    dataset_path: str = "/home/lekizrai/m2internship/dgs_dataset"
    # Directory of DefGraspNet (DGN) dataset which contains some important information
    dgn_dataset_path: str = os.path.join(dataset_path, "dgn_dataset")
    # Directory of simulation outputs (.h5 files) by DefGraspSim
    dgs_output_path: str = os.path.join(dataset_path, "dgs_output")
