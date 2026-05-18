import os
import math
import trimesh
import h5py
import torch

from typing import List
from h5py import File
from torch.utils.data import Dataset

from commons.datatype import Datapoint, NodeType
from utils.tetra import (
    parse_tet_file,
    extract_faces_from_tetras,
    compute_vert_to_tetra_relation
)
from utils.transform import (
    do_rotation,
    do_translation,
    do_scale,
    transform,
    extract_normal_from_trans_matrix
)
from all_datasets.dgs_dataset.dgs_dataset_config import DGSDatasetConfig


"""
The DefGraspSim (DGS) dataset records deformations and stresses
in simulation of tactile sensor thick covering layers when robot
arms holds rigid objects whose different shapes (e.g. sphere, lemon,
polygons, ...). The simulation is conducted using 100 different
grasping poses on each object (i.e. trajectories) and each trajectory
is divided into 50 time slices performing temporal increasing of force.
From now on in this implementation, when mentioning tactile sensor,
that means only its deformable covering layer is considered.
"""

class DGSDataset(Dataset):
    def __init__(self, config: DGSDatasetConfig):
        super().__init__()
        self._config = config

        # Initialize a list of all data point of the dataset
        self._datapoints: List[Datapoint] = []

        # Retrieve dataset
        for file in os.listdir(config.dgs_output_path): # Iterate over all (.h5) files (objects) in DGS output directory
            # Continue if file exists and its extension is .h5, otherwise ignore
            if not os.path.isfile(os.path.join(config.dgs_output_path, file)) or not file.endswith(".h5"):
                continue

            # Continue if focused object list is empty or current object exists in it, otherwise ignore
            obj = file.split("_")[0] # Get object name from file name
            if config.focused_objs and not obj in config.focused_objs:
                continue

            with h5py.File(os.path.join(config.dgs_output_path, file), "r") as h5file:  # Open .h5 file
                n_trajs = h5file["_1_stacked_forces"].shape[-2]  # Number of trajectories for each object
                n_frames = h5file["_1_stacked_forces"].shape[-1]  # Number of frames for each trajectory

                for traj in range(n_trajs): # Iterate over all trajectories of current object
                    # Continue if focused trajectory list is empty or current trajectory exists in it, otherwise ignore
                    if config.focused_trajs and not traj in config.focused_trajs:
                        continue

                    for frame in range(n_frames): # Iterate over all frames of current trajectory
                        # Continue if focused frame list is empty or current frame exists in it, otherwise ignore
                        if config.focused_frames and not frame in config.focused_frames:
                            continue

                        # Preprocess data and add processed datapoint to list
                        datapoint = self.preprocess(h5file, obj, traj, frame)
                        self._datapoints.append(datapoint)

    def preprocess(self, file: File, obj: str, traj: int, frame: int) -> Datapoint:
        print(obj, trạ, frame)
        ########################################
        ## Tactile sensor (ts) data
        ########################################
        # Initialize tactile sensor mesh path and check if file exists, otherwise raise FileNotFound error
        ts_mesh_path = os.path.join(self._config.dataset_path, "tactile_sensor.tet")
        if not os.path.isfile(ts_mesh_path):
            raise FileNotFoundError(ts_mesh_path)

        # Extract raw data from .tet file for only one tactile sensor, two tactile sensors facing each other are needed
        ts_raw_verts, ts_raw_tetras = parse_tet_file(ts_mesh_path)

        # Generate two template (at-rest) tactile sensors (left and right)
        # Generate the left one
        left_ts_template_verts = do_rotation(
            ts_raw_verts.T,  # Transpose for proper operation
            -math.pi, -math.pi / 2, 0,
        )
        left_ts_template_verts = do_translation(
            left_ts_template_verts,
            torch.tensor([-0.0006, 0.0275, 0.099]).reshape(3, 1), # Convention
        ).T # Transpose again to obtain original shape

        # Generate the right one
        right_ts_template_verts = do_scale(
            ts_raw_verts.T,  # Transpose for proper operation
            torch.tensor([-1, 0, 0]).reshape(3, 1),
        ) # This step does flip to generate right set of vertice positions
        right_ts_template_verts = do_rotation(
            right_ts_template_verts,
            0, -math.pi / 2, 0,
        )
        right_ts_template_verts = do_translation(
            right_ts_template_verts,
            torch.tensor([-0.0006, -0.0275, 0.099]).reshape(3, 1), # Convention
        ).T # Transpose again to obtain original shape

        # Gather all template (at-rest) information (from two generated tactile sensors)
        # Note that tetrahedral vertice indices are universal (not only valid in template)
        ts_template_verts = torch.cat([
            left_ts_template_verts,
            right_ts_template_verts
        ], dim=-2)  # All template (at-rest) vertice positions (both left and right)
        ts_tetras = torch.cat([
            ts_raw_tetras,
            # Add elementwise the number of vertices of the left one to separate two sets of tetrahedra (left and right)
            ts_raw_tetras + left_ts_template_verts.shape[-2]
        ], dim=-2) # Tetrahedra

        # Determine tactile sensor node types
        n_ts_verts = ts_template_verts.shape[-2]
        # Initialize all nodes as interior nodes
        ts_node_types = torch.full((n_ts_verts, 1), NodeType.INTERIOR).long() # Will be used as indices
        ts_faces, ts_surface_faces = extract_faces_from_tetras(
            tetras=ts_tetras,
            return_surface_faces=True,
        )  # Extract faces from tetrahedra
        ts_surface_vert_idx = ts_surface_faces.flatten().unique()  # Extract all surface vertice indices
        ts_node_types[ts_surface_vert_idx] = NodeType.SURFACE  # Assign surface node type to them

        # Extract vertice positions from first two frames in considering trajectory
        ts_1st_frame_verts = torch.tensor(file["_1_stacked_positions"][traj, 0, ...]).float()
        ts_2nd_frame_verts = torch.tensor(file["_1_stacked_positions"][traj, 1, ...]).float()

        # Extract vertice positions from current frame in considering trajectory
        ts_verts = torch.tensor(file["_1_stacked_positions"][traj, frame, ...]).float()

        ########################################
        ## (Rigid) object data (not tactile sensor)
        ########################################
        # Initialize object mesh path and check if file exists, otherwise raise FileNotFound error
        obj_mesh_path = os.path.join(self._config.dgn_dataset_path, obj, f"{obj}_processed.stl")
        if not os.path.isfile(obj_mesh_path):
            raise FileNotFoundError(obj_mesh_path)

        # Load mesh and retrieve data
        obj_mesh = trimesh.load_mesh(obj_mesh_path)
        obj_template_verts = torch.tensor(obj_mesh.vertices).float() # Template (at-rest) vertice positions
        obj_faces = torch.tensor(obj_mesh.faces).long() # Faces

        # Set object node types
        n_obj_verts = obj_template_verts.shape[-2]
        obj_node_types = torch.full((n_obj_verts, 1), NodeType.OBJECT).long() # Will be used as indices

        # Transform data to align with DGS output dataset
        # Objects corresponding to first two frames (using transformation matrices from first two frames)
        obj_1st_frame_verts = transform(
            obj_template_verts.T, # Transpose for proper operation
            trans_matrix=torch.tensor(file["_1_stacked_object_frame"][traj, 0, ...])
        ).T # Transpose again to obtain original shape
        obj_2nd_frame_verts = transform(
            obj_template_verts.T, # Transpose for proper operation
            trans_matrix=torch.tensor(file["_1_stacked_object_frame"][traj, 1, ...])
        ).T # Transpose again to obtain original shape

        # Objects corresponding to the current frame (using transformation matrix from current frame)
        obj_verts = transform(
            obj_template_verts.T, # Transpose for proper operation
            trans_matrix=torch.tensor(file["_1_stacked_object_frame"][traj, frame, ...])
        ).T # Transpose again to obtain original shape

        ########################################
        ## Force data
        ########################################
        forces = torch.tensor(file["_1_stacked_forces"][traj, frame]).float()

        ########################################
        ## Stress data
        ########################################
        # Extract tactile sensor vertice stress values
        vert_to_tetra_relation, n_tetras_per_vert = compute_vert_to_tetra_relation(
            tetras=ts_tetras,
            return_n_tetras_per_vert=True
        ) # Compute vertice to tetrahedron relation matrix
        ts_vert_stress_sums = torch.sparse.mm(
            vert_to_tetra_relation,
            torch.tensor(file["_1_stacked_stresses"][traj, frame]).float().unsqueeze(-1)
        ) # Compute at each vertice the sum of stresses from surrounding tetrahedra
        # Compute the average stress value at each vertice
        ts_vert_stresses = torch.div(ts_vert_stress_sums, n_tetras_per_vert.unsqueeze(-1))

        # Extract tactile sensor tetrahedral stress values
        ts_tetras_stresses = torch.tensor(file["_1_stacked_stresses"][traj, frame]).float().unsqueeze(1)

        # Initialize zeros to object vertice stress values
        obj_vert_stresses = torch.zeros(n_obj_verts, 1).float()

        ########################################
        ## Normal data (i.e. perpendicular thing)
        ########################################
        obj_normal = extract_normal_from_trans_matrix(
            trans_matrix=torch.tensor(file["_1_stacked_object_frame"][traj, frame, ...])
        )
        ts_normals = torch.stack([-obj_normal, obj_normal]) # Tactile sensor normals (both left and right)

        ########################################
        ## Assign data to data point
        ########################################
        datapoint: Datapoint = dict()

        # Template (at-rest) data
        datapoint["template.vertices.positions"] = torch.cat([
            ts_template_verts,
            obj_template_verts
        ], dim=-2)

        # First frame
        datapoint["1st_frame.vertices.positions"] = torch.cat([
            ts_1st_frame_verts,
            obj_1st_frame_verts
        ], dim=-2)

        # Second frame
        datapoint["2nd_frame.vertices.positions"] = torch.cat([
            ts_2nd_frame_verts,
            obj_2nd_frame_verts
        ], dim=-2)

        # Forces
        datapoint["forces"] = forces

        # Tactile sensor normals
        datapoint["tactile_sensors.normals"] = ts_normals

        # Vertices
        datapoint["vertices.positions"] = torch.cat([
            ts_verts,
            obj_verts
        ], dim=-2)
        datapoint["vertices.stresses"] = torch.cat([
            ts_vert_stresses,
            obj_vert_stresses
        ], dim=-2)

        # (Tactile sensor) tetrahedra
        datapoint["tetrahedra"] = ts_tetras
        datapoint["tetrahedra.stresses"] = ts_tetras_stresses

        # (Object) faces
        # Adding elementwise the number of vertices of tactile sensor to separate two sets
        # of indices (object face indices and tactile sensor tetrahedral indices)
        datapoint["faces"] = obj_faces + n_ts_verts

        # Node types
        datapoint["nodes.types"] = torch.cat([
            ts_node_types,
            obj_node_types
        ], dim=-2)

        return datapoint

    def __len__(self):
        return len(self._datapoints)

    def __getitem__(self, idx):
        return self._datapoints[idx]
