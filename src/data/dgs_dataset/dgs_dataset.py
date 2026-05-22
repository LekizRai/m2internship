import os
import math
import trimesh
import h5py
import torch

from typing import List, Dict
from h5py import File
from torch.utils.data import Dataset

from commons.datatype import (
    Datapoint,
    DatapointInfo,
    Databatch,
    NodeType
)
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
from data.defgraspsim.dgs_dataset_config import DGSDatasetConfig


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

        #######################################################################
        ## Initialization
        #######################################################################
        # Initialize dictionary containing reusable tactile sensor data
        # It is pre-computed in __init__()
        self._ts_reusable_data: Dict[str, torch.Tensor] = {}

        # Initialize dictionary containing reusable object data
        # It is updated when calling _retrieve_datapoint
        self._obj_reusable_data: Dict[str, Dict[str, torch.Tensor]] = {}

        # Initialize a list of all data point information of the dataset
        # Data point information includes object name, corresponding trajectories, frames and DGS output file name
        self._datapoint_infos: List[DatapointInfo] = []
        self._datapoints: List[Datapoint] = []

        #######################################################################
        ## Collect dataset information (all considered data point information)
        #######################################################################
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

                        # Assign datapoint information
                        datapoint_info: DatapointInfo = {
                            "file": file, # DGS output file corresponding to considered object
                            "obj": obj, # Current object name
                            "traj": traj, # Current trajectory
                            "frame": frame # Current frame
                        }

                        # Add datapoint information to list
                        self._datapoint_infos.append(datapoint_info)

        #######################################################################
        ## Pre-compute reusable (tactile sensor) data
        #######################################################################
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
            torch.tensor([-0.0006, 0.0275, 0.099]).reshape(3, 1),  # Convention
        ).T  # Transpose again to obtain original shape

        # Generate the right one
        right_ts_template_verts = do_scale(
            ts_raw_verts.T,  # Transpose for proper operation
            torch.tensor([-1, 0, 0]).reshape(3, 1),
        )  # This step does flip to generate right set of vertice positions
        right_ts_template_verts = do_rotation(
            right_ts_template_verts,
            0, -math.pi / 2, 0,
        )
        right_ts_template_verts = do_translation(
            right_ts_template_verts,
            torch.tensor([-0.0006, -0.0275, 0.099]).reshape(3, 1),  # Convention
        ).T  # Transpose again to obtain original shape

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
        ], dim=-2)  # Tetrahedra

        # Determine tactile sensor node types
        n_ts_verts = ts_template_verts.shape[-2]
        # Initialize all nodes as interior nodes
        ts_node_types = torch.full((n_ts_verts,), NodeType.INTERIOR).long()  # Will be used as indices
        ts_faces, ts_surface_faces = extract_faces_from_tetras(
            tetras=ts_tetras,
            return_surface_faces=True,
        )  # Extract faces from tetrahedra
        ts_surface_vert_idx = ts_surface_faces.flatten().unique()  # Extract all surface vertice indices
        ts_node_types[ts_surface_vert_idx] = NodeType.SURFACE  # Assign surface node type to them

        # Assign data to dictionary
        self._ts_reusable_data["template_verts"] = ts_template_verts
        self._ts_reusable_data["tetras"] = ts_tetras
        self._ts_reusable_data["node_types"] = ts_node_types

    def _retrieve_datapoint(self, datapoint_info: DatapointInfo) -> Datapoint:
        ########################################
        ## Get data point information
        ########################################
        file = datapoint_info["file"]
        file = h5py.File(os.path.join(self._config.dgs_output_path, file), "r") # Open .h5 file
        obj = datapoint_info["obj"]
        traj = datapoint_info["traj"]
        frame = datapoint_info["frame"]

        ########################################
        ## Tactile sensor (ts) data
        ########################################
        # Extract vertice positions from first two frames in considering trajectory
        ts_1st_frame_verts = torch.tensor(file["_1_stacked_positions"][traj, 0, ...]).float()
        ts_2nd_frame_verts = torch.tensor(file["_1_stacked_positions"][traj, 1, ...]).float()

        # Extract vertice positions from current frame in considering trajectory
        ts_verts = torch.tensor(file["_1_stacked_positions"][traj, frame, ...]).float()

        ########################################
        ## (Rigid) object data (not tactile sensor)
        ########################################
        # Compute object data and add them to the dictionary if they have not been collected
        if obj not in self._obj_reusable_data:
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
            obj_node_types = torch.full((n_obj_verts,), NodeType.OBJECT).long() # Will be used as indices

            # Assign data to dictionary
            reusable_data: Dict[str, torch.Tensor] = {
                "template_verts": obj_template_verts,
                "faces": obj_faces,
                "node_types": obj_node_types
            }
            self._obj_reusable_data[obj] = reusable_data

        # Transform data to align with DGS output dataset
        # Objects corresponding to first two frames (using transformation matrices from first two frames)
        obj_1st_frame_verts = transform(
            self._obj_reusable_data[obj]["template_verts"].T, # Transpose for proper operation
            trans_matrix=torch.tensor(file["_1_stacked_object_frame"][traj, 0, ...])
        ).T # Transpose again to obtain original shape
        obj_2nd_frame_verts = transform(
            self._obj_reusable_data[obj]["template_verts"].T, # Transpose for proper operation
            trans_matrix=torch.tensor(file["_1_stacked_object_frame"][traj, 1, ...])
        ).T # Transpose again to obtain original shape

        # Objects corresponding to the current frame (using transformation matrix from current frame)
        obj_verts = transform(
            self._obj_reusable_data[obj]["template_verts"].T, # Transpose for proper operation
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
            tetras=self._ts_reusable_data["tetras"],
            return_n_tetras_per_vert=True
        ) # Compute vertice to tetrahedron relation matrix
        ts_vert_stress_sums = torch.sparse.mm(
            vert_to_tetra_relation,
            torch.tensor(file["_1_stacked_stresses"][traj, frame]).float().unsqueeze(-1)
        ) # Compute at each vertice the sum of stresses from surrounding tetrahedra
        # Compute the average stress value at each vertice
        ts_vert_stresses = torch.div(ts_vert_stress_sums, n_tetras_per_vert.unsqueeze(-1)).squeeze(-1)

        # Extract tactile sensor tetrahedral stress values
        ts_tetras_stresses = torch.tensor(file["_1_stacked_stresses"][traj, frame]).float()

        # Initialize zeros to object vertice stress values
        obj_vert_stresses = torch.zeros(self._obj_reusable_data[obj]["template_verts"].shape[-2]).float()

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
        datapoint: Datapoint = {
            # Template (at-rest) data
            "template.vertices.positions": torch.cat([
                self._ts_reusable_data["template_verts"],
                self._obj_reusable_data[obj]["template_verts"]
            ], dim=-2),

            # First frame
            "1st_frame.vertices.positions": torch.cat([
                ts_1st_frame_verts,
                obj_1st_frame_verts
            ], dim=-2),

            # Second frame
            "2nd_frame.vertices.positions": torch.cat([
                ts_2nd_frame_verts,
                obj_2nd_frame_verts
            ], dim=-2),

            # Forces
            "forces": forces,

            # Tactile sensor normals
            "tactile_sensors.normals": ts_normals,

            # Vertices
            "vertices.positions": torch.cat([
                ts_verts,
                obj_verts
            ], dim=-2),
            "vertices.stresses": torch.cat([
                ts_vert_stresses,
                obj_vert_stresses
            ]),

            # (Tactile sensor) tetrahedra
            "tetrahedra": self._ts_reusable_data["tetras"],
            "tetrahedra.stresses": ts_tetras_stresses,

            # (Object) faces
            # Adding elementwise the number of vertices of tactile sensor to separate two sets
            # of indices (object face indices and tactile sensor tetrahedral indices)
            "faces": self._obj_reusable_data[obj]["faces"]
                     + self._ts_reusable_data["template_verts"].shape[-2],

            # Node types
            "nodes.types": torch.cat([
                self._ts_reusable_data["node_types"],
                self._obj_reusable_data[obj]["node_types"]
            ]),

            # Datapoint index (default value is zero, used for distinguish among datapoints
            # after combining to form data batch
            "datapoints.indices": torch.zeros(ts_verts.shape[-2] + obj_verts.shape[-2])
        }

        return datapoint

    @staticmethod
    def collate(datapoints: List[Datapoint]) -> Databatch:
        vert_template_pos_lst = [] # List to store all template vertice positions
        vert_1st_frame_pos_lst = [] # List to store all first frame vertice positions
        vert_2nd_frame_pos_lst = [] # List to store all second frame vertice positions
        forces_lst = [] # List to store all forces
        ts_normal_lst = [] # List to store all tactile sensor normals (both left and right)
        vert_pos_lst = [] # List to store all current (considered) frame vertice positions
        vert_stress_lst = [] # List to store all vertice stresses
        tetra_lst = [] # List to store all (tactile sensor) tetrahedra
        tetra_stress_lst = [] # List to store all (tactile sensor) tetrahedral stresses
        face_lst = [] # List to store all (object) faces
        node_type_lst = [] # List to store all node types
        datapoint_index_lst = [] # List to store all data point indices

        # Combine all data point to form a data batch (to form a big combined graph later)
        current_node_index_cumul = 0 # This value is used to compute cumulative node indices when combining data points
        for idx, datapoint in enumerate(datapoints):
            vert_template_pos_lst.append(datapoint["template.vertices.positions"])
            vert_1st_frame_pos_lst.append(datapoint["1st_frame.vertices.positions"])
            vert_2nd_frame_pos_lst.append(datapoint["2nd_frame.vertices.positions"])
            forces_lst.append(datapoint["forces"])
            ts_normal_lst.append(datapoint["tactile_sensors.normals"])
            vert_pos_lst.append(datapoint["vertices.positions"])
            vert_stress_lst.append(datapoint["vertices.stresses"])
            # Add cumulative value to separate sets of (tactile sensor) tetrahedra
            tetra_lst.append(datapoint["tetrahedra"] + current_node_index_cumul)
            tetra_stress_lst.append(datapoint["tetrahedra.stresses"])
            # Add cumulative value to separate sets of (object) faces
            face_lst.append(datapoint["faces"] + current_node_index_cumul)
            node_type_lst.append(datapoint["nodes.types"])
            datapoint_index_lst.append(torch.full_like(
                datapoint["datapoints.indices"],
                idx
            ))

            # Update cumulative node index value
            current_node_index_cumul += datapoint["vertices.positions"].shape[-2]

        # Gather all information from all data points together
        batch: Databatch = {
            "template.vertices.positions": torch.cat(vert_template_pos_lst, dim=-2),
            "1st_frame.vertices.positions": torch.cat(vert_1st_frame_pos_lst, dim=-2),
            "2nd_frame.vertices.positions": torch.cat(vert_2nd_frame_pos_lst, dim=-2),
            "forces": torch.tensor(forces_lst),
            "tactile_sensors.normals": torch.stack(ts_normal_lst, dim=-3),
            "vertices.positions": torch.cat(vert_pos_lst, dim=-2),
            "vertices.stresses": torch.cat(vert_stress_lst, dim=-2),
            "tetrahedra": torch.cat(tetra_lst, dim=-2),
            "tetrahedra.stresses": torch.cat(tetra_stress_lst),
            "faces": torch.cat(face_lst, dim=-2),
            "nodes.types": torch.cat(node_type_lst),
            "datapoints.indices": torch.cat(datapoint_index_lst),
        }

        return batch

    def __len__(self) -> int:
        return len(self._datapoint_infos)

    def __getitem__(self, idx: int) -> Datapoint:
        return self._retrieve_datapoint(self._datapoint_infos[idx])
