#!/usr/bin/env python3
import argparse
import os
import torch
import vtk
import random
import shutil

import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any
from collections import defaultdict
from torch.utils.data import DataLoader

from src.models.tacgraspnet.tacgraspnet import TacGraspNet
from src.models.tacgraspnet.tacgraspnet_config import TacGraspNetConfig
from src.models.tacgraspnet.tacgraspnet_processor import make_tacgraspnet_processors
from src.data.dgs_dataset.dgs_dataset import DGSDataset
from src.data.dgs_dataset.dgs_dataset_config import DGSDatasetConfig
from src.losses.tacgraspnet.mse import MSE
from src.scores.tacgraspnet.mae import DisplacementMAE, StressMAE
from commons.datatype import (
    Datapoint,
    NodeType
)


def get_data_loader(model_config: TacGraspNetConfig):
    # Initialize dataset config for evaluation data loader
    evaluation_dataset_config = DGSDatasetConfig()

    # Get frames for evaluation
    trajs = list(range(100))
    random.shuffle(trajs)
    evaluation_size = int(100 * model_config.validation_ratio)  # 100 is the number of trajectories (grasping poses) for each object
    evaluation_trajs = trajs[:evaluation_size] # Trajectories for validation

    # Construct evaluation data loader
    evaluation_dataset_config.focused_objs = ["lemon02-hollow"] # [model_config.objs[0]]
    evaluation_dataset_config.focused_trajs = evaluation_trajs

    ####### Restraint evaluation for debugging ############
    evaluation_dataset_config.focused_trajs = [40]
    #####################################################

    # Construct validation data loader
    evaluation_dataset = DGSDataset(evaluation_dataset_config)  # Construct evaluation dataset
    evaluation_loader = DataLoader(
        evaluation_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=DGSDataset.collate,
        num_workers=4,
        pin_memory=True,
    )

    return evaluation_loader

def evaluate(model_config: TacGraspNetConfig):
    # Store all given necessary information
    given_dir = model_config.given_dir
    args_path = os.path.join(model_config.given_dir, "args.pth")
    checkpoint_path = os.path.join(model_config.given_dir, "checkpoint.pth")
    create_visualizations = model_config.create_visualizations

    if os.path.isfile(args_path):
        args = torch.load(args_path)
        model_config = TacGraspNetConfig(**args)
        # Update current model configuration using given argument file
        # model_config.update(args) # Note that after update the given_dir and save_dir have been changed
    else:
        return

    # Set batch size to 1 and initialize the model
    model_config.batch_size = 1
    model = TacGraspNet(model_config)

    # Load the evaluated model using checkpoint
    if os.path.isfile(checkpoint_path):
        model_state_dict, _, epoch = torch.load(checkpoint_path)
        model.load_state_dict(model_state_dict)
    else:
        return

    # Initialize the evaluation data loader
    evaluation_loader = get_data_loader(model_config)

    # Initialize processors
    preprocessor, postprocessor = make_tacgraspnet_processors(model_config)

    # Initialize Loss/Metric functions matching your new mse.py logic
    # Initialize loss and score functions
    loss_fn = MSE(model_config)
    disp_mae = DisplacementMAE(model_config)
    stress_mae = StressMAE(model_config)

    traj_losses = defaultdict(list)
    traj_disp_maes = defaultdict(list)
    traj_stress_maes = defaultdict(list)
    # seen_trajs_per_obj = defaultdict(set)

    # Do evaluation
    model.eval() # Set model to evaluation mode
    for data_point in tqdm(evaluation_loader, desc=f"Evaluation on object {model_config.objs[0]}"):
        with torch.no_grad():
            preprocessed_data_point = preprocessor(data_point)
            prediction = model(preprocessed_data_point)

            # Compute loss and scores
            loss = loss_fn(prediction).item()
            disp_mae_val = disp_mae(prediction).item()
            stress_mae_val = stress_mae(prediction).item()

            # Store loss and score values per trajectory
            traj = data_point["trajectories"][0]
            traj_losses[traj].append(loss)
            traj_disp_maes[traj].append(disp_mae_val)
            traj_stress_maes[traj].append(stress_mae_val)

            # Directory to store losses, scores and visualizations
            visualization_save_dir = os.path.join(given_dir, "visualizations")

            if create_visualizations:
                add_vtk_visualization(model_config, prediction, visualization_save_dir)

    # Create file to store metrics
    metrics_file = os.path.join(given_dir, "metrics.txt")
    if os.path.exists(metrics_file): # Delete the existing file if it exists
        os.unlink(metrics_file)

    # Write metrics to file
    with open(metrics_file, "w") as metrics:
        for traj in traj_losses.keys():
            mean_loss = float(np.mean(traj_losses[traj]))
            mean_disp_mae = float(np.mean(traj_disp_maes[traj]))
            mean_stress_mae = float(np.mean(traj_stress_maes[traj]))
            metrics.write(f"{traj}\n")
            metrics.write(
                f"Mean loss {mean_loss:.6f}, mean deformation MAE {mean_disp_mae:.6f}, mean stress MAE {mean_stress_mae:.6f}\n"
            )

def add_vtk_visualization(
        config: TacGraspNetConfig,
        prediction: Datapoint,
        save_dir: str
):
    # Set save directory
    vtk_save_dir = os.path.join(save_dir, prediction["objects"][0], str(prediction["trajectories"][0]))
    if not os.path.exists(vtk_save_dir): # Create a new folder if it does not exist
        os.makedirs(vtk_save_dir)

    # Add ground truth visualizations
    add_tactile_sensors_visualization(config, prediction, False, vtk_save_dir)
    add_object_visualization(config, prediction, False, vtk_save_dir)

    # Add prediction visualization
    add_tactile_sensors_visualization(config, prediction, True, vtk_save_dir)
    add_object_visualization(config, prediction, True, vtk_save_dir)

def add_tactile_sensors_visualization(
        config: TacGraspNetConfig,
        prediction: Datapoint,
        use_prediction: bool,
        save_dir: str
):
    # Create mask for tactile sensor vertices
    ts_mask = (prediction["nodes.types"] != NodeType.OBJECT).flatten()

    # Create list of points and add tactile sensor vertices (points) to that list
    ts_points = vtk.vtkPoints()
    if use_prediction:
        ts_verts = prediction["predictions.vertices.positions"][ts_mask]
    else:
        ts_verts = prediction["vertices.positions"][ts_mask]
    for ts_vert in ts_verts:
        ts_points.InsertNextPoint(tuple(ts_vert))

    # Create unstructured grid for tactile sensor visualization
    ts_ugrid = vtk.vtkUnstructuredGrid()
    ts_ugrid.SetPoints(ts_points)

    # Add tetrahedra to the unstructured grid
    ts_tetras = prediction["tetrahedra"] # Extract all the tactile sensor tetrahedra
    for ts_tetra in ts_tetras:
        ts_vtk_tetra = vtk.vtkTetra()
        for idx, vert in enumerate(ts_tetra):
            ts_vtk_tetra.GetPointIds().SetId(idx, vert)
        ts_ugrid.InsertNextCell(ts_vtk_tetra.GetCellType(), ts_vtk_tetra.GetPointIds())

    ########################################
    ## Displacement length
    ########################################
    # Initialize displacement length array
    disp_length_array = vtk.vtkFloatArray()
    disp_length_array.SetName("Displacement Length (mm)")
    disp_length_array.SetNumberOfComponents(1)

    # Initialize starting positions
    if config.use_template_data: # Use template data if flag is true
        ts_starting_pos = prediction["template.vertices.positions"][ts_mask]
    else: # Otherwise use vertice positions from 2nd frame instead
        ts_starting_pos = prediction["2nd_frame.vertices.positions"][ts_mask]

    if use_prediction: # Visualize prediction if flag is true
        disp_lengths = torch.linalg.norm(
            prediction["predictions.vertices.positions"][ts_mask] - ts_starting_pos,
            dim=-1,
        )
    else: # Otherwise visualize ground truth
        disp_lengths = torch.linalg.norm(
            prediction["vertices.positions"][ts_mask] - ts_starting_pos,
            dim=-1,
        )

    # Set displacement length array
    for disp_length in disp_lengths:
        disp_length_array.InsertNextValue(disp_length * 1000.0)
    ts_ugrid.GetPointData().AddArray(disp_length_array)
    # ts_ugrid.GetPointData().SetScalars(disp_length_array) # Visualization focuses on stresses, not displacements

    ########################################
    ## Stress
    ########################################
    # Initialize stress array
    stress_array = vtk.vtkFloatArray()
    stress_array.SetName("Von Mises Stress (kPa)")
    stress_array.SetNumberOfComponents(1)

    if config.use_node_tetra_separate_decoders: # Set stresses on tetrahedra if flag is true
        if use_prediction:
            stresses = prediction["predictions.tetrahedra.stresses"]
        else:
            stresses = prediction["tetrahedra.stresses"]

        # Set stress array
        for stress in stresses:
            stress_array.InsertNextValue(stress / 1000.0)
        ts_ugrid.GetCellData().AddArray(stress_array)
        ts_ugrid.GetCellData().SetScalars(stress_array) # Focus on stresses
    else: # Otherwise set stresses on vertices
        if use_prediction:
            stresses = prediction["predictions.vertices.stresses"]
        else:
            stresses = prediction["vertices.stresses"]

        # Set stress array
        for stress in stresses:
            stress_array.InsertNextValue(stress / 1000.0)
        ts_ugrid.GetPointData().AddArray(stress_array)
        ts_ugrid.GetPointData().SetScalars(stress_array)

    # Write VTK file
    writer = vtk.vtkUnstructuredGridWriter()
    visualization_type = "prediction" if use_prediction else "ground_truth"
    frame = prediction["frames"][0]
    writer.SetFileName(os.path.join(save_dir, f"ts_{visualization_type}_{frame}.vtk"))
    writer.SetInputData(ts_ugrid)
    writer.Write()

def add_object_visualization(
        config: TacGraspNetConfig,
        prediction: Datapoint,
        use_prediction: bool,
        save_dir: str
):
    # Create mask for tactile sensor vertices
    obj_mask = (prediction["nodes.types"] == NodeType.OBJECT).flatten()

    # Create list of points and add object vertices (points) to that list
    obj_points = vtk.vtkPoints()
    if use_prediction:
        obj_verts = prediction["predictions.vertices.positions"][obj_mask]
    else:
        obj_verts = prediction["vertices.positions"][obj_mask]
    for obj_vert in obj_verts:
        obj_points.InsertNextPoint(tuple(obj_vert))

    # Create unstructured grid for object visualization
    obj_ugrid = vtk.vtkUnstructuredGrid()
    obj_ugrid.SetPoints(obj_points)

    # Add faces to the unstructured grid
    n_ts_verts = torch.sum(~obj_mask).item()
    obj_faces = prediction["faces"] - n_ts_verts # Extract and re-index all the object faces
    for obj_face in obj_faces:
        if len(obj_face) == 3:  # Ensure it is a valid triangle
            obj_vtk_face = vtk.vtkTriangle()
            for idx, vert in enumerate(obj_face):
                obj_vtk_face.GetPointIds().SetId(idx, vert)
            obj_ugrid.InsertNextCell(obj_vtk_face.GetCellType(), obj_vtk_face.GetPointIds())

    writer = vtk.vtkUnstructuredGridWriter()
    visualization_type = "prediction" if use_prediction else "ground_truth"
    frame = prediction["frames"][0]
    writer.SetFileName(os.path.join(save_dir, f"obj_{visualization_type}_{frame}.vtk"))
    writer.SetInputData(obj_ugrid)
    writer.Write()


if __name__ == "__main__":
    evaluation_config = TacGraspNetConfig()
    evaluate(evaluation_config)