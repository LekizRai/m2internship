import random
import time
import wandb
import torch

from tqdm import tqdm
from datetime import datetime
from torch.optim import Adam
from torch.utils.data import DataLoader

from data.dgs_dataset.dgs_dataset_config import DGSDatasetConfig
from data.dgs_dataset.dgs_dataset import DGSDataset
from models.tacgraspnet.tacgraspnet_config import TacGraspNetConfig
from models.tacgraspnet.tacgraspnet_processor import make_tacgraspnet_processors
from models.tacgraspnet.tacgraspnet import TacGraspNet
from losses.tacgraspnet.mse import MSE
from scores.tacgraspnet.r2 import DisplacementR2PerSample, StressR2PerSample
from scores.tacgraspnet.mae import DisplacementMAE, StressMAE


def get_data_loaders(model_config: TacGraspNetConfig):
    # Initialize dataset config for train and validation data loader
    train_dataset_config = DGSDatasetConfig()
    validation_dataset_config = DGSDatasetConfig()

    if model_config.mode == "training": # If we are in training mode
        # Get frames for validation and training
        trajs = list(range(100))
        random.shuffle(trajs)
        validation_size = int(100 * model_config.validation_ratio)  # 100 is the number of trajectories (grasping poses) for each object
        validation_frames = trajs[:validation_size] # Trajectories for validation
        train_frames = trajs[validation_size:] # Trajectories for training

        if model_config.data_strategy == "single_obj": # If we train on one single object
            # Construct train data loader
            train_dataset_config.focused_objs = [model_config.objs[0]] # Only consider the first object in the list
            print(model_config.objs[0])
            train_dataset_config.focused_trajs = train_frames

            # Construct validation dataset
            validation_dataset_config.focused_objs = [model_config.objs[0]]
            validation_dataset_config.focused_trajs = validation_frames

        elif model_config.data_strategy == "multiple_objs_1": # Else if we train on multiple objects with first option
            # Construct train data loader
            train_dataset_config.focused_objs = model_config.objs
            train_dataset_config.focused_trajs = train_frames

            # Construct validation dataset
            validation_dataset_config.focused_objs = model_config.objs
            validation_dataset_config.focused_trajs = validation_frames

        else: # Otherwise, we train on multiple objects with second option
            # Construct train data loader
            train_dataset_config.focused_objs = model_config.objs
            train_dataset_config.focused_trajs = train_frames

            # Construct validation dataset
            validation_dataset_config.focused_objs = model_config.validation_objs # Here, we consider other objects for validation
            validation_dataset_config.focused_trajs = validation_frames

        ##TODO
        train_dataset_config.focused_trajs = [0]
        # train_dataset_config.focused_frames = list(range(25))
        validation_dataset_config.focused_trajs = [0]
        # validation_dataset_config.focused_frames = list(range(25, 50))
        ##
        # Construct train data loader
        train_dataset = DGSDataset(train_dataset_config)  # Construct train dataset
        train_loader = DataLoader(
            train_dataset,
            batch_size=model_config.batch_size,
            shuffle=True,
            collate_fn=DGSDataset.collate,
            num_workers=4,
            pin_memory=True,
        )

        # Construct validation data loader
        validation_dataset = DGSDataset(validation_dataset_config)  # Construct validation dataset
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=DGSDataset.collate,
            num_workers=4,
            pin_memory=True,
        )

        return train_loader, validation_loader
    else:
        return None

def train(model_config: TacGraspNetConfig):
    if model_config.mode == "training":
        print("#" * 15)
        print("Detected device:", model_config.device.upper())
        print("Training on:", "GPU" if torch.cuda.is_available() else "CPU")
        print("#" * 15)

        # Set random seed for PyTorch
        torch.random.manual_seed(42)

        # Construct data loaders
        train_loader, validation_loader = get_data_loaders(model_config)

        # Initialize logging
        logger = wandb.init(
            project="TacGraspNet",
            name=f"train_{model_config.data_strategy}_{datetime.now().strftime('%m%d_%H%M')}",
            config = model_config.__dict__,
        )

        # Initialize model
        model = TacGraspNet(model_config)

        # Initialize processors
        preprocessor, postprocessor = make_tacgraspnet_processors(model_config)

        # Initialize optimizer
        print("##############")
        print("Adam")
        print("##############")
        optimizer = Adam(params=model.parameters(), **model_config.optimizer_params)

        # --- DIAGNOSTIC BLOCK: VERIFY OPTIMIZER REGISTRATION ---
        print("\n" + "=" * 40)
        print("🔍 PARAMETER REGISTRATION CHECK")
        print("=" * 40)

        # 1. Check if the model registered the layers properly
        print("\n--- Trainable Layers in Model ---")
        registered_names = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                registered_names.append(name)
                # Print a few key names to verify ModuleDicts/Lists
                if "edge" in name or "graphnet" in name:
                    print(f"Registered: {name} | Shape: {param.shape}")

        # 2. Check if the optimizer caught all of them
        model_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        optimizer_param_count = sum(p.numel() for group in optimizer.param_groups for p in group['params'])

        print("\n--- Parameter Count Verification ---")
        print(f"Total Learnable Params in Model:     {model_param_count:,}")
        print(f"Total Params Tracked by Optimizer:   {optimizer_param_count:,}")

        if model_param_count == optimizer_param_count:
            print("✅ SUCCESS: Optimizer is tracking all model parameters!")
        else:
            print("❌ DANGER: Optimizer is missing parameters. Re-check ModuleDicts.")
        print("=" * 40 + "\n")
        # -------------------------------------------------------

        # Initialize loss and score functions
        loss_fn = MSE(model_config)
        score_classes = [DisplacementMAE, StressMAE]
        score_fns = {}
        for score_class in score_classes:
            score_fns[score_class] = score_class(model_config)

        # Accumulate
        model.set_is_training(True)
        for batch in tqdm(train_loader, mininterval=5.0, leave=False):
            batch = preprocessor(batch)
            model.accumulate(batch)
        node_output = model._node_output_normalizer._get_statistics()
        tet_output = model._tetra_output_normalizer._get_statistics()
        mesh = model._edge_normalizers["mesh_edges"]._get_statistics()
        contact = model._edge_normalizers["contact_edges"]._get_statistics()
        node = model._node_normalizer._get_statistics()

        # Training
        for epoch in range(model_config.n_epochs):
            ########################################
            ## Train model
            ########################################
            # Set model's mode to "train"
            model.train()
            model.set_is_training(False)
            # model.set_is_training(True) # Set flag to true to wake up normalizers TODO

            # Initialize train loss and score sums
            train_loss_sum = 0.0
            train_score_sums = {}
            for score_class in score_classes:
                train_score_sums[score_class] = 0.0

            # Number of batches to compute average loss and scores
            n_batches = 0.0

            # Train
            for batch in tqdm(train_loader, mininterval=5.0):
                # Optimizing model
                optimizer.zero_grad()
                batch = preprocessor(batch)
                batch = model(batch)
                extra_tet_output = model._tetra_output_normalizer._get_statistics()
                loss = loss_fn(batch)
                loss.backward()
                optimizer.step()

                # Update train loss and score sums
                with torch.no_grad():
                    train_loss_sum += loss.item()
                    for score_class in score_classes:
                        train_score_sums[score_class] += score_fns[score_class](batch).item()

                # Update number of batches variable
                n_batches += 1.0

            ########################################
            ## Validate model
            ########################################
            # Set model's mode to "train"
            model.eval()
            model.set_is_training(False) # Set flag to false to suspend normalizers

            # Initialize validation score sums
            validation_score_sums = {}
            for score_class in score_classes:
                validation_score_sums[score_class] = 0.0

            # Number of data points to compute average scores
            n_data_points = 0.0
            for data_point in tqdm(validation_loader, mininterval=10.0, leave=False):
                # Update validation score sums
                with torch.no_grad():
                    data_point = preprocessor(data_point)
                    data_point = model(data_point)
                    for score_class in score_classes:
                        validation_score_sums[score_class] += score_fns[score_class](data_point).item()

                # Update number of data points variable
                n_data_points += 1.0

            ########################################
            ## Do logging
            ########################################
            # Initialize logs
            logs = {"train/avg_loss": train_loss_sum / n_batches}
            for score_class in score_classes:
                logs["train/avg_scores/" + str(score_fns[score_class])] = train_score_sums[score_class] / n_batches
                logs["validation/avg_scores/" + str(score_fns[score_class])] = validation_score_sums[score_class] / n_data_points

            # Print epoch and average loss and scores as progress
            print("Epoch:", epoch + 1, "| Average loss:", logs["train/avg_loss"])

            # Do logging
            logger.log(logs, step=epoch + 1, commit=True)

        return None
    else:
        return None


if __name__ == "__main__":
    config = TacGraspNetConfig()
    train(config)
