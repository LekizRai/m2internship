import random
import wandb
import torch

from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import random_split, DataLoader

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
        # train_dataset_config.focused_trajs = [0]
        ##
        # Construct train data loader
        train_dataset = DGSDataset(train_dataset_config)  # Construct train dataset
        train_loader = DataLoader(
            train_dataset,
            batch_size=model_config.batch_size,
            shuffle=True,
            collate_fn=DGSDataset.collate,
            # num_workers=2,
            pin_memory=True,
        )

        # Construct validation data loader
        validation_dataset = DGSDataset(validation_dataset_config)  # Construct validation dataset
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=DGSDataset.collate,
            # num_workers=2,
            pin_memory=True,
        )

        return train_loader, validation_loader
    else:
        return None

def train(model_config: TacGraspNetConfig):
    if model_config.mode == "training":
        print(model_config.device)
        print(torch.cuda.is_available())

        # Construct data loaders
        train_loader, validation_loader = get_data_loaders(model_config)

        # Initialize logging
        logging = wandb.init(
            project="TacGraspNet", config=model_config.args.__dict__,
        )

        # Initialize model
        model = TacGraspNet(model_config)

        # Initialize processors
        preprocessor, postprocessor = make_tacgraspnet_processors(model_config)

        # Initialize optimizer
        optimizer = AdamW(model.parameters(), **model_config.optimizer_params)

        # Initialize loss and score functions
        loss_fn = MSE(model_config)
        score_classes = [DisplacementMAE, StressMAE, DisplacementR2PerSample, StressR2PerSample]
        score_fns = {}
        for score_class in score_classes:
            score_fns[score_class] = score_class(model_config)

        torch.autograd.set_detect_anomaly(True, check_nan=False) #TODO
        for epoch in range(model_config.n_epochs):
            # Initialize model
            model.train()

            # Initialize train loss and score sums
            train_loss_sum = 0.0
            train_score_sums = {}
            for score_class in score_classes:
                train_score_sums[score_class] = 0.0

            # Number of batches to compute average loss and scores
            n_batches = 0.0

            # Training model
            for batch in tqdm(train_loader, mininterval=5.0, leave=False):
                # Optimizing model
                optimizer.zero_grad()
                batch = preprocessor(batch)
                batch = model(batch)
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

            # Initialize logs
            logs = {"train/avg_loss": train_loss_sum / n_batches}
            for score_class in score_classes:
                logs["train/avg_scores/" + str(score_fns[score_class])] = train_score_sums[score_class] / n_batches

            # Print epoch and average loss and scores as progress
            print("Epoch:", epoch + 1, "| Average loss:", train_loss_sum / n_batches)

            # Do logging
            logging.log(logs, step=epoch + 1, commit=True)

        return None
    else:
        return None
