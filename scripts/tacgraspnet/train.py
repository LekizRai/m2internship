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
    # Initialize dataset config
    dataset_config = DGSDatasetConfig()

    if model_config.mode == "training": # If we are in training mode
        trajs = list(range(100))
        random.shuffle(trajs)
        validation_size = int(100 * model_config.validation_ratio)  # 100 is the number of trajectories (grasping poses) for each object
        validation_frames = trajs[:validation_size] # Trajectories for validation
        train_frames = trajs[validation_size:] # Trajectories for training
        if model_config.data_strategy == "single_obj": # If we train on one single object
            # Construct train data loader
            dataset_config.focused_objs = [model_config.objs[0]] # Only consider the first object in the list
            dataset_config.focused_trajs = train_frames
            train_dataset = DGSDataset(dataset_config) # Construct train dataset
            train_loader = DataLoader(
                train_dataset,
                batch_size=model_config.batch_size,
                shuffle=True,
                collate_fn=DGSDataset.collate
            )

            # Construct validation dataset
            dataset_config.focused_objs = [model_config.objs[0]]
            dataset_config.focused_trajs = validation_frames
            validation_dataset = DGSDataset(dataset_config) # Construct validation dataset
            validation_loader = DataLoader(
                validation_dataset,
                batch_size=1,
                shuffle=False,
                collate_fn=DGSDataset.collate
            )

            return train_loader, validation_loader
        elif model_config.data_strategy == "multiple_objs_1":
            # Construct train data loader
            dataset_config.focused_objs = model_config.objs
            dataset_config.focused_trajs = train_frames
            train_dataset = DGSDataset(dataset_config)  # Construct train dataset
            train_loader = DataLoader(
                train_dataset,
                batch_size=model_config.batch_size,
                shuffle=True,
                collate_fn=DGSDataset.collate
            )

            # Construct validation dataset
            dataset_config.focused_objs = model_config.objs
            dataset_config.focused_trajs = validation_frames
            validation_dataset = DGSDataset(dataset_config)  # Construct validation dataset
            validation_loader = DataLoader(
                validation_dataset,
                batch_size=1,
                shuffle=False,
                collate_fn=DGSDataset.collate
            )

            return train_loader, validation_loader
        else:
            # Construct train data loader
            dataset_config.focused_objs = model_config.objs
            dataset_config.focused_trajs = train_frames
            train_dataset = DGSDataset(dataset_config)  # Construct train dataset
            train_loader = DataLoader(
                train_dataset,
                batch_size=model_config.batch_size,
                shuffle=True,
                collate_fn=DGSDataset.collate
            )

            # Construct validation dataset
            dataset_config.focused_objs = model_config.validation_objs # Here, we consider other objects for validation
            dataset_config.focused_trajs = validation_frames
            validation_dataset = DGSDataset(dataset_config)  # Construct validation dataset
            validation_loader = DataLoader(
                validation_dataset,
                batch_size=1,
                shuffle=False,
                collate_fn=DGSDataset.collate
            )

            return train_loader, validation_loader
    else:
        return None

def train(model_config: TacGraspNetConfig):
    if model_config.mode == "training":
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

        for epoch in range(model_config.n_epochs):
            # Initialize model
            model.train()

            # Initialize lists of losses and scores
            train_losses = []
            train_scores = {}
            for score_class in score_classes:
                train_scores[score_class] = []

            # Training model
            for batch in tqdm(train_loader, mininterval=5.0, leave=False):
                # Optimizing model
                optimizer.zero_grad()
                batch = preprocessor(batch)
                batch = model(batch)
                loss = loss_fn(batch)
                loss.backward()
                optimizer.step()

                # Append losses and scores to lists
                train_losses.append(loss.cpu().detach())
                for score_class in score_classes:
                    train_scores[score_class].append(score_fns[score_class](batch).cpu().detach())

            # Initialize logs
            logs = {"train/avg_loss": torch.tensor(train_losses).mean()}
            for score_class in score_classes:
                logs["train/avg_scores/" + str(score_fns[score_class])] = torch.tensor(train_scores[score_class]).mean()

            # Print epoch and average loss and scores as progress
            print("Epoch:", epoch + 1, "| Average loss:", torch.tensor(train_losses).mean())

            # Do logging
            logging.log(logs, step=epoch + 1, commit=True)

        return None
    else:
        return None
