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

def sample_train():
    # Initialize dataset
    dataset_config = DGSDatasetConfig()
    dataset_config.focused_objs = ["potato2"]
    dataset_config.focused_trajs = [30]
    dataset = DGSDataset(dataset_config)

    # Train test split
    model_config = TacGraspNetConfig()
    train_size = int(model_config.training_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(
        train_dataset,
        batch_size=model_config.batch_size,
        shuffle=True,
        collate_fn=DGSDataset.collate
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=DGSDataset.collate
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

        # Print mean values of losses and scores as progress
        print("Epoch:", epoch + 1, "| Loss:", torch.tensor(train_losses).mean())
        for score_class in score_classes:
            print(str(score_fns[score_class]) + ":", torch.tensor(train_scores[score_class]).mean())


if __name__ == "__main__":
    sample_train()