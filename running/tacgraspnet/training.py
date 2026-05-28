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
    dataset_config.focused_objs = ["sphere01"]
    dataset_config.focused_trajs = [25]
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

    for epoch in range(model_config.n_epochs):
        model.train()

        for batch in tqdm(train_loader, mininterval=5.0):
            optimizer.zero_grad()
            batch = preprocessor(batch)
            batch = model(batch)
            loss = loss_fn(batch)
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    sample_train()