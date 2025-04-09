import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import wandb

from advanced_ba_project.data import get_dataloaders
from advanced_ba_project.model import UNet
from advanced_ba_project.logger import log
from advanced_ba_project.train import train_model


def sweep_train():
    # Initialize W&B sweep run
    wandb.init()
    config = wandb.config

    # Load data
    train_loader, val_loader = get_dataloaders(
        data_path=Path("data/raw/Forest Segmented"),
        metadata_file="meta_data.csv",
        roboflow_train_path=Path("data/raw/roboflow/train"),
        roboflow_val_path=Path("data/raw/roboflow/valid"),
        batch_size=config.hyperparameters.batch_size,
        subset=False,
    )

    # Initialize model
    model = UNet(
        in_channels=3,
        out_channels=1,
        init_features=config.model.init_features,
    )

    # Define loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.hyperparameters.learning_rate,
        weight_decay=config.hyperparameters.weight_decay,
    )

    # Train
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=config.hyperparameters.num_epochs,
        device=device,
    )

    wandb.finish()


if __name__ == "__main__":
    sweep_train()
