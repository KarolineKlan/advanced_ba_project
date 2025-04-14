import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import wandb

from advanced_ba_project.data import get_dataloaders
from advanced_ba_project.model import UNet
from advanced_ba_project.logger import log
from advanced_ba_project.train import train_model, DiceBCELoss  # Include scheduler + loss!

def sweep_train():
    wandb.init()
    config = wandb.config

    # Load data
    train_loader, val_loader = get_dataloaders(
        data_path=Path("data/raw/Forest Segmented"),
        metadata_file="meta_data.csv",
        roboflow_train_path=Path("data/raw/roboflow/train"),
        roboflow_val_path=Path("data/raw/roboflow/valid"),
        batch_size=config.batch_size,
        subset=False,
    )

    # Build model
    model = UNet(
        in_channels=3,
        out_channels=1,
        init_features=config.init_features,
        dropout_rate=config.dropout_rate,
    )

    criterion = DiceBCELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=config.num_epochs,
        device=device,
    )

    # ✅ Save model using run name for traceability
    run_name = wandb.run.name.replace(" ", "_")
    model_path = f"models/unet_model_{run_name}.pth"
    torch.save(model.state_dict(), model_path)
    wandb.save(model_path)
    print(f"✅ Model saved to {model_path}")

    wandb.finish()


if __name__ == "__main__":
    sweep_train()
