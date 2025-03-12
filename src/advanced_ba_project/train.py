import os
import datetime
from pathlib import Path
import hydra
import wandb
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig

from advanced_ba_project.data import get_dataloaders
from advanced_ba_project.logger import log
from advanced_ba_project.model import UNet


# Define the training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device="cuda"):
    model.to(device)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        log.info(f"Epoch {epoch + 1}/{num_epochs}")

        # Training phase
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        log.info(f"Train Loss: {epoch_loss:.4f}")

        # Log training loss to W&B
        wandb.log({"Train Loss": epoch_loss, "Epoch": epoch + 1})

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        log.info(f"Validation Loss: {val_loss:.4f}")

        # Log validation loss to W&B
        wandb.log({"Validation Loss": val_loss, "Epoch": epoch + 1})

    return train_losses, val_losses


@hydra.main(config_path=f"{os.getcwd()}/configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    """Train U-Net using configuration from Hydra."""

    log.info(f"Using Hydra Config: {cfg}")

    # Timestamp for unique model/log naming
    timestamp = cfg.timestamp

    # Initialize W&B (Always On)
    wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        name=f"{cfg.experiment_name}_{timestamp}",
        config=cfg.hyperparameters,
        mode=cfg.wandb.mode,  # Online or offline
    )

    # Load data
    train_loader, val_loader = get_dataloaders(
        data_path=Path(cfg.dataset.data_path),
        metadata_file=cfg.dataset.metadata_file,
        batch_size=cfg.hyperparameters.batch_size,
        subset=cfg.dataset.subset
    )

    # Initialize model
    model = UNet(in_channels=3, out_channels=1)

    # Define loss function & optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.hyperparameters.learning_rate, weight_decay=cfg.hyperparameters.weight_decay)

    # Train model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs=cfg.hyperparameters.num_epochs, device=device
    )

    # Save trained model
    model_path = f"models/unet_model_{timestamp}.pth"
    torch.save(model.state_dict(), model_path)
    log.success(f"Model saved as {model_path}")
    wandb.save(model_path)

    # Save the loss plot
    loss_plot_path = f"reports/figures/loss_plot_{timestamp}.png"
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training & Validation Loss")
    plt.savefig(loss_plot_path)
    log.success(f"Loss plot saved as {loss_plot_path}")
    wandb.log({"Loss Plot": wandb.Image(loss_plot_path)})

    # Finish W&B logging
    wandb.finish()


if __name__ == "__main__":
    main()
