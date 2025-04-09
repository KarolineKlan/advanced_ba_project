import os
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score

import wandb
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

        # Metric accumulators
        total_pixels = 0
        metrics = {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "iou": 0}

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                # Post-process predictions
                preds = torch.sigmoid(outputs)
                preds_bin = (preds > 0.5).float()

                # Flatten predictions and masks
                preds_flat = preds_bin.cpu().numpy().flatten().astype(int)
                masks_flat = masks.cpu().numpy().flatten().astype(int)

                batch_pixels = preds_flat.shape[0]
                total_pixels += batch_pixels

                # Update metric accumulators
                metrics["accuracy"] += accuracy_score(masks_flat, preds_flat) * batch_pixels
                metrics["precision"] += precision_score(masks_flat, preds_flat, zero_division=0) * batch_pixels
                metrics["recall"] += recall_score(masks_flat, preds_flat, zero_division=0) * batch_pixels
                metrics["f1"] += f1_score(masks_flat, preds_flat, zero_division=0) * batch_pixels
                metrics["iou"] += jaccard_score(masks_flat, preds_flat, zero_division=0) * batch_pixels

        # Normalize results
        val_loss /= len(val_loader)
        for k in metrics:
            metrics[k] /= total_pixels

        val_losses.append(val_loss)

        # Log to terminal
        log.info(f"Validation Loss: {val_loss:.4f}")
        log.info(
            f"Accuracy: {metrics['accuracy']:.4f} | "
            f"Precision: {metrics['precision']:.4f} | "
            f"Recall: {metrics['recall']:.4f} | "
            f"F1: {metrics['f1']:.4f} | "
            f"IoU: {metrics['iou']:.4f}"
        )

        # Log to Weights & Biases
        wandb.log(
            {
                "Train Loss": epoch_loss,
                "Validation Loss": val_loss,
                "Val Accuracy": metrics["accuracy"],
                "Val Precision": metrics["precision"],
                "Val Recall": metrics["recall"],
                "Val F1": metrics["f1"],
                "Val IoU": metrics["iou"],
            },
            step=int(epoch + 1)
        )

    return train_losses, val_losses


@hydra.main(config_path=f"{os.getcwd()}/configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    """Train U-Net using configuration from Hydra."""

    log.info(f"Using Hydra Config: {cfg}")

    # Timestamp for unique model/log naming
    timestamp = cfg.timestamp

    # Convert Hydra config to a JSON-friendly dictionary
    wandb_config = OmegaConf.to_container(cfg.hyperparameters, resolve=True)

    # Initialize W&B
    wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        name=f"{cfg.experiment_name}_{timestamp}",
        config=wandb_config,
        mode=cfg.wandb.mode,  # Online or offline
    )

    # Load data
    train_loader, val_loader = get_dataloaders(
        data_path=Path(to_absolute_path(cfg.dataset.data_path)),
        metadata_file=cfg.dataset.metadata_file,
        roboflow_train_path=Path(to_absolute_path(cfg.dataset.roboflow_train_path)),
        roboflow_val_path=Path(to_absolute_path(cfg.dataset.roboflow_val_path)),
        batch_size=cfg.hyperparameters.batch_size,
        subset=cfg.dataset.subset,
    )

    # Initialize model
    model = UNet(in_channels=3, out_channels=1)

    # Define loss function & optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=cfg.hyperparameters.learning_rate, weight_decay=cfg.hyperparameters.weight_decay
    )

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
