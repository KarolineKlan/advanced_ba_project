import argparse
import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from advanced_ba_project.data import get_dataloaders
from advanced_ba_project.logger import log  # Import your logger
from advanced_ba_project.model import UNet


# Define the training function
def train_model(
    model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device="cuda"
):
    model.to(device)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        log.info(f"Epoch {epoch+1}/{num_epochs}")  # Use loguru instead of print

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
        log.info(f"Train Loss: {epoch_loss:.4f}")  # Logging instead of print

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
        log.info(f"Validation Loss: {val_loss:.4f}\n")

    return train_losses, val_losses


if __name__ == "__main__":
    # Generate a timestamp for unique filenames
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # CLI Argument Parser
    parser = argparse.ArgumentParser(description="Train U-Net model")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--num-epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--subset", type=str, default="false", help="Use small subset for quick testing (true/false)")
    args = parser.parse_args()

    # Convert subset argument from string to boolean
    subset = args.subset.lower() in ["true", "1", "yes"]

    log.info(f"Starting training with batch size {args.batch_size} for {args.num_epochs} epochs")

    # Load data
    data_path = Path("data/raw/Forest Segmented")
    metadata_file = "meta_data.csv"
    train_loader, val_loader = get_dataloaders(data_path, metadata_file, batch_size=args.batch_size, subset=subset)

    # Initialize model
    model = UNet(in_channels=3, out_channels=1)

    # Define loss function & optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=args.num_epochs, device=device)

    # Ensure model and plot directories exist
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)

    # Save the trained model with timestamp
    model_path = f"models/unet_model_{timestamp}.pth"
    torch.save(model.state_dict(), model_path)
    log.success(f"Model saved as {model_path}")

    # Save the loss plot with timestamp
    loss_plot_path = f"reports/figures/loss_plot_{timestamp}.png"
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training & Validation Loss")
    plt.savefig(loss_plot_path)
    log.success(f"Loss plot saved as {loss_plot_path}")
