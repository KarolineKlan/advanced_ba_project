import torch
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import datetime
from advanced_ba_project.model import UNet
from advanced_ba_project.data import get_dataloaders
from advanced_ba_project.logger import log

def load_model(model_path, device):
    """Loads the trained U-Net model from models/ directory using the given model name."""
    model_path = Path(model_path)
    if not model_path.exists():
        log.error(f"Model file not found: {model_path}")
        exit(1)
    
    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    log.success(f"Loaded model from {model_path}")
    return model

def visualize_predictions(model, val_loader, device, num_samples=5):
    """Visualizes and compares predicted masks with ground truth."""
    
    # Generate a timestamp for unique filenames
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = f"reports/figures/mask_comparison_{timestamp}.png"

    images, true_masks = next(iter(val_loader))  # Get a batch of validation images
    images, true_masks = images.to(device), true_masks.to(device)

    with torch.no_grad():
        predicted_masks = model(images)
        predicted_masks = torch.sigmoid(predicted_masks)  # Convert logits to probability
        predicted_masks = (predicted_masks > 0.5).float()  # Apply threshold

    # Convert tensors to NumPy for visualization
    images = images.cpu().numpy().transpose(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]
    true_masks = true_masks.cpu().numpy().squeeze(1)  # Remove single-channel dim
    predicted_masks = predicted_masks.cpu().numpy().squeeze(1)  # Remove single-channel dim

    # Plot images, ground truth masks, and predicted masks
    fig, axes = plt.subplots(num_samples, 3, figsize=(10, num_samples * 3))

    for i in range(num_samples):
        axes[i, 0].imshow((images[i] * 0.5) + 0.5)  # Undo normalization for visualization
        axes[i, 0].set_title("Original Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(true_masks[i], cmap="gray")
        axes[i, 1].set_title("Ground Truth Mask")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(predicted_masks[i], cmap="gray")
        axes[i, 2].set_title("Predicted Mask")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    log.success(f"Mask comparison saved as {save_path}")
    plt.show()

if __name__ == "__main__":
    # CLI Argument Parser
    parser = argparse.ArgumentParser(description="Visualize U-Net model predictions")
    parser.add_argument("--model-path", type=str, required=True, help="Path of the trained model (e.g. models/unet_model_2024-03-11_15-30-45.pth)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for visualization")
    parser.add_argument("--subset", type=str, default="true", help="Use small subset for quick testing (true/false)")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples to visualize")
    args = parser.parse_args()

    # Convert subset argument from string to boolean
    subset = args.subset.lower() in ["true", "1", "yes"]

    # Device selection
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log.info("Loading validation data...")
    data_path = Path("data/raw/Forest Segmented")
    metadata_file = "meta_data.csv"
    _, val_loader = get_dataloaders(data_path, metadata_file, batch_size=args.batch_size, subset=subset)

    # Load trained model
    model = load_model(args.model_path, device)

    # Visualize predictions
    visualize_predictions(model, val_loader, device, num_samples=args.num_samples)
