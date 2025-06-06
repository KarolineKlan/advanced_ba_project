import datetime
import os
import random
from pathlib import Path

import hydra
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from torchvision import transforms
from tqdm import tqdm

from advanced_ba_project.data import ForestDataset, RoboflowTreeDataset, get_dataloaders


def green_tree_detector(image_batch, threshold=0.1):
    """
    Simple baseline that detects trees based on green channel values.

    Args:
        image_batch: Tensor of shape [batch_size, 3, 256, 256]
        threshold: How much greener a pixel must be compared to other channels

    Returns:
        Tensor of shape [batch_size, 1, 256, 256] with binary tree mask
    """
    batch_size = image_batch.shape[0]
    device = image_batch.device

    # Extract RGB channels
    r = image_batch[:, 0]  # Red channel
    g = image_batch[:, 1]  # Green channel
    b = image_batch[:, 2]  # Blue channel

    # Consider a pixel a tree if green value is dominant
    # g > r + threshold AND g > b + threshold
    tree_mask = ((g > (r + threshold)) & (g > (b + threshold))).float()

    # Reshape to [batch_size, 1, 256, 256] to match ground truth format
    return tree_mask.unsqueeze(1)

def evaluate_detector(dataloader, threshold=0.1):
    """Evaluate the green tree detector on the given dataloader."""
    device = next(iter(dataloader))[0].device

    # Initialize metrics
    total_pixels = 0
    metrics = {
        'accuracy': 0,
        'precision': 0,
        'recall': 0,
        'f1': 0,
        'iou': 0
    }

    # Process each batch
    for images, masks in tqdm(dataloader, desc="Evaluating baseline detector"):
        # Generate predictions
        predictions = green_tree_detector(images, threshold=threshold)

        # Convert to binary predictions (0 or 1)
        pred_binary = (predictions > 0.5).float()

        # Flatten tensors for metric calculation
        pred_flat = pred_binary.cpu().numpy().flatten().astype(int)
        mask_flat = masks.cpu().numpy().flatten().astype(int)

        # Update metrics
        batch_pixels = pred_flat.shape[0]
        total_pixels += batch_pixels

        metrics['accuracy'] += accuracy_score(mask_flat, pred_flat) * batch_pixels
        metrics['precision'] += precision_score(mask_flat, pred_flat, zero_division=0) * batch_pixels
        metrics['recall'] += recall_score(mask_flat, pred_flat, zero_division=0) * batch_pixels
        metrics['f1'] += f1_score(mask_flat, pred_flat, zero_division=0) * batch_pixels
        metrics['iou'] += jaccard_score(mask_flat, pred_flat, zero_division=0) * batch_pixels

    # Calculate final metrics
    for key in metrics:
        metrics[key] /= total_pixels

    return metrics



def visualize_baseline_predictions(val_loader, device, green_tree_detector, threshold=0.01, num_samples=5):
    """Visualizes predictions from the baseline green detector alongside ground truth."""

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs("reports/figures", exist_ok=True)
    save_path = f"reports/figures/baseline_mask_comparison_{timestamp}.png"

    images, true_masks = next(iter(val_loader))
    images, true_masks = images.to(device), true_masks.to(device)

    # Run baseline green detector
    with torch.no_grad():
        predicted_masks = green_tree_detector(images, threshold=threshold)
        predicted_masks = (predicted_masks > 0.5).float()

    # Convert tensors to NumPy for visualization
    images = images.cpu().numpy().transpose(0, 2, 3, 1)
    true_masks = true_masks.cpu().numpy().squeeze(1)
    predicted_masks = predicted_masks.cpu().numpy().squeeze(1)

    # Plot images, ground truth masks, and predicted masks
    fig, axes = plt.subplots(num_samples, 3, figsize=(10, num_samples * 3))

    for i in range(num_samples):
        axes[i, 0].imshow((images[i] * 0.5) + 0.5)  # Undo normalization
        axes[i, 0].set_title("Original Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(true_masks[i], cmap="gray")
        axes[i, 1].set_title("Ground Truth Mask")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(predicted_masks[i], cmap="gray")
        axes[i, 2].set_title(f"Baseline Mask (threshold={threshold:.2f})")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[INFO] Baseline mask comparison saved as {save_path}")
    plt.show()




@hydra.main(config_path=f"{os.getcwd()}/configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    data_path = Path("data/raw/Forest Segmented")
    metadata_file = "meta_data.csv"
    roboflow_train_path = Path("data/raw/roboflow/train")
    roboflow_val_path = Path("data/raw/roboflow/valid")
    roboflow_test_path = Path("data/raw/roboflow/test")

    train_loader, val_loader, test_loader = get_dataloaders(
        data_path,
        metadata_file,
        roboflow_train_path,
        roboflow_val_path,
        roboflow_test_path,
        batch_size=32,
        img_dim=256,
        subset=False,  # True if you want to reduce sizex
        apply_augmentation=True,
    )

    # Try different thresholds
    thresholds = [0]#[0, 0.01, 0.025, 0.05]
    best_threshold = None
    best_f1 = -1

    print("Evaluating green tree detector baseline with different thresholds...")
    for threshold in thresholds:
        print(f"\nTesting threshold: {threshold}")
        metrics = evaluate_detector(val_loader, threshold=threshold)

        print(f"Results:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print(f"  IoU:       {metrics['iou']:.4f}")

        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_threshold = threshold

    print(f"\nBest threshold: {best_threshold} with F1 score: {best_f1:.4f}")

    return None

def visualize_baseline_pred_col(val_loader, device, green_tree_detector, threshold=0.01, num_samples=10, seed=42):
    """
    Visualizes baseline green detector masks overlaid in green/red with legend and fixed 5x2 layout.

    Args:
        val_loader: DataLoader for validation images and masks
        device: Device for computation
        green_tree_detector: function for green detection (returns mask)
        threshold: float, green detection threshold
        num_samples: number of samples to show (will be capped at 10 for 5x2 grid)
        seed: random seed for sample selection
    """
    # Ensure exactly 10 samples for 5x2 layout
    num_samples = min(num_samples, 10)

    # Seed for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Setup save path
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs("reports/figures", exist_ok=True)
    save_path = f"reports/figures/baseline_colored_overlay_{timestamp}.png"

    # Load a batch
    images, true_masks = next(iter(val_loader))
    images, true_masks = images.to(device), true_masks.to(device)

    # Randomly choose samples
    indices = random.sample(range(images.shape[0]), min(num_samples, images.shape[0]))

    with torch.no_grad():
        predictions = green_tree_detector(images, threshold=threshold)
        predictions = (predictions > 0.5).float()

    # Convert for visualization
    images_np = images.cpu().numpy().transpose(0, 2, 3, 1)
    predictions_np = predictions.cpu().numpy().squeeze(1)

    # Set up 2 rows × 5 columns layout
    fig, axes = plt.subplots(2, 5, figsize=(18, 6))

    for plot_idx, idx in enumerate(indices):
        img = (images_np[idx] * 0.5) + 0.5  # Undo normalization
        mask = predictions_np[idx]

        overlay = img.copy()
        red = np.array([1, 0.3, 0])
        green = np.array([0, 1, 0])

        overlay[mask == 1] = green  # Forest
        overlay[mask == 0] = red    # Non-forest

        col = plot_idx
        axes[0, col].imshow(img)
        axes[0, col].set_title(f"Original {idx}")
        axes[0, col].axis("off")

        axes[1, col].imshow(overlay)
        axes[1, col].set_title(f"With Mask {idx}")
        axes[1, col].axis("off")

    # Legend
    forest_patch = mpatches.Patch(color='#228B22', label='Forest')
    non_forest_patch = mpatches.Patch(color='#CD5C5C', label='Non-Forest')
    fig.legend(handles=[forest_patch, non_forest_patch], loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.savefig(save_path)
    print(f"[INFO] Baseline green overlay (5x2 layout) saved to {save_path}")
    plt.show()



def visualize_raw_and_baseline(
    green_tree_detector,
    dataset_name: str,
    data_path: Path,
    metadata_file: str = None,
    image_dir: Path = None,
    label_dir: Path = None,
    num_images: int = 5,
    img_dim: int = 256,
    seed: int = 42,
    indices: list = None,
    device="cpu",
    threshold: float = 0.01
):
        # Define transforms
        transform = transforms.Compose([
            transforms.Resize((img_dim, img_dim)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
        target_transform = transforms.Compose([
            transforms.Resize((img_dim, img_dim)),
            transforms.ToTensor()
        ])

        # Load dataset
        if dataset_name == "forest":
            dataset = ForestDataset(
                data_path=data_path,
                metadata_file=metadata_file,
                transform=transform,
                target_transform=target_transform
            )
        elif dataset_name == "roboflow":
            dataset = RoboflowTreeDataset(
                image_dir=image_dir,
                label_dir=label_dir,
                patch_size=img_dim,
                transform=transform,
                target_transform=target_transform
            )
        else:
            raise ValueError("dataset_name must be either 'forest' or 'roboflow'")

        # Pick indices
        if indices is None:
            random.seed(seed)
            indices = random.sample(range(len(dataset)), min(num_images, len(dataset)))
        else:
            num_images = len(indices)

        # Extract images/masks
        samples = [dataset[i] for i in indices]
        images = torch.stack([img for img, _ in samples]).to(device)
        masks = torch.stack([mask for _, mask in samples]).to(device)

        # Predict using baseline
        with torch.no_grad():
            preds = green_tree_detector(images, threshold=threshold)
            preds = (preds > 0.5).float()

        # Plot
        fig, axs = plt.subplots(3, num_images, figsize=(4 * num_images, 12))

        for i in range(num_images):
            img = images[i].cpu().permute(1, 2, 0).numpy()
            img = (img * 0.5) + 0.5
            img = np.clip(img, 0, 1)

            gt_mask = masks[i].cpu().squeeze().numpy()
            pred_mask = preds[i].cpu().squeeze().numpy()

            forest_color = np.array([0, 1, 0])      # Green
            nonforest_color = np.array([1, 0.3, 0]) # Reddish
            alpha = 0.5

            # Ground Truth overlay
            overlay_gt = img.copy()
            mask_bool = gt_mask > 0.5
            overlay_gt[mask_bool] = (1 - alpha) * overlay_gt[mask_bool] + alpha * forest_color
            overlay_gt[~mask_bool] = (1 - alpha) * overlay_gt[~mask_bool] + alpha * nonforest_color

            # Baseline Prediction overlay
            overlay_pred = img.copy()
            mask_pred_bool = pred_mask > 0.5
            overlay_pred[mask_pred_bool] = (1 - alpha) * overlay_pred[mask_pred_bool] + alpha * forest_color
            overlay_pred[~mask_pred_bool] = (1 - alpha) * overlay_pred[~mask_pred_bool] + alpha * nonforest_color

            axs[0, i].imshow(img)
            axs[0, i].set_title(f"Original {indices[i]}")
            axs[1, i].imshow(overlay_gt)
            axs[1, i].set_title(f"Ground Truth {indices[i]}")
            axs[2, i].imshow(overlay_pred)
            axs[2, i].set_title(f"Baseline Mask {indices[i]}")

            for ax in axs[:, i]:
                ax.axis("off")

        plt.tight_layout()
        plt.show()



def get_top_forest_ids(
    dataset_name: str,
    data_path: Path,
    metadata_file: str = None,
    image_dir: Path = None,
    label_dir: Path = None,
    img_dim: int = 256,
    top_k: int = 5
):
    transform = transforms.Compose([
        transforms.Resize((img_dim, img_dim)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    target_transform = transforms.Compose([
        transforms.Resize((img_dim, img_dim)),
        transforms.ToTensor()
    ])

    if dataset_name == "forest":
        dataset = ForestDataset(
            data_path=data_path,
            metadata_file=metadata_file,
            transform=transform,
            target_transform=target_transform
        )
    elif dataset_name == "roboflow":
        dataset = RoboflowTreeDataset(
            image_dir=image_dir,
            label_dir=label_dir,
            patch_size=img_dim,
            transform=transform,
            target_transform=target_transform
        )
    else:
        raise ValueError("dataset_name must be either 'forest' or 'roboflow'")

    forest_coverage = []
    for idx in range(len(dataset)):
        _, mask = dataset[idx]
        forest_pixels = (mask > 0.5).sum().item()
        forest_coverage.append((idx, forest_pixels))

    top_ids = sorted(forest_coverage, key=lambda x: x[1], reverse=True)[:top_k]
    return [idx for idx, _ in top_ids]

if __name__ == "__main__":


    top_ids = get_top_forest_ids(
    dataset_name="roboflow",
    data_path=Path("data/raw/forest"),
    image_dir=Path("data/raw/roboflow/test/images"),
    label_dir=Path("data/raw/roboflow/test/labelTxt"),
    img_dim=256,
    top_k=15
)
    print("Top forest IDs:", top_ids)

    #main()
    # data_path = Path("data/raw/Forest Segmented")
    # metadata_file = "meta_data.csv"
    # roboflow_train_path = Path("data/raw/roboflow/train")
    # roboflow_val_path = Path("data/raw/roboflow/valid")
    # roboflow_test_path = Path("data/raw/roboflow/test")
    # metadata_file = "meta_data.csv"

    visualize_raw_and_baseline(
        green_tree_detector=green_tree_detector,
        data_path=Path("data/raw/forest"),
        dataset_name="roboflow",
        image_dir=Path("data/raw/roboflow/test/images"),
        label_dir=Path("data/raw/roboflow/test/labelTxt"),
        num_images=5,
        img_dim=256,
        device="cpu",
        indices=[11,349,359,383,473],
        threshold=0
    )

