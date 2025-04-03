from data import ForestDataset, get_dataloaders
import hydra
import os
from pathlib import Path
import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score

from omegaconf import DictConfig
from PIL import Image

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


def visualize_prediction(image, threshold=0.01, ground_truth=None):
    """
    Takes a single image and visualizes the original image alongside the predicted mask.
    
    Args:
        image: Tensor of shape [3, 256, 256] - single RGB image
        threshold: Threshold for green detection
        ground_truth: Optional ground truth mask of shape [1, 256, 256]
    
    Returns:
        Displays the visualization and returns the predicted mask
    """
    import matplotlib.pyplot as plt
    
    

    image = Image.open(image).convert("RGB")
    image = np.array(image)
    image = image.transpose(2, 0, 1)
    
    # Ensure image is a tensor with batch dimension and on CPU
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)
    if len(image.shape) == 3:
        image = image.unsqueeze(0)  # Add batch dimension
        
    # Generate prediction
    with torch.no_grad():
        prediction = green_tree_detector(image, threshold=threshold)
    
    # Convert to numpy for visualization
    image_np = image[0].permute(1,2,0).cpu().numpy()  # [H, W, 3]
    mask_np = prediction[0, 0].cpu().numpy()            # [H, W]
    
    # Create figure
    n_plots = 3 if ground_truth is not None else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(4*n_plots, 4))
    
    # Plot original image
    axes[0].imshow(image_np)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    # Plot predicted mask
    axes[1].imshow(mask_np, cmap='viridis')
    axes[1].set_title(f"Predicted Mask (threshold={threshold:.2f})")
    axes[1].axis("off")
    
    # Plot ground truth if provided
    if ground_truth is not None:
        if isinstance(ground_truth, torch.Tensor):
            if len(ground_truth.shape) == 4:
                gt_np = ground_truth[0, 0].cpu().numpy()
            else:
                gt_np = ground_truth[0].cpu().numpy()
        else:
            gt_np = ground_truth
            
        axes[2].imshow(gt_np, cmap='viridis')
        axes[2].set_title("Ground Truth")
        axes[2].axis("off")
    
    plt.tight_layout()
    plt.show()
    
    return prediction



@hydra.main(config_path=f"{os.getcwd()}/configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    # Load data
    train_loader, val_loader = get_dataloaders(
        data_path=Path(cfg.dataset.data_path),
        metadata_file=cfg.dataset.metadata_file,
        batch_size=cfg.hyperparameters.batch_size,
        subset=cfg.dataset.subset,
    )
    
    # Try different thresholds
    thresholds = [0, 0.01, 0.025, 0.05]
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


if __name__ == "__main__":
    #main()
    
    visualize_prediction(image='/Users/kristofferkjaer/Desktop/DTU_masters/F25/ABA/advanced_ba_project/data/raw/Forest Segmented/images/3484_sat_24.jpg', threshold=0)