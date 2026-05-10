import torch
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, accuracy_score, confusion_matrix
import torchvision.transforms as transforms

from model import create_model
from dataset import HeadDetectionDataset
from utils import load_checkpoint, plot_predictions, calculate_metrics

def binarize_counts(counts, bins):
    """Convert continuous head counts into discrete bins for classification metrics."""
    return np.digitize(counts, bins, right=True)

def evaluate(model, test_loader, device, bins=None, lower_bins=None, higher_bins=None, lower_range=20, higher_range_start=40):
    if bins is None:
        # Bins for full range
        bins = [0, 10, 20, 40, 60, 80]
    if lower_bins is None:
        # Finer bins for lower range (0-20)
        lower_bins = [0, 5, 10, 15, 20]
    if higher_bins is None:
        # Bins for higher range (40-80)
        higher_bins = [40, 50, 60, 70, 80]
    
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            # Squeeze to remove extra dimension [batch_size, 1] -> [batch_size]
            outputs = outputs.squeeze().cpu().numpy()
            labels = labels.squeeze().cpu().numpy()
            predictions.extend(outputs)
            targets.extend(labels)
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Ensure non-negative predictions for head counts
    predictions = np.clip(predictions, 0, None)
    
    # Binarize predictions and targets for full range
    binary_predictions = binarize_counts(predictions, bins)
    binary_targets = binarize_counts(targets, bins)
    
    # Filter for lower range (0-20)
    lower_mask = (targets <= lower_range) & (predictions <= lower_range)
    lower_predictions = predictions[lower_mask]
    lower_targets = targets[lower_mask]
    lower_binary_predictions = binarize_counts(lower_predictions, lower_bins)
    lower_binary_targets = binarize_counts(lower_targets, lower_bins)
    
    # Filter for higher range (40-80)
    higher_mask = (targets >= higher_range_start) & (predictions >= higher_range_start)
    higher_predictions = predictions[higher_mask]
    higher_targets = targets[higher_mask]
    higher_binary_predictions = binarize_counts(higher_predictions, higher_bins)
    higher_binary_targets = binarize_counts(higher_targets, higher_bins)
    
    return (predictions, targets, binary_predictions, binary_targets, bins,
            lower_predictions, lower_targets, lower_binary_predictions, lower_binary_targets, lower_bins,
            higher_predictions, higher_targets, higher_binary_predictions, higher_binary_targets, higher_bins)

def plot_bin_histogram(targets, predictions, bins, save_path='results/bin_histogram.png', title='Histogram of True vs Predicted Head Counts'):
    """Plot histogram of binned true and predicted head counts."""
    plt.figure(figsize=(10, 6))
    plt.hist(targets, bins=bins, alpha=0.5, label='True Counts', color='blue', edgecolor='black')
    plt.hist(predictions, bins=bins, alpha=0.5, label='Predicted Counts', color='orange', edgecolor='black')
    plt.xlabel('Head Count')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()

def plot_range_scatter(predictions, targets, save_path, title, color='green'):
    """Plot scatter for a specific range of head counts."""
    plt.figure(figsize=(10, 6))
    plt.scatter(targets, predictions, alpha=0.5, color=color)
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
    plt.xlabel('True Head Count')
    plt.ylabel('Predicted Head Count')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, bins, save_path, title):
    """Plot confusion matrix for binned predictions."""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(bins) - 1))
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(bins) - 1)
    plt.xticks(tick_marks, [f'{bins[i]}-{bins[i+1]}' for i in range(len(bins) - 1)], rotation=45)
    plt.yticks(tick_marks, [f'{bins[i]}-{bins[i+1]}' for i in range(len(bins) - 1)])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model, _ = create_model()
    model = model.to(device)
    
    # Create optimizer (needed for loading checkpoint)
    optimizer = Adam(model.parameters(), lr=0.001)
    
    # Load best model
    checkpoint_path = 'checkpoints/best_model.pth'
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    epoch, val_loss, val_mae = load_checkpoint(model, optimizer, checkpoint_path)
    
    # Only create the test loader
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    test_dataset = HeadDetectionDataset(
        root_dir=r'C:\4 sem\final project\final\dataset\test',  # Updated path
        transform=test_transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Define bins for full range, lower range, and higher range
    bins = [0, 10, 20, 40, 60, 80]
    lower_bins = [0, 5, 10, 15, 20]
    higher_bins = [40, 50, 60, 70, 80]
    lower_range = 20
    higher_range_start = 40
    higher_range_end = 80  # For display purposes
    
    # Evaluate model
    try:
        (predictions, targets, binary_predictions, binary_targets, bins,
         lower_predictions, lower_targets, lower_binary_predictions, lower_binary_targets, lower_bins,
         higher_predictions, higher_targets, higher_binary_predictions, higher_binary_targets, higher_bins) = evaluate(
            model, test_loader, device, bins, lower_bins, higher_bins, lower_range, higher_range_start)
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return
    
    # Calculate regression metrics for full range
    try:
        metrics = calculate_metrics(predictions, targets)
    except Exception as e:
        print(f"Error calculating regression metrics: {e}")
        return
    
    # Calculate classification metrics for full range
    try:
        f1 = f1_score(binary_targets, binary_predictions, average='weighted')
        precision = precision_score(binary_targets, binary_predictions, average='weighted', zero_division=0)
        accuracy = accuracy_score(binary_targets, binary_predictions)
    except Exception as e:
        print(f"Error calculating classification metrics: {e}")
        return
    
    # Calculate regression metrics for lower range
    try:
        lower_metrics = calculate_metrics(lower_predictions, lower_targets)
    except Exception as e:
        print(f"Error calculating lower range regression metrics: {e}")
        return
    
    # Calculate classification metrics for lower range
    try:
        lower_f1 = f1_score(lower_binary_targets, lower_binary_predictions, average='weighted')
        lower_precision = precision_score(lower_binary_targets, lower_binary_predictions, average='weighted', zero_division=0)
        lower_accuracy = accuracy_score(lower_binary_targets, lower_binary_predictions)
    except Exception as e:
        print(f"Error calculating lower range classification metrics: {e}")
        return
    
    # Calculate regression metrics for higher range
    try:
        higher_metrics = calculate_metrics(higher_predictions, higher_targets)
    except Exception as e:
        print(f"Error calculating higher range regression metrics: {e}")
        return
    
    # Calculate classification metrics for higher range
    try:
        higher_f1 = f1_score(higher_binary_targets, higher_binary_predictions, average='weighted')
        higher_precision = precision_score(higher_binary_targets, higher_binary_predictions, average='weighted', zero_division=0)
        higher_accuracy = accuracy_score(higher_binary_targets, higher_binary_predictions)
    except Exception as e:
        print(f"Error calculating higher range classification metrics: {e}")
        return
    
    # Print all metrics
    print("\nTest Metrics (Full Range):")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAPE: {metrics['mape']:.2f}%")
    print(f"R-squared: {metrics['r2']:.4f}")
    print(f"F1 Score (binned): {f1:.4f}")
    print(f"Precision (binned): {precision:.4f}")
    print(f"Accuracy (binned): {accuracy:.4f}")
    print(f"Classification bins: {bins}")
    
    print(f"\nTest Metrics (Lower Range: 0-{lower_range}):")
    print(f"MAE: {lower_metrics['mae']:.4f}")
    print(f"RMSE: {lower_metrics['rmse']:.4f}")
    print(f"MAPE: {lower_metrics['mape']:.2f}%")
    print(f"R-squared: {lower_metrics['r2']:.4f}")
    print(f"F1 Score (binned): {lower_f1:.4f}")
    print(f"Precision (binned): {lower_precision:.4f}")
    print(f"Accuracy (binned): {lower_accuracy:.4f}")
    print(f"Classification bins: {lower_bins}")
    
    print(f"\nTest Metrics (Higher Range: {higher_range_start}-{higher_range_end}):")
    print(f"MAE: {higher_metrics['mae']:.4f}")
    print(f"RMSE: {higher_metrics['rmse']:.4f}")
    print(f"MAPE: {higher_metrics['mape']:.2f}%")
    print(f"R-squared: {higher_metrics['r2']:.4f}")
    print(f"F1 Score (binned): {higher_f1:.4f}")
    print(f"Precision (binned): {higher_precision:.4f}")
    print(f"Accuracy (binned): {higher_accuracy:.4f}")
    print(f"Classification bins: {higher_bins}")
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Plot predictions for full range
    try:
        plot_predictions(predictions, targets, save_path='results/predictions.png')
    except Exception as e:
        print(f"Error plotting full range predictions: {e}")
    
    # Plot scatter for lower range
    try:
        plot_range_scatter(lower_predictions, lower_targets, 
                          save_path='results/lower_range_predictions.png',
                          title=f'Predictions vs True Values (Lower Range: 0-{lower_range})',
                          color='green')
    except Exception as e:
        print(f"Error plotting lower range scatter: {e}")
    
    # Plot scatter for higher range
    try:
        plot_range_scatter(higher_predictions, higher_targets,
                          save_path='results/higher_range_predictions.png',
                          title=f'Predictions vs True Values (Higher Range: {higher_range_start}-{higher_range_end})',
                          color='purple')
    except Exception as e:
        print(f"Error plotting higher range scatter: {e}")
    
    # Plot histogram of binned counts for full range
    try:
        plot_bin_histogram(targets, predictions, bins, 
                          save_path='results/bin_histogram.png',
                          title='Histogram of True vs Predicted Head Counts (Full Range)')
    except Exception as e:
        print(f"Error plotting full range histogram: {e}")
    
    # Plot histogram of binned counts for lower range
    try:
        plot_bin_histogram(lower_targets, lower_predictions, lower_bins, 
                          save_path='results/lower_range_bin_histogram.png',
                          title=f'Histogram of True vs Predicted Head Counts (Lower Range: 0-{lower_range})')
    except Exception as e:
        print(f"Error plotting lower range histogram: {e}")
    
    # Plot histogram of binned counts for higher range
    try:
        plot_bin_histogram(higher_targets, higher_predictions, higher_bins,
                          save_path='results/higher_range_bin_histogram.png',
                          title=f'Histogram of True vs Predicted Head Counts (Higher Range: {higher_range_start}-{higher_range_end})')
    except Exception as e:
        print(f"Error plotting higher range histogram: {e}")
    
    # Plot confusion matrix for full range
    try:
        plot_confusion_matrix(binary_targets, binary_predictions, bins,
                            save_path='results/confusion_matrix_full.png',
                            title='Confusion Matrix (Full Range)')
    except Exception as e:
        print(f"Error plotting full range confusion matrix: {e}")
    
    # Plot confusion matrix for lower range
    try:
        plot_confusion_matrix(lower_binary_targets, lower_binary_predictions, lower_bins,
                            save_path='results/confusion_matrix_lower.png',
                            title=f'Confusion Matrix (Lower Range: 0-{lower_range})')
    except Exception as e:
        print(f"Error plotting lower range confusion matrix: {e}")
    
    # Plot confusion matrix for higher range
    try:
        plot_confusion_matrix(higher_binary_targets, higher_binary_predictions, higher_bins,
                            save_path='results/confusion_matrix_higher.png',
                            title=f'Confusion Matrix (Higher Range: {higher_range_start}-{higher_range_end})')
    except Exception as e:
        print(f"Error plotting higher range confusion matrix: {e}")
    
    # Save predictions to file
    try:
        results = np.column_stack((targets, predictions, binary_targets, binary_predictions))
        np.savetxt('results/predictions.csv', results, delimiter=',', 
                   header='True_Count,Predicted_Count,True_Bin,Predicted_Bin', comments='')
    except Exception as e:
        print(f"Error saving predictions: {e}")
    
    print("\nResults saved to 'results' directory")

if __name__ == '__main__':
    main()