import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
import json

def save_checkpoint(state, filename):
    """
    Save model checkpoint
    """
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(model, optimizer, filename):
    """
    Load model checkpoint
    """
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        val_loss = checkpoint['val_loss']
        val_mae = checkpoint['val_mae']
        print(f"Loaded checkpoint from epoch {epoch}")
        print(f"Validation Loss: {val_loss:.4f}, Validation MAE: {val_mae:.4f}")
        return epoch, val_loss, val_mae
    else:
        print(f"No checkpoint found at {filename}")
        return 0, float('inf'), float('inf')

def plot_predictions(predictions, targets, save_path=None):
    """
    Plot predictions vs targets
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(targets, predictions, alpha=0.5)
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
    plt.xlabel('True Head Count')
    plt.ylabel('Predicted Head Count')
    plt.title('Predictions vs True Values')
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def calculate_metrics(predictions, targets):
    """
    Calculate various metrics for evaluation
    """
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(predictions - targets))
    
    # Root Mean Square Error
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    
    # Mean Absolute Percentage Error
    # Avoid division by zero by adding a small epsilon
    epsilon = 1e-10
    mape = np.mean(np.abs((targets - predictions) / (targets + epsilon))) * 100
    
    # R-squared
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + epsilon))
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2
    }

def test_single_image(model, image_path, device):
    """
    Test the model with a single image and return the prediction
    """
    # Load and preprocess the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        prediction = model(image_tensor)
    
    # Get ground truth from COCO annotation file
    root_dir = os.path.dirname(image_path)
    annotation_path = os.path.join(root_dir, '_annotations.coco.json')
    ground_truth = None
    if os.path.exists(annotation_path):
        with open(annotation_path, 'r') as f:
            coco = json.load(f)
        
        # Get image filename from path
        img_name = os.path.basename(image_path)
        
        # Build image_id to filename mapping
        image_id_to_filename = {img['id']: img['file_name'] for img in coco['images']}
        filename_to_image_id = {img['file_name']: img['id'] for img in coco['images']}
        
        # Count heads for the specific image
        image_id = filename_to_image_id.get(img_name)
        if image_id is not None:
            head_count = 0
            for ann in coco['annotations']:
                if ann['image_id'] == image_id and coco['categories'][ann['category_id']]['name'] == 'head':
                    head_count += 1
            ground_truth = float(head_count)
    
    return prediction.item(), ground_truth, image

def visualize_prediction(image, prediction, ground_truth=None, save_path=None):
    """
    Visualize the prediction on a single image
    """
    plt.figure(figsize=(10, 6))
    plt.imshow(image)
    title = f'Predicted Heads: {prediction:.1f}'
    if ground_truth is not None:
        title += f' (Ground Truth: {ground_truth:.1f})'
    plt.title(title)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path)
    plt.close()