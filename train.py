import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
from datetime import datetime
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from torch.utils.tensorboard import SummaryWriter
import glob

from model import create_model
from dataset import get_data_loaders
from utils import save_checkpoint, load_checkpoint, test_single_image, visualize_prediction

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    predictions = []
    targets = []
    
    progress_bar = tqdm(train_loader, desc='Training')
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels.view(-1, 1))
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Update statistics
        total_loss += loss.item()
        predictions.extend(outputs.detach().cpu().numpy())
        targets.extend(labels.cpu().numpy())
        
        # Update progress bar
        progress_bar.set_postfix({'loss': loss.item()})
    
    # Calculate metrics
    avg_loss = total_loss / len(train_loader)
    mae = np.mean(np.abs(np.array(predictions) - np.array(targets)))
    
    return avg_loss, mae

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels.view(-1, 1))
            
            # Update statistics
            total_loss += loss.item()
            predictions.extend(outputs.cpu().numpy())
            targets.extend(labels.cpu().numpy())
    
    # Calculate metrics
    avg_loss = total_loss / len(val_loader)
    mae = np.mean(np.abs(np.array(predictions) - np.array(targets)))
    
    return avg_loss, mae

def train(config):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model and move to device
    model, criterion = create_model()
    model = model.to(device)
    
    # Create optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Check for and load latest checkpoint
    checkpoint_path = 'checkpoints/latest_model.pth'
    start_epoch = 0
    best_val_loss = float('inf')
    if os.path.isfile(checkpoint_path):
        start_epoch, val_loss, val_mae = load_checkpoint(model, optimizer, checkpoint_path)
        best_val_loss = val_loss
        print(f"Resuming training from epoch {start_epoch}")
    
    # Get data loaders
    train_loader, val_loader, _ = get_data_loaders(
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    # Create directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('sample_predictions', exist_ok=True)
    
    # Get a sample image for testing
    sample_images = glob.glob(r'C:\4 sem\final project\final\dataset\valid\*.jpg') + glob.glob(r'C:\4 sem\final project\final\dataset\valid\*.png')
    if not sample_images:
        raise FileNotFoundError("No sample images found in validation set")
    sample_image_path = sample_images[0]
    print(f"Using sample image: {sample_image_path}")
    
    # Initialize tensorboard
    writer = SummaryWriter(f'runs/head_detection_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    
    # Training loop
    for epoch in range(start_epoch, config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        
        # Train
        train_loss, train_mae = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_mae = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('MAE/train', train_mae, epoch)
        writer.add_scalar('MAE/val', val_mae, epoch)
        
        print(f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}")
        
        # Test with sample image
        prediction, ground_truth, image = test_single_image(model, sample_image_path, device)
        print(f"\nSample Image Prediction:")
        print(f"Predicted Heads: {prediction:.1f}")
        if ground_truth is not None:
            print(f"Ground Truth: {ground_truth:.1f}")
        
        # Save sample prediction visualization
        visualize_prediction(
            image, 
            prediction, 
            ground_truth,
            save_path=f'sample_predictions/epoch_{epoch+1:03d}.png'
        )
        
        # Save checkpoint if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_mae': val_mae
            }, f'checkpoints/best_model.pth')
            print(f"New best model saved! (Val Loss: {val_loss:.4f})")
        
        # Save latest checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_mae': val_mae
        }, f'checkpoints/latest_model.pth')
    
    writer.close()
    print("Training completed!")

if __name__ == '__main__':
    config = {
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 100,
        'num_workers': 4
    }
    
    train(config)