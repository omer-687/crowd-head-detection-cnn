import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import json
import numpy as np

class HeadDetectionDataset(Dataset):
    def __init__(self, root_dir, transform=None, annotation_file='_annotations.coco.json'):
        """
        Args:
            root_dir (string): Directory with all the images and annotations
            transform (callable, optional): Optional transform to be applied on a sample
            annotation_file (string): COCO annotation file name
        """
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        self.annotation_path = os.path.join(root_dir, annotation_file)
        
        # Debug: Check if the annotation file exists
        if not os.path.exists(self.annotation_path):
            raise FileNotFoundError(f"Annotation file not found at {self.annotation_path}")
        
        # Load COCO annotation file
        with open(self.annotation_path, 'r') as f:
            coco = json.load(f)
        
        # Build image_id to filename mapping
        self.image_id_to_filename = {img['id']: img['file_name'] for img in coco['images']}
        self.filename_to_image_id = {img['file_name']: img['id'] for img in coco['images']}
        
        # Build image_id to head count mapping
        self.image_id_to_count = {img_id: 0 for img_id in self.image_id_to_filename.keys()}
        for ann in coco['annotations']:
            if coco['categories'][ann['category_id']]['name'] == 'head':
                self.image_id_to_count[ann['image_id']] += 1
        
        # List of image files (only those in the annotation file)
        self.image_files = [img['file_name'] for img in coco['images']]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # Get head count from annotation
        image_id = self.filename_to_image_id[img_name]
        num_heads = float(self.image_id_to_count[image_id])
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(num_heads, dtype=torch.float32)

def get_data_loaders(batch_size=32, num_workers=4):
    """
    Create data loaders for training, validation, and testing
    """
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Basic transform for validation and testing
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Update paths to reflect the new project directory with dataset subdirectory
    base_dir = r'C:\4 sem\final project\final\dataset'
    train_dataset = HeadDetectionDataset(
        root_dir=os.path.join(base_dir, 'train'),
        transform=train_transform
    )
    
    val_dataset = HeadDetectionDataset(
        root_dir=os.path.join(base_dir, 'valid'),
        transform=val_transform
    )
    
    test_dataset = HeadDetectionDataset(
        root_dir=os.path.join(base_dir, 'test'),
        transform=val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader