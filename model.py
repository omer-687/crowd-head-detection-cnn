import torch
import torch.nn as nn
import torch.nn.functional as F

class HeadDetectionModel(nn.Module):
    def __init__(self):
        super(HeadDetectionModel, self).__init__()
        
        # Feature extraction layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 14 * 14, 512)  # Assuming input size of 224x224
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)  # Output: single value for head count
        
    def forward(self, x):
        # Feature extraction
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(-1, 256 * 14 * 14)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class HeadDetectionLoss(nn.Module):
    def __init__(self):
        super(HeadDetectionLoss, self).__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, predictions, targets):
        # MSE loss for regression
        mse_loss = self.mse(predictions, targets)
        
        # Add L1 loss for better handling of outliers
        l1_loss = F.l1_loss(predictions, targets)
        
        # Combine losses
        total_loss = mse_loss + 0.1 * l1_loss
        return total_loss

def create_model():
    model = HeadDetectionModel()
    criterion = HeadDetectionLoss()
    return model, criterion 