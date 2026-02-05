import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Layer 1: Detects simple edges
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) # Keeps math stable
        
        # Layer 2: Detects complex shapes (loops, curves)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout: This is the "secret sauce" to handle messy drawings
        self.dropout = nn.Dropout(0.25)
        
        # Fully Connected Layers
        # After two 2x2 pools, 28x28 becomes 7x7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # Block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Flatten
        x = x.view(-1, 64 * 7 * 7)
        
        # Fully Connected with Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x) # Randomly ignores neurons to make the brain stronger
        x = self.fc2(x)
        return x