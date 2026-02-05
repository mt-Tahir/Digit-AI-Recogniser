import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from model import MyModel

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 100% BEST TRANSFORM: This solves the "Nara Mara" (messy) problem ---
train_transform = transforms.Compose([
    transforms.RandomRotation(20),             # Handles tilted drawings
    transforms.RandomAffine(0, translate=(0.15, 0.15), scale=(0.8, 1.2)), # Handles off-center/size
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))       # Stabilizes the "brain" of the model
])

# Load Dataset
train_dataset = datasets.MNIST(root="data", train=True, download=True, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialize
model = MyModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- Training Loop ---
epochs = 30  # Increased to 30 for better deep learning
print(f"Starting Training on {device}...")

for epoch in range(epochs):
    running_loss = 0
    model.train() # Set to training mode
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

# --- Save Final Model ---
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/mymodel.pth")
print("\nSuccess! Strong model saved to models/mymodel.pth")