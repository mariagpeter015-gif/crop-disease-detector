import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from collections import Counter

from model import CNN

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# Transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5]
    )
])

# Train dataset
train_dataset = datasets.ImageFolder(
    root=r"C:\Users\maria\OneDrive\Documents\crop-disease-detector\dataset\Tomato Leaf Disease\train",
    transform=transform
)

# Test dataset
test_dataset = datasets.ImageFolder(
    root=r"C:\Users\maria\OneDrive\Documents\crop-disease-detector\dataset\Tomato Leaf Disease\test",
    transform=transform
)

# DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

# Dataset information
print("Classes:", train_dataset.classes)

print("Class mapping:", train_dataset.class_to_idx)

train_labels = [label for _, label in train_dataset.samples]

print("Train distribution:", Counter(train_labels))

print("Train Images:", len(train_dataset))

print("Test Images:", len(test_dataset))

# Load model
model = CNN().to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

# Store losses
train_losses = []

# Number of epochs
epochs = 5

# Training loop
for epoch in range(epochs):

    model.train()

    running_loss = 0.0

    for images, labels in train_loader:

        images, labels = images.to(device), labels.to(device)

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        running_loss += loss.item()

    # Average loss
    avg_loss = running_loss / len(train_loader)

    train_losses.append(avg_loss)

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    # Validation
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    print(f"Validation Accuracy: {accuracy:.2f}%")

# Save model
torch.save(
    model.state_dict(),
    r"C:\Users\maria\OneDrive\Documents\crop-disease-detector\crop_disease_model.pth"
)

print("Training completed!")

# Plot training loss curve
plt.plot(train_losses)

plt.title("Training Loss Curve")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.show()