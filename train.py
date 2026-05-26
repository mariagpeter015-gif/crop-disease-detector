import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from collections import Counter

# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# =========================
# Train Transforms (TASK 4)
# =========================
train_transform = transforms.Compose([

    transforms.Resize((128, 128)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(15),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5]
    )
])

# =========================
# Test Transforms
# =========================
test_transform = transforms.Compose([

    transforms.Resize((128, 128)),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5]
    )
])

# =========================
# Train Dataset
# =========================
train_dataset = datasets.ImageFolder(
    root=r"C:\Users\maria\OneDrive\Documents\crop-disease-detector\dataset\Tomato Leaf Disease\train",
    transform=train_transform
)

# =========================
# Test Dataset
# =========================
test_dataset = datasets.ImageFolder(
    root=r"C:\Users\maria\OneDrive\Documents\crop-disease-detector\dataset\Tomato Leaf Disease\test",
    transform=test_transform
)

# =========================
# DataLoaders
# =========================
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

# =========================
# Dataset Information
# =========================
print("Classes:", train_dataset.classes)

print("Class mapping:", train_dataset.class_to_idx)

train_labels = [label for _, label in train_dataset.samples]

print("Train distribution:", Counter(train_labels))

print("Train Images:", len(train_dataset))

print("Test Images:", len(test_dataset))

# =========================
# RESNET18 MODEL (TASK 5)
# =========================
model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)

# Freeze pretrained layers
for param in model.parameters():
    param.requires_grad = False

# Replace classifier head
num_features = model.fc.in_features

model.fc = nn.Sequential(

    nn.Linear(num_features, 128),

    nn.ReLU(),

    nn.Dropout(0.3),

    nn.Linear(128, 3)
)

model = model.to(device)

# =========================
# Loss Function
# =========================
weights = torch.tensor([1.8, 1.0, 1.2]).to(device)

criterion = nn.CrossEntropyLoss(weight=weights)

# =========================
# Optimizer
# =========================
optimizer = optim.Adam(
    model.fc.parameters(),
    lr=0.001
)

# =========================
# Training Setup
# =========================
epochs = 10

train_losses = []

best_accuracy = 0.0

# =========================
# Training Loop
# =========================
for epoch in range(epochs):

    model.train()

    running_loss = 0.0

    for images, labels in train_loader:

        images = images.to(device)

        labels = labels.to(device)

        # Clear old gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        running_loss += loss.item()

    # =========================
    # Average Training Loss
    # =========================
    avg_loss = running_loss / len(train_loader)

    train_losses.append(avg_loss)

    print(f"\nEpoch [{epoch+1}/{epochs}]")

    print(f"Training Loss: {avg_loss:.4f}")

    # =========================
    # Validation
    # =========================
    model.eval()

    correct = 0

    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    print(f"Validation Accuracy: {accuracy:.2f}%")

    # =========================
    # Save Best Model
    # =========================
    if accuracy > best_accuracy:

        best_accuracy = accuracy

        torch.save(
            model.state_dict(),
            r"C:\Users\maria\OneDrive\Documents\crop-disease-detector\crop_disease_resnet.pth"
        )

        print("Best model saved!")

# =========================
# Training Finished
# =========================
print("\nTraining Completed!")

print(f"Best Accuracy: {best_accuracy:.2f}%")

# =========================
# Plot Loss Curve
# =========================
plt.plot(train_losses)

plt.title("Training Loss Curve (ResNet18)")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.show()