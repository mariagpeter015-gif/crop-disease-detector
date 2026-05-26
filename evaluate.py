import torch
import matplotlib.pyplot as plt
import seaborn as sns

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# =========================
# Image Transforms
# (NO augmentation for evaluation)
# =========================
transform = transforms.Compose([

    transforms.Resize((128, 128)),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5]
    )
])

# =========================
# Test Dataset
# =========================
test_dataset = datasets.ImageFolder(
    root=r"C:\Users\maria\OneDrive\Documents\crop-disease-detector\dataset\Tomato Leaf Disease\test",
    transform=transform
)

# =========================
# DataLoader
# =========================
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

# =========================
# Class Names
# =========================
classes = test_dataset.classes

print("Classes:", classes)

# =========================
# Load ResNet18 Model
# =========================
model = models.resnet18(weights=None)

# Replace classifier head
model.fc = torch.nn.Sequential(

    torch.nn.Linear(
        model.fc.in_features,
        128
    ),

    torch.nn.ReLU(),

    torch.nn.Dropout(0.3),

    torch.nn.Linear(128, 3)
)

# =========================
# Load Trained Weights
# =========================
model.load_state_dict(torch.load(
    r"C:\Users\maria\OneDrive\Documents\crop-disease-detector\crop_disease_resnet.pth",
    map_location=device
))

model = model.to(device)

model.eval()

# =========================
# Store Predictions
# =========================
y_true = []

y_pred = []

# =========================
# Evaluation
# =========================
with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        y_true.extend(labels.cpu().numpy())

        y_pred.extend(predicted.cpu().numpy())

# =========================
# Confusion Matrix
# =========================
cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:\n")

print(cm)

# =========================
# Classification Report
# =========================
print("\nClassification Report:\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=classes
    )
)

# =========================
# Plot Confusion Matrix
# =========================
plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=classes,
    yticklabels=classes
)

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.title("Confusion Matrix")

plt.show()