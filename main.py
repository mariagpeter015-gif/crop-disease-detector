import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Device check (optional)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Transform (same as training)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Load dataset
dataset = datasets.ImageFolder(
    root=r'C:\Users\maria\OneDrive\Documents\crop-disease-detector\dataset\Tomato Leaf Disease\train',
    transform=transform
)

# DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Debug info
print("Classes:", dataset.classes)
print("Class mapping:", dataset.class_to_idx)
print("Total Images:", len(dataset))