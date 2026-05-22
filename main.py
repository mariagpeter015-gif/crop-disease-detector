import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Image transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Load dataset
dataset = datasets.ImageFolder(
    root='C:/Users/maria/OneDrive/Documents/crop-disease-detector/binary_dataset',
    transform=transform
)

# Create DataLoader
train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

# Print dataset information
print("Classes:", dataset.classes)
print("Total Images:", len(dataset))