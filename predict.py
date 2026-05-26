import sys
import torch

from PIL import Image
from torchvision import transforms, datasets, models

# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Check Command Line Input
# =========================
if len(sys.argv) != 2:

    print("Usage: python predict.py <image_path>")

    sys.exit()

# =========================
# Get Image Path
# =========================
img_path = sys.argv[1]

# =========================
# Load Class Names
# =========================
temp_dataset = datasets.ImageFolder(
    r"C:\Users\maria\OneDrive\Documents\crop-disease-detector\dataset\Tomato Leaf Disease\train"
)

classes = temp_dataset.classes

# =========================
# Image Transforms
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
# Load Image using PIL
# =========================
image = Image.open(img_path).convert("RGB")

# =========================
# Apply Transforms
# =========================
image = transform(image)

# Add batch dimension
image = image.unsqueeze(0)

image = image.to(device)

# =========================
# Prediction
# =========================
with torch.no_grad():

    output = model(image)

    probs = torch.softmax(output, dim=1)

    confidence, predicted = torch.max(probs, 1)

# =========================
# Output
# =========================
print("\nPrediction:")

print(classes[predicted.item()])

print("\nConfidence:")

print(round(confidence.item() * 100, 2), "%")