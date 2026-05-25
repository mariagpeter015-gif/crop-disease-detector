import torch
from PIL import Image
from torchvision import transforms, datasets
from model import CNN

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = CNN().to(device)

model.load_state_dict(torch.load(
    r"C:\Users\maria\OneDrive\Documents\crop-disease-detector\crop_disease_model.pth",
    map_location=device
))

model.eval()

# Load class names automatically from dataset
temp_dataset = datasets.ImageFolder(
    r"C:\Users\maria\OneDrive\Documents\crop-disease-detector\dataset\Tomato Leaf Disease\train"
)

classes = temp_dataset.classes

print("Class Mapping:", classes)

# Image transforms (MUST match training)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5]
    )
])

# Image path
img_path = r"C:\Users\maria\OneDrive\Documents\crop-disease-detector\leafeb.jpg"

# Load image
image = Image.open(img_path).convert("RGB")

# Apply transforms
image = transform(image).unsqueeze(0).to(device)

# Prediction
with torch.no_grad():

    output = model(image)

    probs = torch.softmax(output, dim=1)

    confidence, predicted = torch.max(probs, 1)

# Output
print("Prediction:", classes[predicted.item()])

print("Confidence:", round(confidence.item() * 100, 2), "%")