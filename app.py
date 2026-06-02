import io

import torch
import torch.nn as nn

from PIL import Image

from fastapi import FastAPI, File, UploadFile
from torchvision import transforms, models

# =========================
# FastAPI App
# =========================
app = FastAPI()

# =========================
# Device
# =========================
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

# =========================
# Class Names
# =========================
classes = [
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy"
]

print("Class Mapping:", classes)

# =========================
# Load ResNet18 Model
# =========================
model = models.resnet18(weights=None)

model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 3)
)

# =========================
# Load Trained Weights
# =========================
model.load_state_dict(
    torch.load(
        "crop_disease_resnet.pth",
        map_location=device
    )
)

model = model.to(device)

model.eval()

print("Model Loaded Successfully")

# =========================
# Image Transform
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
# Home Endpoint
# =========================
@app.get("/")
def home():

    return {
        "message": "Crop Disease Detection API Running"
    }

# =========================
# Prediction Endpoint
# =========================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    image = image.to(device)

    with torch.no_grad():

        output = model(image)

        probs = torch.softmax(
            output,
            dim=1
        )

        confidence, predicted = torch.max(
            probs,
            1
        )

    prediction = classes[predicted.item()]

    confidence = round(
        confidence.item() * 100,
        2
    )

    return {
        "prediction": prediction,
        "confidence": confidence
    }