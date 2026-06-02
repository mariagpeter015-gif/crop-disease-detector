# Crop Disease Detection using Deep Learning

This project is part of an internship task focused on building a Computer Vision system for autonomous crop disease detection using deep learning. The internship is conducted by Zelbytes Pvt Ltd., Thiruvananthapuram, Kerala, India.

---

## Project Objective

To develop an end-to-end deep learning system that classifies crop leaf images as healthy or diseased and provides predictions with confidence scores, along with cure and prevention suggestions.

---

## Dataset

The project uses the **Tomato Disease Dataset**, which contains images of healthy and diseased tomato crop leaves.

For this task, the dataset was reorganized into three classes:

- Tomato___Healthy  
- Tomato___Early_blight  
- Tomato___Late_blight  

---

## Methodology

### 1. Data Preprocessing
- Resized images to 128×128 pixels
- Converted images to tensor format
- Normalized pixel values

### 2. Data Augmentation (Training Only)
- Random horizontal flips
- Random rotations
- Color jitter (brightness, contrast, saturation)

### 3. Model Architecture
- Transfer Learning using **ResNet18**
- Pretrained on ImageNet
- Custom classifier head added:
  - Linear layer (128 neurons)
  - ReLU activation
  - Dropout (0.3)
  - Final output layer (3 classes)

### 4. Training Strategy
- Frozen base layers of ResNet18
- Optimizer: Adam
- Loss Function: CrossEntropyLoss
- Weighted loss to handle class imbalance
- Trained for multiple epochs with validation monitoring

---

## Evaluation Metrics

- Confusion Matrix
- Precision
- Recall
- F1-score
- Classification Report (sklearn)

---

## Model Deployment

### 🔹 FastAPI Backend
- Model served using FastAPI
- Endpoint: `/predict`
- Accepts image uploads
- Returns:
  - Predicted class
  - Confidence score

### 🔹 Docker Containerization
- Model packaged inside Docker container
- Ensures portability and deployment consistency

### 🔹 Integration Testing
- Python `requests` client sends images to API
- Receives and logs predictions from containerized service

---

## Frontend (Streamlit)

A simple web interface built using Streamlit that allows users to:

- Upload leaf images
- Get disease predictions
- View confidence scores
- Receive:
  - Cure recommendations
  - Prevention tips

---

## Technologies Used

- Python  
- PyTorch  
- Torchvision  
- FastAPI  
- Docker  
- Streamlit  
- Scikit-learn  
- Matplotlib / Seaborn  
- VS Code  

---

## Project Structure

```
crop-disease-detector/
│
├── train.py
├── app.py
├── predict.py
├── test_client.py
├── frontend.py
├── Dockerfile
├── crop_disease_resnet.pth
├── dataset/
├── test_images/
├── .gitignore
└── README.md
```
---

## How to Run the Project

### 1. Install dependencies
```bash
pip install -r requirements.txt
```
### 2. Run FastAPI backend (Docker)
```bash
docker run -p 8000:8000 crop-disease-api
```
### 3. Run frontend (Streamlit)
```bash
streamlit run frontend.py
```
### 4. Run integration test
```bash
python test_client.py
```
---

## Results

High validation accuracy achieved using ResNet18 transfer learning
Robust performance due to data augmentation
Reliable predictions across multiple disease classes

---

## Future Improvements

Expand to multiple crop species
Deploy on cloud (AWS / Render / Streamlit Cloud)
Improve UI with advanced dashboards
Add real-time camera detection

---

## Author

Internship Project – Zelbytes Pvt Ltd
Computer Vision Engineer Internship
Thiruvananthapuram, Kerala

---

## Conclusion

This project demonstrates a complete end-to-end deep learning pipeline including dataset preparation, model training, evaluation, API deployment, Docker containerization, integration testing, and frontend development.
