# Crop Disease Detection using Deep Learning (Computer Vision Engineer)

This project is part of the internship task focused on building a Computer Vision system for autonomous quality inspection of crops using deep learning. The internship is conducted by Zelbytes Pvt Ltd. at Thiruvananthapuram, Kerala, India.

---

## Task 1: Framework Initialization

The goal of this task was to set up a deep learning environment and prepare the dataset pipeline for training a model to classify crops as **healthy or diseased**.

---

## Dataset

The project uses the **PlantVillage dataset**, which contains images of healthy and diseased crop leaves across multiple plant species.

For this task, the dataset was reorganized into two categories:
- Healthy crops
- Diseased crops

---

## Methodology

- Set up Python environment with PyTorch
- Installed required libraries: `torch`, `torchvision`
- Loaded dataset using `torchvision.datasets.ImageFolder`
- Applied image preprocessing:
  - Resize images to 128×128
  - Convert images to tensor format
- Created DataLoader to batch and shuffle data efficiently
- Verified dataset loading by printing class labels and total image count

---

## Technologies Used

- Python
- PyTorch
- Torchvision
- VS Code

---

## Project Structure
```text
crop-disease-detector/
│
├── main.py                  
├── model.py               
├── train.py                
├── test.py                 
├── dataset.py              
├── organize_dataset.py     
├── .gitignore              
└── README.md             
```
---


---

## How to Run

1. Clone the repository
2. Install dependencies
3. Run the dataset loader

---



