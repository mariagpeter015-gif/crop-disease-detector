import os
import shutil

source_dir = r"C:\Users\crop-disease-detector\dataset\PlantVillage"
target_dir = r"C:\Users\crop-disease-detector\binary_dataset"

healthy_dir = os.path.join(target_dir, "healthy")
diseased_dir = os.path.join(target_dir, "diseased")

os.makedirs(healthy_dir, exist_ok=True)
os.makedirs(diseased_dir, exist_ok=True)

for folder_name in os.listdir(source_dir):

    folder_path = os.path.join(source_dir, folder_name)

    if os.path.isdir(folder_path):

        # Decide destination
        if "healthy" in folder_name.lower():
            destination = healthy_dir
        else:
            destination = diseased_dir

        # Copy images
        for image_name in os.listdir(folder_path):

            src = os.path.join(folder_path, image_name)
            dst = os.path.join(destination, image_name)

            shutil.copy(src, dst)

print("Binary dataset created successfully!")
