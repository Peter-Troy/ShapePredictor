import os
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

DATASET_DIR = "dataset"
PROCESSED_DIR = "processed"
IMG_SIZE = 64

os.makedirs(PROCESSED_DIR, exist_ok=True)

# Image transformations
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize pixels to [-1, 1]
])

for shape in os.listdir(DATASET_DIR):
    input_folder = os.path.join(DATASET_DIR, shape)
    output_folder = os.path.join(PROCESSED_DIR, shape)
    os.makedirs(output_folder, exist_ok=True)

    for file in tqdm(os.listdir(input_folder), desc=f"Processing {shape}"):
        img_path = os.path.join(input_folder, file)
        img = Image.open(img_path).convert("L")  # Convert to grayscale
        img = transform(img)
        torch.save(img, os.path.join(output_folder, file.replace(".png", ".pt")))

print("Preprocessing complete!")
