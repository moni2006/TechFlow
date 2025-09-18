from datasets import load_dataset
import os
from PIL import Image

# Load the Fashionpedia dataset
dataset = load_dataset("detection-datasets/fashionpedia")

# Define the folder to save images
dataset_folder = 'Data'
os.makedirs(dataset_folder, exist_ok=True)

# Function to save images
def save_images(dataset, dataset_folder, num_images=1000):
    for i in range(num_images):
        # Access the image directly as a PIL Image
        pil_image = dataset['train'][i]['image']  # This should already be a PIL Image
        
        # Save the image as a .png file
        pil_image.save(os.path.join(dataset_folder, f'image_{i+1}.png'))

# Save the first 1000 images
save_images(dataset, dataset_folder, num_images=1000)

print(f"Saved the first 1000 images to {dataset_folder}")
