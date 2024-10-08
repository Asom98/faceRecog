import pandas as pd
import numpy as np
from PIL import Image
import os

# Load and preprocess dataset
def load_data(csv_filename, base_folder, image_size=(64, 64)):
    df = pd.read_csv(csv_filename)

    # Remove "Ahegao" class if present
    df = df[df['label'] != 'Ahegao']

    # Initialize lists for images and labels
    images = []
    labels = []

    # Iterate through CSV, load images, and resize them
    for index, row in df.iterrows():
        img_path = os.path.join(base_folder, row['path'].strip())
        label = row['label']

        # Load and preprocess image
        try:
            img = Image.open(img_path).convert('RGB').resize(image_size)
            img = np.array(img) / 255.0  # Normalize to [0, 1]
            images.append(img)
            labels.append(label)
        except (IOError, OSError) as e:
            print(f"Skipping image {img_path} due to error: {e}")
            continue  # Skip corrupted or unreadable images

    return np.array(images), np.array(labels)
