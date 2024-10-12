import pandas as pd
import numpy as np
from PIL import Image
import os
import datetime


# Load and preprocess dataset
def normalize_images(csv_filename: str, base_folder: str, target_folder: str):
    """
    Load and preprocess dataset from a CSV file.
    1. remove Ahegao
    2. resize image to 64x64
    3. normalize
    4. save new images in target folder

    Args:
        csv_filename (str): Path to the CSV file containing image paths and labels.
        base_folder (str): Base folder where images are stored.
        target_folder (str): Target folder where images will be saved after normalization.
    """

    df = pd.read_csv(csv_filename)

    # Remove "Ahegao" class if present
    df = df[df['label'] != 'Ahegao']

    # Initialize lists for images and labels
    images = []
    labels = []

    # stop watch log start time
    start_time = datetime.datetime.now()
    print(f"Start time: {start_time}")

    # Iterate through CSV, load images, and resize them
    for index, row in df.iterrows():
        path = row['path'].strip()
        image_path = os.path.join(base_folder, path)
        label = row['label']

        # Load and preprocess image
        try:
            save_path = os.path.join(target_folder, path)

            # Check if the file already exists
            if os.path.exists(save_path):
                print(f"Skipping {save_path} as it already normalized.")
                continue

            cropped_image = Image.open(image_path).convert('RGB').resize((64, 64))
            normalized_image = np.array(cropped_image) / 255.0  # Normalize to [0, 1]

            # Save the image to the target folder
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))

            cropped_image.save(save_path)

            images.append(normalized_image)
            labels.append(label)
        except (IOError, OSError) as e:
            print(f"Skipping image {image_path} due to error: {e}")
            continue  # Skip corrupted or unreadable images

    # stop watch log end time
    end_time = datetime.datetime.now()
    print(f"End time: {end_time}")
    print(f"duration: {end_time - start_time}")

    # save result in new folder

    return np.array(images), np.array(labels)
