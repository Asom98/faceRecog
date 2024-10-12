import pandas as pd
from PIL import Image
import os
import datetime


# Load and preprocess dataset
def resize_images(csv_filename: str, base_folder: str, target_folder: str):
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

    df = _load_csv(csv_filename)

    # Initialize lists for images and labels
    images = []
    labels = []

    # stop watch log start time
    start_time = datetime.datetime.now()
    print(r"Normalize images")
    print(f"Start time: {start_time}")

    # Iterate through CSV, load images, and resize them
    skipped_count = 0
    processed_count = 0
    for index, row in df.iterrows():
        path = row['path'].strip()
        image_path = os.path.join(base_folder, path)
        label = row['label']

        # Load and preprocess image
        try:
            save_path = os.path.join(target_folder, path)

            # Check if the file already exists
            if os.path.exists(save_path):
                skipped_count += 1
                print(f"Skipping {save_path} as it already normalized.")
                continue

            resized_image = Image.open(image_path).convert('RGB').resize((64, 64))

            # Save the image to the target folder
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))

            resized_image.save(save_path)

            labels.append(label)
            processed_count += 1
        except (IOError, OSError) as e:
            skipped_count += 1
            print(f"Skipping image {image_path} due to error: {e}")
            continue  # Skip corrupted or unreadable images

    # stop watch log end time
    end_time = datetime.datetime.now()
    print(f"End time: {end_time}")
    print(f"duration: {end_time - start_time}")
    print(f"processed images: {processed_count}")
    print(f"skipped images: {skipped_count}")


def grayscale_images(csv_filename: str, base_folder: str, target_folder: str):
    df = _load_csv(csv_filename)

    start_time = datetime.datetime.now()
    print(r"RGB to Grayscale images")
    print(f"Start time: {start_time}")

    # Iterate through CSV, load images, and resize them
    skipped_count = 0
    processed_count = 0
    for index, row in df.iterrows():
        path = row['path'].strip()
        image_path = os.path.join(base_folder, path)
        label = row['label']

        # Load and preprocess image
        try:
            save_path = os.path.join(target_folder, path)

            # Check if the file already exists
            if os.path.exists(save_path):
                skipped_count += 1
                print(f"Skipping {save_path} as it already normalized.")
                continue

            grayscale_image = Image.open(image_path).convert('L')

            # Save the image to the target folder
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))

            grayscale_image.save(save_path)
            processed_count += 1
        except (IOError, OSError) as e:
            skipped_count += 1
            print(f"Skipping image {image_path} due to error: {e}")
            continue  # Skip corrupted or unreadable images

    # stop watch log end time
    end_time = datetime.datetime.now()
    print(f"End time: {end_time}")
    print(f"duration: {end_time - start_time}")
    print(f"processed images: {processed_count}")
    print(f"skipped images: {skipped_count}")


def _load_csv(csv_filename: str):
    df = pd.read_csv(csv_filename)
    # Remove "Ahegao" class if present
    return df[df['label'] != 'Ahegao']
