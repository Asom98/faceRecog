import pandas as pd
import numpy as np
from PIL import Image
import os
import datetime
from numpy import ndarray
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# Load and preprocess dataset
def resize_and_save(csv_filename: str, base_folder: str, target_folder: str) -> None:
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

    # stop watch log start time
    start_time = datetime.datetime.now()
    print(r"Resize images")
    print(f"Start time: {start_time}")

    # Iterate through CSV, load images, and resize them
    skipped_count = 0
    processed_count = 0
    for index, row in df.iterrows():
        path = row['path'].strip()
        image_path = os.path.join(base_folder, path)

        # Load and preprocess image
        try:
            save_path = os.path.join(target_folder, path)

            # Check if the file already exists
            if os.path.exists(save_path):
                skipped_count += 1
                print(f"Skipping {save_path} as it already resized.")
                continue

            resized_image = Image.open(image_path).convert('RGB').resize((64, 64))

            # Save the image to the target folder
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))

            resized_image.save(save_path)

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


def load_and_normalize(csv_filename: str, base_folder: str) -> tuple[ndarray, ndarray]:
    """
    load resized image and normalize them, then return the array of images and label

    :param csv_filename:
    :param base_folder:
    :return: (ndarray, ndarray)
    """
    df = _load_csv(csv_filename)

    # Initialize lists for images and labels
    images = []
    labels = []

    # Iterate through CSV, load images, and resize them
    for index, row in df.iterrows():
        image_path = os.path.join(base_folder, row['path'].strip())
        label = row['label']

        # Load and preprocess image
        try:
            image = Image.open(image_path)
            normalize_image = np.array(image) / 255.0  # Normalize to [0, 1]
            images.append(normalize_image)
            labels.append(label)
        except (IOError, OSError) as e:
            print(f"Skipping image {image_path} due to error: {e}")
            continue  # Skip corrupted or unreadable images

    return np.array(images), np.array(labels)


def load_digits_images(base_folder: str) -> tuple[ndarray, ndarray]:
    """
    load resized image and normalize them, then return the array of images and label

    :param base_folder:
    :return: (ndarray, ndarray)
    """

    # Initialize lists for images and labels
    images = []
    labels = []

    # Iterate through CSV, load images, and resize them
    # loop from 0 to 9
    for digit_folder in range(10):
        # load all files in sub folder and label it
        label = str(digit_folder)
        folder = os.path.join(base_folder, label)
        print(f"Opening folder {folder} ...")

        files = load_all_files_in_folder(folder)
        for file in files:
            # Load and preprocess image
            try:
                image = Image.open(file)
                normalize_image = np.array(image) / 255.0  # Normalize to [0, 1]
                images.append(normalize_image)
                labels.append(label)
            except (IOError, OSError) as e:
                print(f"Skipping image {file} due to error: {e}")
                continue  # Skip corrupted or unreadable images

    return np.array(images), np.array(labels)


def load_all_files_in_folder(folder_path: str):
    """
    Load all files in the specified folder.

    Args:
        folder_path (str): Path to the folder containing files.

    Returns:
        list: A list of file paths.
    """
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths


def to_grayscale(images: ndarray) -> ndarray:
    return np.dot(images[..., :3], [0.2989, 0.5870, 0.1140])


def to_histgram(images: ndarray) -> ndarray:
    hog_features = []
    for img in images:
        features = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
        hog_features.append(features)
    return np.array(hog_features)


def to_pca(images: ndarray, n_components: int = 100) -> ndarray:
    pca = PCA(n_components=n_components)
    return pca.fit_transform(images)


def to_norm(features):
    scaler = StandardScaler()
    return scaler.fit_transform(features)


def pipe_from_folder_to_norm(base_folder: str) -> tuple[any, ndarray]:
    i1, labels = load_digits_images(base_folder)
    i2 = to_grayscale(i1)
    i3 = to_histgram(i2)
    i4 = to_pca(i3)
    i5 = to_norm(i4)
    return i5, labels

def pipe_from_image_to_norm(image_filename: str):
    image = Image.open(image_filename)
    image.histogram()
    i1 = np.array(image) / 255.0  # Normalize to [0, 1]
    i2 = to_grayscale(i1)
    i3 = to_histgram(i2)
    i4 = to_pca(i3)
    i5 = to_norm(i4)
    return i5


def _load_csv(csv_filename: str):
    df = pd.read_csv(csv_filename)
    # Remove "Ahegao" class if present
    return df[df['label'] != 'Ahegao']
