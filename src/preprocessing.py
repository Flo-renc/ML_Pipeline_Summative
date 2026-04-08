"""
Preprocessing Module for Handwritten Character Recognition
Handles original and augmented data, preprocessing, and augmentation
"""

import os
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DataPreprocessor:
    def __init__(self, img_size=(64, 64)):
        """
        Initialize the preprocessor.
        Args:
            img_size: tuple, target image size (height, width)
        """
        self.img_size = img_size
        self.class_names = []

    # ---------------------------
    # Load original training data
    # ---------------------------
    def load_images_from_folder(self, folder_path):
        """
        Load images from folder-based dataset.
        Each subfolder is a class.
        Returns:
            images: numpy array (N, H, W, 1)
            labels: numpy array of integers (matching class_names.npy)
            class_names: list of class names
        """
        images, labels = [], []
        folder = Path(folder_path)
        class_folders = sorted([f for f in folder.iterdir() if f.is_dir()])

        if not self.class_names:
        # Load class_names from your clean class_names.npy
            import numpy as np
            self.class_names = np.load("models/class_names.npy", allow_pickle=True).tolist()

        for class_folder in class_folders:
            class_name = class_folder.name
            if class_name not in self.class_names:
                print(f"[WARNING] Skipping unknown class folder: {class_name}")
                continue
            idx = self.class_names.index(class_name)  # label matches the clean class_names.npy

            for img_file in class_folder.glob("*.*"):
                try:
                    img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    img = cv2.resize(img, self.img_size)
                    img = img.astype(np.float32) / 255.0
                    img = np.expand_dims(img, axis=-1)
                    images.append(img)
                    labels.append(idx)
                except Exception as e:
                    print(f"[ERROR] Could not load {img_file}: {e}")

        images = np.array(images, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)

        print(f"Loaded {len(images)} images from {len(class_folders)} folders")
        return images, labels, self.class_names

        
    # ---------------------------
    # Load augmented data from CSV
    # ---------------------------
    def load_images_from_csv(self, csv_path, images_folder):
        import numpy as np
        """
        Load augmented images using a CSV file mapping filenames to labels.
        Returns:
            images: numpy array (N, H, W, 1)
            labels: numpy array of integers
        """
        df = pd.read_csv(csv_path)
        images, labels = [], []

        for idx, row in df.iterrows():
            img_path = os.path.join(images_folder, row['filename'])
            label = row['label']
            if label not in self.class_names:
                self.class_names.append(label)
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, self.img_size)
                img = img.astype(np.float32) / 255.0
                img = np.expand_dims(img, axis=-1)
                images.append(img)
                labels.append(self.class_names.index(label))
            except Exception as e:
                print(f"[ERROR] Could not load {img_path}: {e}")

        images = np.array(images, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)

        print(f"Loaded {len(images)} augmented images from CSV")
        return images, labels

    # ---------------------------
    # Combine original + augmented and split
    # ---------------------------
    def prepare_train_val_data(self, orig_folder, aug_csv=None, aug_folder=None, val_size=0.2, random_state=42):
        """
        Load original and augmented data, combine, and split into train/val sets.
        Returns:
            X_train, X_val, y_train, y_val (ready for CNN)
        """
        X_orig, y_orig, _ = self.load_images_from_folder(orig_folder)

        if aug_csv and aug_folder:
            X_aug, y_aug = self.load_images_from_csv(aug_csv, aug_folder)
            X_all = np.concatenate([X_orig, X_aug], axis=0)
            y_all = np.concatenate([y_orig, y_aug], axis=0)
        else:
            X_all, y_all = X_orig, y_orig

        # One-hot encode labels
        num_classes = len(self.class_names)
        y_all_encoded = to_categorical(y_all, num_classes=num_classes)

        # Stratified train/val split
        X_train, X_val, y_train, y_val = train_test_split(
            X_all, y_all_encoded, test_size=val_size, stratify=y_all, random_state=random_state
        )

        print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        return X_train, X_val, y_train, y_val

    # ---------------------------
    # Data augmentation for handwritten characters
    # ---------------------------
    def get_data_augmenter(self,
                           rotation_range=10,
                           width_shift_range=0.1,
                           height_shift_range=0.1,
                           zoom_range=0.1,
                           shear_range=0.0,   # optional
                           brightness_range=None,
                           horizontal_flip=False):  # avoid flipping letters
        """
        Create ImageDataGenerator with safe parameters for handwriting
        """
        return ImageDataGenerator(
            rotation_range=rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            zoom_range=zoom_range,
            shear_range=shear_range,
            brightness_range=brightness_range,
            horizontal_flip=horizontal_flip,
            fill_mode='nearest'
        )
    
    def process_new_data_for_training(self, data_dir):
        X = []
        y = []

        data_dir = Path(data_dir)

        for label_folder in data_dir.iterdir():
            if not label_folder.is_dir():
                continue

            label = label_folder.name

            for img_path in label_folder.glob("*"):
                try:
                    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

                    if img is None:
                        continue

                    img = cv2.resize(img, (64, 64))
                    img = img / 255.0
                    img = img.reshape(64, 64, 1)

                    X.append(img)
                    y.append(label)

                except Exception as e:
                    print(f"Skipping {img_path}: {e}")

        if len(X) == 0:
            return np.array([]), np.array([])

        X = np.array(X)

        # Load correct class order
        class_names = np.load("models/class_names.npy", allow_pickle=True).tolist()

        label_map = {name: i for i, name in enumerate(class_names)}

        y_encoded = np.array([label_map[label] for label in y if label in label_map])
        return X, y_encoded