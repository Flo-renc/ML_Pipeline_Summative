# retrain_model.py
"""
Retraining script for Alphanumeric Handwritten Character Recognition
"""

import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.utils import to_categorical
from src.model import build_character_recognition_model
import os

# -----------------------------
# Configuration
# -----------------------------
DATA_DIR = "data/retrain"      # folder containing class subfolders
BATCH_SIZE = 32
IMG_SIZE = (64, 64)            # matches your main CNN model input
EPOCHS = 10                    # adjust as needed
NUM_CLASSES = len(os.listdir(DATA_DIR))  # auto count classes

# -----------------------------
# Load dataset
# -----------------------------
train_ds = image_dataset_from_directory(
    DATA_DIR,
    image_size=IMG_SIZE,
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    label_mode='categorical',  # for one-hot labels
    shuffle=True,
    seed=42
)

# Normalize images to [0,1]
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

# Prefetch for performance
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# -----------------------------
# Build the model
# -----------------------------
model = build_character_recognition_model(num_classes=NUM_CLASSES, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1))
model.summary()

# -----------------------------
# Retrain the model
# -----------------------------
history = model.fit(
    train_ds,
    epochs=EPOCHS
)

# -----------------------------
# Save the retrained model
# -----------------------------
MODEL_SAVE_PATH = "./models/final_alphanumeric_model.h5"
model.save(MODEL_SAVE_PATH, save_format='h5')
print(f"Retrained model saved to {MODEL_SAVE_PATH}")