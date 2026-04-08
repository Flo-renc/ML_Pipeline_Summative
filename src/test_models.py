"""
train_models.py
Trains a CNN for handwritten character recognition and saves the model and class names.
"""

import numpy as np
from pathlib import Path
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from src.preprocessing import DataPreprocessor
from src.model import CharacterRecognitionModel

# -----------------------------
# Paths
# -----------------------------
DATA_DIR = "data/train"
MODEL_DIR = "models"
MODEL_PATH = Path(MODEL_DIR) / "best_model.keras"
CLASS_NAMES_PATH = Path(MODEL_DIR) / "class_names.npy"
METADATA_PATH = Path(MODEL_DIR) / "metadata.json"

# Ensure model directory exists
Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load and preprocess data
# -----------------------------
print("Loading training images...")
pre = DataPreprocessor(img_size=(32, 32))
X, y, class_names = pre.load_images_from_folder(DATA_DIR)
X = pre.preprocess_images(X)

num_classes = len(class_names)
y_onehot = to_categorical(y, num_classes=num_classes)

print(f"Loaded {len(X)} images from {num_classes} classes")
print(f"X shape: {X.shape}, y shape: {y_onehot.shape}")

# -----------------------------
# Create data augmenter
# -----------------------------
datagen = pre.get_data_augmenter(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2
)

# -----------------------------
# Initialize and compile model
# -----------------------------
model_obj = CharacterRecognitionModel(input_shape=(32, 32, 1), num_classes=num_classes)
model_obj.build_cnn()  # You can also use build_functional_cnn()
model_obj.compile_model(learning_rate=0.0005)  # Lower LR for small dataset

# -----------------------------
# Callbacks
# -----------------------------
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]

# -----------------------------
# Train model with augmentation
# -----------------------------
print("Training model with data augmentation...")
history = model_obj.model.fit(
    datagen.flow(X, y_onehot, batch_size=32),
    steps_per_epoch=len(X)//32,
    validation_split=0.2,
    epochs=50,
    callbacks=callbacks,
    verbose=1
)

# -----------------------------
# Save class names
# -----------------------------
np.save(CLASS_NAMES_PATH, np.array(class_names))
print(f"Saved {len(class_names)} class names to {CLASS_NAMES_PATH}")

# -----------------------------
# Save training metadata
# -----------------------------
eval_results = model_obj.evaluate(X, np.argmax(y_onehot, axis=1), class_names=class_names)
model_obj.save_model_metadata(METADATA_PATH, class_names, eval_results)

print(f"Training completed! Model saved at {MODEL_PATH}")
