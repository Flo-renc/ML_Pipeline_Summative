"""
train_models.py
Trains a CNN for handwritten character recognition and saves the model, class names, and metadata.
"""

import numpy as np
from pathlib import Path
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from src.preprocessing import DataPreprocessor
from src.model import CharacterRecognitionModel
import json

# -----------------------------
# Paths
# -----------------------------
TRAIN_DIR = "data/train"
TEST_DIR = "data/test"
MODEL_DIR = "models"
MODEL_PATH = Path(MODEL_DIR) / "best_model.keras"
CLASS_NAMES_PATH = Path(MODEL_DIR) / "class_names.npy"
METADATA_PATH = Path(MODEL_DIR) / "metadata.json"

Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load and preprocess data
# -----------------------------
print("Loading and preprocessing data...")
data = DataPreprocessor(img_size=(32, 32))
X, y, class_names = data.load_images_from_folder(TRAIN_DIR)
X = data.preprocess_images(X)
num_classes = len(class_names)
y_onehot = to_categorical(y, num_classes=num_classes)

# Split train/validation
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    X, y_onehot, test_size=0.2, random_state=42, stratify=y
)
print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

# Load test data
X_test, y_test, _ = data.load_images_from_folder(TEST_DIR)
X_test = data.preprocess_images(X_test)

# -----------------------------
# Data augmentation
# -----------------------------
datagen = data.get_data_augmenter(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    brightness_range=(0.8, 1.2),
    horizontal_flip=True
)
datagen.fit(X_train)

# -----------------------------
# Initialize and compile model
# -----------------------------
model_obj = CharacterRecognitionModel(input_shape=(32, 32, 1), num_classes=num_classes)
model_obj.build_cnn()
model_obj.compile_model(learning_rate=0.001)

# -----------------------------
# Callbacks
# -----------------------------
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
]

# -----------------------------
# Train model with augmentation
# -----------------------------
print("Training model with data augmentation...")
history = model_obj.model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    steps_per_epoch=len(X_train) // 32,
    validation_data=(X_val, y_val),
    epochs=50,
    callbacks=callbacks,
    verbose=1
)

# -----------------------------
# Evaluate on test data
# -----------------------------
from sklearn.metrics import accuracy_score
y_test_labels = np.argmax(to_categorical(y_test, num_classes=num_classes), axis=1)
y_pred = np.argmax(model_obj.model.predict(X_test), axis=1)
test_acc = accuracy_score(y_test_labels, y_pred)
print(f"\nTest Accuracy: {test_acc:.4f}")

# -----------------------------
# Save class names
# -----------------------------
np.save(CLASS_NAMES_PATH, np.array(class_names))
print(f"Saved {len(class_names)} class names to {CLASS_NAMES_PATH}")

# -----------------------------
# Save metadata
# -----------------------------
metadata = {
    'model_architecture': 'CNN',
    'input_shape': list(X_train.shape[1:]),
    'num_classes': num_classes,
    'class_names': class_names,
    'total_parameters': int(model_obj.model.count_params()),
    'test_accuracy': float(test_acc),
}
with open(METADATA_PATH, 'w') as f:
    json.dump(metadata, f, indent=4)
print(f"Saved metadata to {METADATA_PATH}")

print(f"\nTraining completed! Model saved at {MODEL_PATH}")
