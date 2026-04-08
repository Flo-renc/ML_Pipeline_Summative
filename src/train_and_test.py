import sys
from src.preprocessing import DataPreprocessor
from src.model import CharacterRecognitionModel
from src.prediction import CharacterPredictor
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path

# -----------------------------
# STEP 1: Load and preprocess data
# -----------------------------
pre = DataPreprocessor(img_size=(32, 32))
X, y, class_names = pre.load_images_from_folder("data/train")

print(f"Loaded {len(X)} images from {len(class_names)} classes")

# Preprocess images
X = pre.preprocess_images(X)

# Convert labels to one-hot
num_classes = len(class_names)
y_onehot = to_categorical(y, num_classes=num_classes)

# Save class names for prediction later
Path("models").mkdir(exist_ok=True)
np.save("models/class_names.npy", np.array(class_names))
print("Saved class_names.npy!")

# Split into train/val (90/10)
split_idx = int(0.9 * len(X))
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y_onehot[:split_idx], y_onehot[split_idx:]

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1
)

history = model_obj.train(
    X_train, y_train,
    X_val, y_val,
    batch_size=32,
    epochs=50,
    datagen=datagen,
    model_save_path='models/best_model.keras'
)

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")

# -----------------------------
# STEP 2: Initialize model
# -----------------------------
model_obj = CharacterRecognitionModel(input_shape=(32, 32, 1), num_classes=num_classes)

# Build Sequential CNN
model = model_obj.build_sequential_cnn()
print(f"Model built with {model.count_params():,} parameters")

# Compile model
model_obj.compile_model(learning_rate=0.001)

# -----------------------------
# STEP 3: Train model
# -----------------------------
history = model_obj.train(
    X_train, y_train,
    X_val, y_val,
    batch_size=32,
    epochs=50  # Train properly
)

# Save final model
model_path = "models/handwritten_cnn_model.keras"
model_obj.save_model(model_path)

# -----------------------------
# STEP 4: Test prediction on sample.png
# -----------------------------
sample_image = "data/test/sample.png"
predictor = CharacterPredictor(model_path, metadata_path=None, img_size=(32, 32))

result = predictor.predict(sample_image)
print("\nSingle Prediction Result:")
print(f"Predicted Character: {result['predicted_character']}")
print(f"Confidence: {result['confidence_percent']:.2f}%")
print("Top 3 Predictions:")
for i, pred in enumerate(result['top_3_predictions'], 1):
    print(f"  {i}. {pred['character']}: {pred['confidence_percent']:.2f}%")

# -----------------------------
# STEP 5: Model Info
# -----------------------------
info = predictor.get_model_info()
print("\nModel Info")
print(info)
