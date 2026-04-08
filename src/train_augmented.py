"""
Training Script for Alphanumeric Handwritten Character Recognition
Uses both original and augmented datasets with safe preprocessing and augmentation
"""

import os
from src.preprocessing import DataPreprocessor
from src.model import build_character_recognition_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ---------------------------
# 1. Paths
# ---------------------------
ORIG_TRAIN_FOLDER = "data/train"            # original train images
AUGMENTED_FOLDER = "data/augmented_images"        # folder containing augmented images
AUGMENTED_CSV = "data/image_labels.csv"  # CSV mapping augmented images to labels

# ---------------------------
# 2. Initialize Preprocessor
# ---------------------------
preprocessor = DataPreprocessor(img_size=(64,64))

# ---------------------------
# 3. Load and prepare train/val data
# ---------------------------
X_train, X_val, y_train, y_val = preprocessor.prepare_train_val_data(
    orig_folder=ORIG_TRAIN_FOLDER,
    aug_csv=AUGMENTED_CSV,
    aug_folder=AUGMENTED_FOLDER,
    val_size=0.2
)

# ---------------------------
# 4. Data augmentation
# ---------------------------
datagen = preprocessor.get_data_augmenter(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False  # flipping letters may create wrong labels
)
datagen.fit(X_train)

# ---------------------------
# 5. Build model
# ---------------------------
num_classes = y_train.shape[1]
input_shape = X_train.shape[1:]  # (64,64,1)
model = build_character_recognition_model(num_classes=num_classes, input_shape=input_shape)
model.summary()

# ---------------------------
# 6. Define callbacks
# ---------------------------
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1),
    ModelCheckpoint("best_alphanumeric_model.h5", monitor='val_accuracy', save_best_only=True, verbose=1)
]

# ---------------------------
# 7. Train model
# ---------------------------
batch_size = 32
epochs = 50

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=batch_size),
    steps_per_epoch=len(X_train)//batch_size,
    validation_data=(X_val, y_val),
    epochs=epochs,
    callbacks=callbacks
)

# ---------------------------
# 8. Save final model
# ---------------------------
model.save("final_alphanumeric_model.h5")
print("Training complete. Model saved as 'final_alphanumeric_model.h5'")
