"""
Test predictions for Handwritten Character Recognition (grayscale, 64x64x1)

"""

from pathlib import Path
from src.prediction import CharacterPredictor

# -----------------------------
# CONFIGURATION
# -----------------------------
MODEL_PATH = Path("models/final_alphanumeric_model.h5")  # your trained RGB model
CLASS_NAMES_PATH = Path("models/class_names.npy")          # class names from training
TEST_IMAGES_DIR = Path("notebook")                        # directory with test images
SAMPLE_IMAGE = TEST_IMAGES_DIR / "z.042.png"              # single test image

# -----------------------------
# Initialize Predictor
# -----------------------------
predictor = CharacterPredictor(
    model_path=MODEL_PATH,
    class_names_path=CLASS_NAMES_PATH,
    img_size=(64, 64)
)






# -----------------------------
# Single Image Prediction
# -----------------------------
if SAMPLE_IMAGE.exists():
    print("Making prediction for single image...")
    result = predictor.predict(SAMPLE_IMAGE)
    print("\nSingle Prediction Result:")
    print(f"Predicted Character: {result['predicted_character']}")
    print(f"Confidence: {result['confidence_percent']:.2f}%")
    print("Top 3 Predictions:")
    for i, pred in enumerate(result['top_k_predictions'], 1):
        print(f"  {i}. {pred['character']}: {pred['confidence_percent']:.2f}%")
else:
    print(f"No sample image found at {SAMPLE_IMAGE}, skipping single prediction.")

# -----------------------------
# Batch Prediction
# -----------------------------
import glob

image_paths = glob.glob(str(TEST_IMAGES_DIR / "*.png")) + \
              glob.glob(str(TEST_IMAGES_DIR / "*.jpg")) + \
              glob.glob(str(TEST_IMAGES_DIR / "*.jpeg"))

if image_paths:
    print("\nMaking batch predictions...")
    batch_results = predictor.predict_batch(image_paths)
    print("\nBatch Prediction Results:")
    for res in batch_results:
        print(f"{res['image_path']}: {res['predicted_character']} ({res['confidence_percent']:.2f}%)")
else:
    print(f"No image files found in {TEST_IMAGES_DIR}, skipping batch prediction.")

# -----------------------------
# Model Info
# -----------------------------
info = predictor.get_model_info()
print("\nModel Info:")
for k, v in info.items():
    print(f"{k}: {v}")
