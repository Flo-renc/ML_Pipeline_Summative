"""
Prediction Module for Handwritten Character Recognition
Now correctly aligned with training on 64×64 grayscale images.
"""

import numpy as np
import cv2
from pathlib import Path
from tensorflow import keras

class CharacterPredictor:
    """Predict characters using a trained CNN model."""

    def __init__(self, model=None, model_path=None, class_names_path="models/class_names.npy", img_size=(64, 64)):
        self.img_size = img_size      # >>> FIXED to 64x64
        self.model = model
        self.class_names = []
        if self.model is None and model_path:
            self.load_model(model_path)

        self.load_class_names(class_names_path)

    # -------------------------------
    # Model Loading
    # -------------------------------
    def load_model(self, model_path):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        print(f"Loading model from {model_path}...")
        self.model = keras.models.load_model(str(model_path), compile=False, safe_mode=False)
        print(f"Model loaded. Input shape: {self.model.input_shape}")

    def load_class_names(self, class_names_path):
        class_names_path = Path(class_names_path)
        if not class_names_path.exists():
            raise FileNotFoundError(f"class_names.npy not found: {class_names_path}")

        self.class_names = np.load(class_names_path, allow_pickle=True).tolist()
        print(f"Loaded {len(self.class_names)} class names")

    # -------------------------------
    # Preprocessing
    # -------------------------------
    def preprocess_image(self, image_input):
        """
        Preprocess:
        - Read image
        - Convert to grayscale
        - Resize to 64×64
        - Normalize
        - Return shape: (1, 64, 64, 1)
        """

        # Read image
        if isinstance(image_input, (str, Path)):
            img = cv2.imread(str(image_input), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Could not read image: {image_input}")
        elif isinstance(image_input, np.ndarray):
            img = image_input.copy()

            # Ensure grayscale
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("image_input must be a file path or numpy array")

        # Resize
        img = cv2.resize(img, self.img_size)

        # Normalize
        img = img.astype("float32") / 255.0

        # Correct shape
        img = img.reshape(1, self.img_size[0], self.img_size[1], 1)
        return img

    # -------------------------------
    # Single Prediction
    # -------------------------------
    def predict(self, image_input, top_k=3):
        img = self.preprocess_image(image_input)
        predictions = self.model.predict(img, verbose=0)[0]

        top_k = min(top_k, len(predictions))
        top_k_idx = np.argsort(predictions)[-top_k:][::-1]

        top_k_predictions = [
            {
                "character": self.class_names[i],
                "confidence": float(predictions[i]),
                "confidence_percent": float(predictions[i] * 100),
            }
            for i in top_k_idx
        ]

        best_idx = top_k_idx[0]

        return {
            "predicted_character": self.class_names[best_idx],
            "confidence": float(predictions[best_idx]),
            "confidence_percent": float(predictions[best_idx] * 100),
            "top_k_predictions": top_k_predictions,
        }

    # -------------------------------
    # Batch Prediction
    # -------------------------------
    def predict_batch(self, image_paths):
        batch_images = []
        valid_paths = []

        for p in image_paths:
            try:
                img = self.preprocess_image(p)
                batch_images.append(img[0])   # remove batch dimension
                valid_paths.append(str(p))
            except Exception as e:
                print(f"Error processing {p}: {e}")

        if not batch_images:
            return []

        batch_images = np.array(batch_images)
        predictions = self.model.predict(batch_images, verbose=0)

        results = []

        for i, pred in enumerate(predictions):
            idx = int(np.argmax(pred))
            results.append({
                "image_name": Path(valid_paths[i]).name,
                "predicted_character": self.class_names[idx],
                "confidence": float(pred[idx]),
                "confidence_percent": float(pred[idx] * 100),
            })

        return results

    # -------------------------------
    # Model Info
    # -------------------------------
    def get_model_info(self):
        return {
            "model_loaded": self.model is not None,
            "input_shape": list(self.model.input_shape) if self.model else None,
            "output_shape": list(self.model.output_shape) if self.model else None,
            "num_classes": len(self.class_names),
            "class_names_sample": self.class_names[:10],
        }


# ---------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------
def predict_from_file(model_path, image_path, class_names_path="models/class_names.npy"):
    predictor = CharacterPredictor(model_path, class_names_path, img_size=(64, 64))
    result = predictor.predict(image_path)

    print(f"\nPredicted Character: {result['predicted_character']}")
    print(f"Confidence: {result['confidence_percent']:.2f}%")
    print("\nTop Predictions:")
    for r in result["top_k_predictions"]:
        print(f"  {r['character']} — {r['confidence_percent']:.2f}%")

    return result
