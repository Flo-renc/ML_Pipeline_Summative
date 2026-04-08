"""
Handwritten Character Recognition System
Source package initialization
"""

__version__ = "1.0.0"
__author__ = "Florence Kabeya"

# Preprocessing
from .preprocessing import DataPreprocessor

# Model
from .model import(
    build_character_recognition_model,
    build_lightweight_model,
    build_deep_model
)

# Prediction
from .prediction import CharacterPredictor

__all__ = [
    'DataPreprocessor',
    'build_character_recognition_model',
    'build_lightweight_model',
    'build_deep_model',
    'CharacterPredictor',
    'predict_from_file',
    'batch_predict_from_folder'
]
