"""Central configuration for training and inference."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Paths
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_MODEL_ARTIFACT = MODELS_DIR / "model.joblib"
DEFAULT_MODEL_FILE = MODELS_DIR / "tb_cnn_classifier.keras"

# Data and model settings
CLASS_NAMES = ["Normal", "Tuberculosis"]
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 123
VALIDATION_SPLIT = 0.2

# Training defaults
EPOCHS = 50
LEARNING_RATE = 1e-4
