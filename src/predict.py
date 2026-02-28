"""Inference helpers for trained TB classifier."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import tensorflow as tf
from PIL import Image

from src.config import DEFAULT_MODEL_ARTIFACT
from src.preprocessing import preprocess_image_path, preprocess_uploaded_image


def load_artifacts(artifact_path: str | Path = DEFAULT_MODEL_ARTIFACT) -> dict[str, Any]:
    """Load metadata artifact and model."""
    artifact = joblib.load(artifact_path)
    model_path = Path(artifact["model_path"])

    if not model_path.is_absolute():
        model_path = Path(artifact_path).resolve().parent / model_path

    artifact["model"] = tf.keras.models.load_model(model_path)
    return artifact


def predict_from_array(image_array: np.ndarray, artifact: dict[str, Any]) -> dict[str, Any]:
    """Predict class probabilities from preprocessed image array."""
    model = artifact["model"]
    class_names = artifact["class_names"]

    probabilities = model.predict(image_array, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))

    return {
        "predicted_index": predicted_index,
        "predicted_class": class_names[predicted_index],
        "confidence": float(probabilities[predicted_index]),
        "probabilities": {class_names[i]: float(probabilities[i]) for i in range(len(class_names))},
    }


def predict_from_image_path(image_path: str | Path, artifact_path: str | Path = DEFAULT_MODEL_ARTIFACT) -> dict[str, Any]:
    """Load image from disk and return prediction."""
    artifact = load_artifacts(artifact_path)
    image_size = tuple(artifact["image_size"])
    image_array = preprocess_image_path(image_path=image_path, image_size=image_size)
    return predict_from_array(image_array=image_array, artifact=artifact)


def predict_from_pil_image(image: Image.Image, artifact: dict[str, Any]) -> dict[str, Any]:
    """Predict directly from PIL image using already loaded artifact."""
    image_size = tuple(artifact["image_size"])
    image_array = preprocess_uploaded_image(image=image, image_size=image_size)
    return predict_from_array(image_array=image_array, artifact=artifact)
