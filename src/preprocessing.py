"""Shared preprocessing utilities for training and inference."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing import image_dataset_from_directory

from src.config import BATCH_SIZE, CLASS_NAMES, IMG_SIZE, SEED, VALIDATION_SPLIT


AUTOTUNE = tf.data.AUTOTUNE


def create_datasets(
    base_dir: str | Path,
    image_size: Tuple[int, int] = IMG_SIZE,
    batch_size: int = BATCH_SIZE,
    class_names: list[str] = CLASS_NAMES,
    seed: int = SEED,
    validation_split: float = VALIDATION_SPLIT,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Create train/validation/test datasets with deterministic splits.

    Split logic matches the original notebook:
    1. 80% training
    2. Remaining 20% split equally into validation and test
    """
    base_dir = Path(base_dir)

    train_ds = image_dataset_from_directory(
        base_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=class_names,
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=seed,
        validation_split=validation_split,
        subset="training",
        verbose=True,
    )

    val_test_ds = image_dataset_from_directory(
        base_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=class_names,
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False,
        seed=seed,
        validation_split=validation_split,
        subset="validation",
        verbose=True,
    )

    val_batches = tf.data.experimental.cardinality(val_test_ds)
    test_ds = val_test_ds.take(val_batches // 2)
    val_ds = val_test_ds.skip(val_batches // 2)

    return (
        train_ds.prefetch(AUTOTUNE),
        val_ds.prefetch(AUTOTUNE),
        test_ds.prefetch(AUTOTUNE),
    )


def preprocess_uploaded_image(image: Image.Image, image_size: Tuple[int, int] = IMG_SIZE) -> np.ndarray:
    """Resize and format an uploaded image for model inference."""
    rgb_image = image.convert("RGB").resize(image_size)
    image_array = np.asarray(rgb_image, dtype=np.float32)
    return np.expand_dims(image_array, axis=0)


def preprocess_image_path(image_path: str | Path, image_size: Tuple[int, int] = IMG_SIZE) -> np.ndarray:
    """Load an image from disk and preprocess it for model inference."""
    image = Image.open(image_path)
    return preprocess_uploaded_image(image=image, image_size=image_size)
