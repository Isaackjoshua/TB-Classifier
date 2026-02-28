"""Model training script."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow.keras import Input, Sequential, layers
from tensorflow.keras.optimizers import Adam

from src.config import (
    BATCH_SIZE,
    CLASS_NAMES,
    DATA_DIR,
    DEFAULT_MODEL_ARTIFACT,
    DEFAULT_MODEL_FILE,
    EPOCHS,
    IMG_SIZE,
    LEARNING_RATE,
    MODELS_DIR,
    SEED,
    VALIDATION_SPLIT,
)
from src.preprocessing import create_datasets


def build_model(input_shape: tuple[int, int, int] = (224, 224, 3), num_classes: int = 2) -> tf.keras.Model:
    """Build the CNN architecture from the notebook."""
    model = Sequential(
        [
            Input(shape=input_shape),
            layers.Conv2D(32, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(args: argparse.Namespace) -> dict:
    """Run full training pipeline and persist artifacts."""
    train_ds, val_ds, test_ds, class_names = create_datasets(
        base_dir=args.data_dir,
        image_size=(args.image_height, args.image_width),
        batch_size=args.batch_size,
        class_names=args.class_names,
        seed=args.seed,
        validation_split=args.validation_split,
    )

    model = build_model(
        input_shape=(args.image_height, args.image_width, 3),
        num_classes=len(class_names),
    )

    history = model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=val_ds,
    )

    test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)

    y_pred_probs = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(np.concatenate([labels.numpy() for _, labels in test_ds], axis=0), axis=1)

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    keras_model_path = models_dir / args.keras_model_name
    artifact_path = models_dir / args.artifact_name

    model.save(keras_model_path)

    artifact = {
        "model_path": str(keras_model_path),
        "class_names": class_names,
        "image_size": (args.image_height, args.image_width),
        "batch_size": args.batch_size,
        "seed": args.seed,
        "validation_split": args.validation_split,
        "metrics": {
            "test_loss": float(test_loss),
            "test_accuracy": float(test_accuracy),
        },
        "history": history.history,
    }
    joblib.dump(artifact, artifact_path)

    print(f"Saved Keras model to: {keras_model_path}")
    print(f"Saved artifact metadata to: {artifact_path}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print("\nClassification report:")
    print(report)

    return artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TB chest X-ray classifier")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR), help="Dataset root with class folders")
    parser.add_argument("--models-dir", type=str, default=str(MODELS_DIR), help="Directory to save artifacts")
    parser.add_argument("--artifact-name", type=str, default=DEFAULT_MODEL_ARTIFACT.name, help="Metadata artifact filename")
    parser.add_argument("--keras-model-name", type=str, default=DEFAULT_MODEL_FILE.name, help="Keras model filename")
    parser.add_argument("--image-height", type=int, default=IMG_SIZE[0])
    parser.add_argument("--image-width", type=int, default=IMG_SIZE[1])
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--validation-split", type=float, default=VALIDATION_SPLIT)
    parser.add_argument(
        "--class-names",
        nargs="+",
        default=CLASS_NAMES,
        help="Class folder names in the data directory (e.g., Normal Tuberculosis).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
