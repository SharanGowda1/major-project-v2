"""
inference/predictor.py — Model Loading and Prediction
======================================================
Handles loading the trained model from disk and running
predictions on new leaf images.

Kept separate from train.py so that inference is lightweight —
no need to import the training pipeline just to predict.
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config


def load_model_and_classes() -> tuple:
    """
    Loads the trained model and class names from disk.

    The model is saved in Keras native format (.keras) which includes
    the full architecture + trained weights + optimizer state.

    Returns:
        model        (tf.keras.Model): Loaded trained model
        class_names  (list[str])     : List of 38 disease class name strings

    Raises:
        FileNotFoundError: If the model or class names file doesn't exist.
                           This means training hasn't been run yet.
    """
    if not os.path.exists(config.MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"\n[ERROR] Trained model not found at: {config.MODEL_SAVE_PATH}\n"
            f"Please run training first:\n"
            f"  python train.py\n"
        )

    if not os.path.exists(config.CLASS_NAMES_PATH):
        raise FileNotFoundError(
            f"\n[ERROR] Class names not found at: {config.CLASS_NAMES_PATH}\n"
            f"Please run training first:\n"
            f"  python train.py\n"
        )

    print("[Predictor] Loading model from disk...")
    model = tf.keras.models.load_model(config.MODEL_SAVE_PATH)
    print(f"[Predictor] Model loaded: {config.MODEL_SAVE_PATH}")

    with open(config.CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)
    print(f"[Predictor] {len(class_names)} classes loaded.")

    return model, class_names


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Loads and preprocesses a single image for inference.

    Applies the SAME transformations as the validation pipeline:
      - Resize to IMG_SIZE (128×128)
      - Normalize pixels to [0.0, 1.0]
      - Add batch dimension → shape (1, 128, 128, 3)

    Args:
        image_path (str): Path to any .jpg or .png leaf image

    Returns:
        np.ndarray: Preprocessed image batch, shape (1, 128, 128, 3)

    Raises:
        FileNotFoundError: If the image path does not exist
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"[ERROR] Image not found: {image_path}")

    img = tf.keras.utils.load_img(image_path, target_size=config.IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)   # (128, 128, 3), values [0, 255]
    img_array = img_array / 255.0                   # normalize to [0.0, 1.0]
    img_batch = np.expand_dims(img_array, axis=0)   # add batch dim → (1, 128, 128, 3)
    return img_batch


def predict(
    image_path: str,
    model: tf.keras.Model,
    class_names: list,
) -> list:
    """
    Runs the model on a single image and returns the top-K predictions.

    Args:
        image_path  : Path to the leaf image
        model       : Loaded Keras model
        class_names : List of 38 class name strings

    Returns:
        list of dicts, sorted by confidence descending:
        [
            {"rank": 1, "class": "Tomato___Early_blight", "confidence": 0.872},
            {"rank": 2, "class": "Tomato___Late_blight",  "confidence": 0.091},
            {"rank": 3, "class": "Tomato___healthy",      "confidence": 0.021},
        ]
    """
    img_batch = preprocess_image(image_path)

    # Forward pass — no training, no augmentation
    probabilities = model.predict(img_batch, verbose=0)[0]  # shape: (38,)

    # Get top-K indices sorted by probability descending
    top_k_indices = np.argsort(probabilities)[::-1][: config.TOP_K]

    results = []
    for rank, idx in enumerate(top_k_indices, start=1):
        results.append({
            "rank":       rank,
            "class":      class_names[idx],
            "confidence": float(probabilities[idx]),
        })

    return results


def format_diagnosis_report(
    image_path: str,
    predictions: list,
) -> str:
    """
    Formats the top-K predictions into a human-readable diagnosis report.

    Args:
        image_path  : Path to the analysed image (shown in report header)
        predictions : List of dicts from predict()

    Returns:
        str: Formatted multi-line report string
    """
    top = predictions[0]
    class_name = top["class"]
    confidence = top["confidence"] * 100

    # Determine plant and disease from the class name
    # Format is: "Plant___Disease" or "Plant___healthy"
    parts       = class_name.split("___")
    plant_name  = parts[0].replace("_", " ")
    disease     = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"
    is_healthy  = "healthy" in class_name.lower()
    status      = "HEALTHY ✅" if is_healthy else "DISEASE DETECTED ⚠️"

    lines = [
        "",
        "=" * 58,
        "   🌿  PLANT DISEASE DIAGNOSIS REPORT  🌿",
        "   Built-from-scratch CNN · PlantVillage Dataset",
        "=" * 58,
        f"   Image   : {os.path.basename(image_path)}",
        f"   Status  : {status}",
        "-" * 58,
        f"   Plant   : {plant_name}",
        f"   Disease : {disease}",
        f"   Confidence : {confidence:.1f}%",
        "-" * 58,
        f"   Top {config.TOP_K} Predictions:",
    ]

    for p in predictions:
        bar_len = int(p["confidence"] * 20)
        bar     = "█" * bar_len + "░" * (20 - bar_len)
        name    = p["class"].replace("___", " — ").replace("_", " ")
        pct     = p["confidence"] * 100
        lines.append(f"   [{p['rank']}] {bar} {pct:5.1f}%  {name}")

    lines += ["=" * 58, ""]
    return "\n".join(lines)
