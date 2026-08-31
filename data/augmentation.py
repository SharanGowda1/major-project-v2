"""
data/augmentation.py — Data Augmentation Pipeline
===================================================
Builds a Sequential stack of augmentation layers applied ONLY to
training data. Validation data is NEVER augmented (only rescaled).

WHY AUGMENTATION?
-----------------
Our CNN must detect disease on leaves photographed in the real world —
different angles, lighting, orientations. Augmentation artificially
creates those variations during training so the model generalizes better
and doesn't memorize the exact training images (overfitting).

Each layer here is a deliberate choice explained in its comment.
"""

import tensorflow as tf
import sys
import os

# Allow imports from the project root (plant-disease-cnn/)
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config


def build_augmentation_layer() -> tf.keras.Sequential:
    """
    Returns a Keras Sequential model containing all augmentation layers.

    This is used as the FIRST layer inside the training pipeline.
    At inference time, this layer is NOT called — only rescaling is applied.

    Returns:
        tf.keras.Sequential: Augmentation + rescaling pipeline for training.
    """
    augmentation = tf.keras.Sequential(
        [
            # --- RESCALING ---
            # Converts pixel values from [0, 255] → [0.0, 1.0]
            # Neural networks train much faster with normalized inputs.
            # MUST be the first transformation.
            tf.keras.layers.Rescaling(1.0 / 255),

            # --- RANDOM FLIP ---
            # Randomly mirrors images horizontally and vertically.
            # WHY: A diseased leaf looks the same upside-down or mirrored.
            # This doubles/quadruples the effective number of training samples.
            tf.keras.layers.RandomFlip(
                config.AUGMENTATION["random_flip"]
            ),

            # --- RANDOM ROTATION ---
            # Rotates the image by up to ±(factor × 360°) degrees.
            # factor=0.2 means up to ±72° rotation.
            # WHY: Photos of leaves are taken at random angles in the field.
            tf.keras.layers.RandomRotation(
                config.AUGMENTATION["random_rotation"],
                fill_mode="reflect",  # fills empty corners with reflected pixels
            ),

            # --- RANDOM ZOOM ---
            # Randomly zooms in or out by up to 10%.
            # WHY: Camera distance varies in real-world photography.
            tf.keras.layers.RandomZoom(
                config.AUGMENTATION["random_zoom"],
                fill_mode="reflect",
            ),

            # --- RANDOM BRIGHTNESS ---
            # Shifts pixel brightness by up to ±10%.
            # WHY: Lighting conditions vary (cloudy, sunny, indoors, outdoors).
            tf.keras.layers.RandomBrightness(
                config.AUGMENTATION["random_brightness"],
                value_range=(0.0, 1.0),  # clamp values to [0,1] after shift
            ),
        ],
        name="augmentation_pipeline",
    )

    return augmentation


def build_rescaling_layer() -> tf.keras.Sequential:
    """
    Returns ONLY the rescaling layer — used for validation/test data.
    Validation data must NOT be augmented, only normalized.

    Returns:
        tf.keras.Sequential: Rescaling-only pipeline for validation/inference.
    """
    rescaling = tf.keras.Sequential(
        [
            tf.keras.layers.Rescaling(1.0 / 255),
        ],
        name="rescaling_only",
    )

    return rescaling
