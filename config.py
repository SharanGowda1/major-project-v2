"""
config.py — Central configuration for Plant Disease CNN
========================================================
All hyperparameters and paths live here.
Change values here ONLY — never hardcode them in other files.
"""

import os

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

# Root of this project (the plant-disease-cnn/ folder)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Where your dataset lives.
# ── LOCAL (default) ────────────────────────────────────────
DATA_DIR = os.path.join(PROJECT_ROOT, "raw", "color")
# ── KAGGLE (switch to this when running on Kaggle) ─────────
# DATA_DIR = "/kaggle/input/plantvillage-dataset/color"

# Where trained model weights will be saved
MODELS_DIR = os.path.join(PROJECT_ROOT, "saved_models")
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "plant_disease_cnn_best.keras")

# Where class names JSON will be saved (needed for inference)
CLASS_NAMES_PATH = os.path.join(MODELS_DIR, "class_names.json")

# Where output plots and Grad-CAM images will be saved
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

# ---------------------------------------------------------------------------
# IMAGE & DATA SETTINGS
# ---------------------------------------------------------------------------

# Image size fed into the CNN
# 128x128: good balance of speed vs accuracy for CPU-only training
IMG_SIZE = (128, 128)
IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 3  # RGB

# Number of disease classes in PlantVillage dataset
NUM_CLASSES = 38

# ---------------------------------------------------------------------------
# TRAINING HYPERPARAMETERS
# ---------------------------------------------------------------------------

# Batch size: number of images processed per gradient update
# 16 is safe for WSL with integrated GPU — avoids OOM crashes
BATCH_SIZE = 16

# Maximum number of training epochs
# EarlyStopping will stop sooner if val_accuracy stops improving
EPOCHS = 25

# Initial learning rate for Adam optimizer
# Adam adapts this per-parameter during training
LEARNING_RATE = 1e-3

# Train/Validation split ratio (0.2 = 20% goes to validation)
VALIDATION_SPLIT = 0.2

# Random seed — keeps train/val split consistent across runs
SEED = 42

# ---------------------------------------------------------------------------
# AUGMENTATION SETTINGS
# ---------------------------------------------------------------------------

# Applied only to training data to prevent overfitting
AUGMENTATION = {
    "random_flip": "horizontal_and_vertical",  # leaves have no fixed orientation
    "random_rotation": 0.2,                    # +-20% of 360 degrees = +-72 degrees
    "random_zoom": 0.1,                        # +-10% zoom
    "random_brightness": 0.1,                  # +-10% brightness shift
}

# ---------------------------------------------------------------------------
# MODEL ARCHITECTURE SETTINGS
# ---------------------------------------------------------------------------

# Dropout rate after the Dense(512) layer
# 0.5 = randomly turns off 50% of neurons during training
DROPOUT_RATE = 0.5

# Number of units in the fully connected layer before the output
FC_UNITS = 512

# ---------------------------------------------------------------------------
# CALLBACKS / TRAINING CONTROL
# ---------------------------------------------------------------------------

# EarlyStopping: stop training if val_accuracy doesn't improve for N epochs
EARLY_STOPPING_PATIENCE = 5

# ReduceLROnPlateau: halve learning rate if val_loss doesn't improve for N epochs
LR_REDUCE_PATIENCE = 3
LR_REDUCE_FACTOR = 0.5
LR_MIN = 1e-6  # Never let LR go below this

# ---------------------------------------------------------------------------
# INFERENCE SETTINGS
# ---------------------------------------------------------------------------

# Number of top predictions to display
TOP_K = 3
