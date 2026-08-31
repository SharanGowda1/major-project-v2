"""
train.py — Training Entry Point
=================================
Run this file to execute the full training pipeline from start to finish.

Usage:
    python train.py

What it does (in order):
    1. Load and verify dataset (54,305 images, 38 classes)
    2. Build the custom CNN from scratch (no pre-trained weights)
    3. Print architecture summary
    4. Compile the model (Adam + CategoricalCrossEntropy)
    5. Train with smart callbacks (checkpoint, early stopping, LR reduce)
    6. Evaluate on validation set
    7. Generate training curves and per-class accuracy plots
    8. Run full classification report + confusion matrix
    9. Generate Grad-CAM on one sample image from each predicted class

All outputs saved to: plant-disease-cnn/outputs/
Best model saved to:  plant-disease-cnn/saved_models/plant_disease_cnn_best.keras
"""

import os
import sys
import numpy as np

# Make sure all modules resolve correctly regardless of where this is run from
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import config
from data.loader import load_datasets, get_dataset_info
from model.network import build_model, print_model_summary
from training.trainer import compile_model, train, evaluate
from evaluation.metrics import get_true_and_predicted, run_full_evaluation
from visualization.plots import generate_all_plots


def main():
    print("\n" + "=" * 58)
    print("  PLANT DISEASE CNN — TRAINING PIPELINE")
    print("  Custom Architecture · Built from Scratch")
    print("  No pre-trained weights · No transfer learning")
    print("=" * 58 + "\n")

    # ----------------------------------------------------------------
    # STEP 1: Load Dataset
    # ----------------------------------------------------------------
    print("[Step 1/5] Loading dataset...")
    train_ds, val_ds, class_names = load_datasets()
    get_dataset_info(class_names)

    # ----------------------------------------------------------------
    # STEP 2: Build Model
    # ----------------------------------------------------------------
    print("[Step 2/5] Building CNN architecture from scratch...")
    model = build_model(num_classes=len(class_names))
    print_model_summary(model)

    # ----------------------------------------------------------------
    # STEP 3: Compile
    # ----------------------------------------------------------------
    print("[Step 3/5] Compiling model (Adam + CategoricalCrossEntropy)...")
    model = compile_model(model)

    # ----------------------------------------------------------------
    # STEP 4: Train
    # ----------------------------------------------------------------
    print("[Step 4/5] Starting training...")
    history = train(model, train_ds, val_ds)

    # ----------------------------------------------------------------
    # STEP 5: Evaluate + Visualize
    # ----------------------------------------------------------------
    print("[Step 5/5] Evaluating and generating outputs...")

    # Final validation metrics
    evaluate(model, val_ds)

    # Collect all predictions for plotting and metrics
    y_true, y_pred = get_true_and_predicted(model, val_ds)

    # Training curves + per-class accuracy bar chart
    generate_all_plots(history, y_true, y_pred, class_names)

    # Full classification report + 38×38 confusion matrix PNG
    run_full_evaluation(model, val_ds, class_names)

    print("\n" + "=" * 58)
    print("  TRAINING COMPLETE")
    print(f"  Model : {config.MODEL_SAVE_PATH}")
    print(f"  Plots : {config.OUTPUTS_DIR}/")
    print("=" * 58 + "\n")
    print("Next step: run  python predict.py --image <path_to_leaf.jpg>\n")


if __name__ == "__main__":
    main()
