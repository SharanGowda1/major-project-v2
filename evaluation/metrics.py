"""
evaluation/metrics.py — Model Evaluation & Performance Analysis
================================================================
After training, we need to understand HOW WELL the model performs
across all 38 disease classes — not just overall accuracy.

WHY PER-CLASS METRICS?
-----------------------
Overall accuracy can be misleading. If 80% of images are "Tomato healthy"
and the model just predicts that always, it gets 80% accuracy but is useless.

Per-class metrics tell the truth:
  - Precision : Of all images predicted as "Apple Scab", how many truly were?
  - Recall    : Of all real "Apple Scab" images, how many did we catch?
  - F1 Score  : Harmonic mean of precision and recall (best single metric)
  - Confusion Matrix: Full 38×38 grid showing every prediction vs true label
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for WSL/headless
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config


def get_true_and_predicted(
    model: tf.keras.Model,
    val_ds: tf.data.Dataset,
) -> tuple:
    """
    Runs inference on the entire validation dataset to collect
    true labels and predicted labels for metric computation.

    Args:
        model  : Trained Keras model
        val_ds : Validation tf.data.Dataset (unbatched labels are one-hot)

    Returns:
        y_true (np.ndarray): Integer true class indices, shape (N,)
        y_pred (np.ndarray): Integer predicted class indices, shape (N,)
    """
    y_true_list = []
    y_pred_list = []

    print("[Metrics] Running inference on validation set...")
    for images, labels in val_ds:
        # labels are one-hot encoded → convert to integer class index
        true_indices = np.argmax(labels.numpy(), axis=1)
        y_true_list.extend(true_indices)

        # model output is softmax probabilities → argmax gives predicted class
        preds = model.predict(images, verbose=0)
        pred_indices = np.argmax(preds, axis=1)
        y_pred_list.extend(pred_indices)

    return np.array(y_true_list), np.array(y_pred_list)


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
) -> None:
    """
    Prints a full classification report with per-class precision,
    recall, and F1 score for all 38 disease classes.

    Args:
        y_true      : Integer array of true class indices
        y_pred      : Integer array of predicted class indices
        class_names : List of 38 class name strings
    """
    print("\n" + "=" * 65)
    print("  PER-CLASS CLASSIFICATION REPORT")
    print("=" * 65)
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=3,
    )
    print(report)
    print("=" * 65 + "\n")

    # Save report to text file
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    report_path = os.path.join(config.OUTPUTS_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"[Metrics] Report saved to: {report_path}")


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
) -> None:
    """
    Generates and saves a 38×38 confusion matrix heatmap as a PNG.

    The matrix shows:
      - Rows    = True class
      - Columns = Predicted class
      - Diagonal = Correct predictions (bright)
      - Off-diag = Misclassifications (which class was confused with which)

    Args:
        y_true      : Integer array of true class indices
        y_pred      : Integer array of predicted class indices
        class_names : List of 38 class name strings
    """
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)

    # Normalize each row so values are 0.0–1.0 (proportion per true class)
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(20, 18))
    im = ax.imshow(cm_normalized, interpolation="nearest", cmap="Blues")

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    ax.set_title(
        "Confusion Matrix — Plant Disease CNN (From Scratch)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    # Tick labels: shortened names to fit
    short_names = [n.replace("___", "\n").replace("_", " ") for n in class_names]
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(short_names, rotation=90, fontsize=5)
    ax.set_yticklabels(short_names, fontsize=5)

    ax.set_xlabel("Predicted Class", fontsize=11)
    ax.set_ylabel("True Class", fontsize=11)

    # Color diagonal cells green to highlight correct predictions
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = cm_normalized[i, j]
            text_color = "white" if val > 0.5 else "black"
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center",
                fontsize=3.5,
                color=text_color,
            )

    plt.tight_layout()
    save_path = os.path.join(config.OUTPUTS_DIR, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Metrics] Confusion matrix saved to: {save_path}")


def run_full_evaluation(
    model: tf.keras.Model,
    val_ds: tf.data.Dataset,
    class_names: list,
) -> None:
    """
    Runs the complete evaluation pipeline:
      1. Collect predictions on validation set
      2. Print per-class classification report
      3. Save confusion matrix PNG

    Args:
        model       : Trained Keras model
        val_ds      : Validation tf.data.Dataset
        class_names : List of 38 class name strings
    """
    y_true, y_pred = get_true_and_predicted(model, val_ds)
    print_classification_report(y_true, y_pred, class_names)
    save_confusion_matrix(y_true, y_pred, class_names)
    print("[Metrics] Full evaluation complete.")
