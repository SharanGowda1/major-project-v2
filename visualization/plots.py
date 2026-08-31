"""
visualization/plots.py — Training Curves & Performance Charts
==============================================================
Generates all plots needed for your project report and presentation.

All plots are saved to the outputs/ directory as PNG files.
matplotlib uses the "Agg" backend so it works on WSL without
needing a display server.

PLOTS GENERATED:
  1. training_curves.png    — accuracy & loss over epochs (train vs val)
  2. per_class_accuracy.png — bar chart of F1 score per disease class
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — required for WSL
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import f1_score

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config


# ── Shared style ─────────────────────────────────────────────────────────────

COLORS = {
    "train_acc":  "#2196F3",   # blue
    "val_acc":    "#4CAF50",   # green
    "train_loss": "#FF5722",   # deep orange
    "val_loss":   "#9C27B0",   # purple
    "best_bar":   "#4CAF50",   # green (top classes)
    "worst_bar":  "#F44336",   # red   (bottom classes)
    "grid":       "#EEEEEE",
    "bg":         "#FAFAFA",
}

FONT = {"family": "DejaVu Sans"}
plt.rc("font", **FONT)


# ── Plot 1: Training Curves ───────────────────────────────────────────────────

def plot_training_curves(history) -> None:
    """
    Plots training and validation accuracy + loss curves on two subplots.

    Reading this plot tells you:
      - Did the model learn? (training accuracy should rise)
      - Is it overfitting? (val accuracy drops while train keeps rising)
      - When did EarlyStopping fire? (curves stop before EPOCHS)
      - When did ReduceLROnPlateau kick in? (loss drop becomes steeper)

    Args:
        history: tf.keras.callbacks.History object returned by model.fit()
    """
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)

    epochs_ran = range(1, len(history.history["accuracy"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(COLORS["bg"])

    # ── Accuracy subplot ────────────────────────────────────────────
    ax1 = axes[0]
    ax1.set_facecolor(COLORS["bg"])
    ax1.plot(
        epochs_ran, history.history["accuracy"],
        color=COLORS["train_acc"], linewidth=2.0,
        marker="o", markersize=4, label="Train Accuracy",
    )
    ax1.plot(
        epochs_ran, history.history["val_accuracy"],
        color=COLORS["val_acc"], linewidth=2.0,
        marker="s", markersize=4, label="Validation Accuracy",
        linestyle="--",
    )

    # Mark the best validation accuracy epoch
    best_epoch = int(np.argmax(history.history["val_accuracy"])) + 1
    best_val   = max(history.history["val_accuracy"])
    ax1.axvline(x=best_epoch, color="gray", linestyle=":", linewidth=1.2,
                label=f"Best epoch ({best_epoch})")
    ax1.annotate(
        f"{best_val*100:.1f}%",
        xy=(best_epoch, best_val),
        xytext=(best_epoch + 0.5, best_val - 0.05),
        fontsize=9, color="gray",
        arrowprops=dict(arrowstyle="->", color="gray", lw=1.0),
    )

    ax1.set_title("Model Accuracy Over Epochs", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.set_ylim(0, 1.05)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax1.legend(fontsize=9, loc="lower right")
    ax1.grid(True, color=COLORS["grid"], linewidth=0.8)
    ax1.spines[["top", "right"]].set_visible(False)

    # ── Loss subplot ────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(COLORS["bg"])
    ax2.plot(
        epochs_ran, history.history["loss"],
        color=COLORS["train_loss"], linewidth=2.0,
        marker="o", markersize=4, label="Train Loss",
    )
    ax2.plot(
        epochs_ran, history.history["val_loss"],
        color=COLORS["val_loss"], linewidth=2.0,
        marker="s", markersize=4, label="Validation Loss",
        linestyle="--",
    )

    best_loss_epoch = int(np.argmin(history.history["val_loss"])) + 1
    best_val_loss   = min(history.history["val_loss"])
    ax2.axvline(x=best_loss_epoch, color="gray", linestyle=":", linewidth=1.2,
                label=f"Best epoch ({best_loss_epoch})")
    ax2.annotate(
        f"{best_val_loss:.3f}",
        xy=(best_loss_epoch, best_val_loss),
        xytext=(best_loss_epoch + 0.5, best_val_loss + 0.1),
        fontsize=9, color="gray",
        arrowprops=dict(arrowstyle="->", color="gray", lw=1.0),
    )

    ax2.set_title("Model Loss Over Epochs", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Loss (Categorical Cross-Entropy)", fontsize=11)
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(True, color=COLORS["grid"], linewidth=0.8)
    ax2.spines[["top", "right"]].set_visible(False)

    plt.suptitle(
        "Plant Disease CNN — Training History\n(Built from scratch · No pre-trained weights)",
        fontsize=12, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    save_path = os.path.join(config.OUTPUTS_DIR, "training_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plots] Training curves saved to: {save_path}")


# ── Plot 2: Per-class F1 Bar Chart ───────────────────────────────────────────

def plot_per_class_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
) -> None:
    """
    Plots a horizontal bar chart of F1 score for every disease class,
    sorted from best to worst. Highlights top-10 in green, bottom-10 in red.

    This answers: "Which diseases does our CNN handle best/worst?"

    Args:
        y_true      : Integer array of true class indices
        y_pred      : Integer array of predicted class indices
        class_names : List of 38 class name strings
    """
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)

    # Per-class F1 score (zero_division=0 avoids warning for empty classes)
    f1_scores = f1_score(y_true, y_pred, average=None, zero_division=0)

    # Sort classes by F1 score descending
    sorted_idx   = np.argsort(f1_scores)[::-1]
    sorted_f1    = f1_scores[sorted_idx]
    sorted_names = [
        class_names[i].replace("___", "\n").replace("_", " ")
        for i in sorted_idx
    ]

    # Color: green for top-10, red for bottom-10, gray for middle
    colors = []
    n = len(sorted_f1)
    for rank in range(n):
        if rank < 10:
            colors.append(COLORS["best_bar"])
        elif rank >= n - 10:
            colors.append(COLORS["worst_bar"])
        else:
            colors.append("#90A4AE")  # gray

    fig, ax = plt.subplots(figsize=(10, 14))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    bars = ax.barh(
        range(n), sorted_f1,
        color=colors, edgecolor="white", linewidth=0.5, height=0.75,
    )

    # Value labels on bars
    for i, (bar, val) in enumerate(zip(bars, sorted_f1)):
        ax.text(
            val + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}",
            va="center", ha="left", fontsize=7, color="#333333",
        )

    ax.set_yticks(range(n))
    ax.set_yticklabels(sorted_names, fontsize=6.5)
    ax.set_xlabel("F1 Score (0 = worst, 1 = best)", fontsize=11)
    ax.set_xlim(0, 1.12)
    ax.set_title(
        "Per-Class F1 Score — Plant Disease CNN\n(From Scratch · PlantVillage Dataset)",
        fontsize=12, fontweight="bold", pad=12,
    )
    ax.axvline(x=0.9, color="#FF9800", linestyle="--", linewidth=1.0,
               label="F1 = 0.90 reference")
    ax.grid(True, axis="x", color=COLORS["grid"], linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.invert_yaxis()  # best class at top

    legend_patches = [
        mpatches.Patch(color=COLORS["best_bar"],  label="Top 10 classes"),
        mpatches.Patch(color=COLORS["worst_bar"], label="Bottom 10 classes"),
        mpatches.Patch(color="#90A4AE",            label="Middle classes"),
    ]
    ax.legend(handles=legend_patches, fontsize=8, loc="lower right")

    plt.tight_layout()
    save_path = os.path.join(config.OUTPUTS_DIR, "per_class_accuracy.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plots] Per-class F1 chart saved to: {save_path}")


# ── Convenience wrapper ───────────────────────────────────────────────────────

def generate_all_plots(
    history,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
) -> None:
    """
    Generates all plots in one call.

    Args:
        history    : Keras History object from model.fit()
        y_true     : Integer array of true class indices
        y_pred     : Integer array of predicted class indices
        class_names: List of 38 class name strings
    """
    print("\n[Plots] Generating all plots...")
    plot_training_curves(history)
    plot_per_class_accuracy(y_true, y_pred, class_names)
    print(f"[Plots] All plots saved to: {config.OUTPUTS_DIR}/\n")
