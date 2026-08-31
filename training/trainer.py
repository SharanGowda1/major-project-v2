"""
training/trainer.py — Model Compilation and Training
======================================================
This file handles two things:
  1. compile_model() — attaches the optimizer, loss function, and metrics
  2. train()         — runs the training loop (model.fit)
  3. evaluate()      — reports final performance on validation set

WHAT IS "COMPILING" A MODEL?
------------------------------
Keras models must be "compiled" before training. Compilation attaches:
  - Optimizer : the algorithm that updates weights (Adam)
  - Loss      : the function that measures how wrong predictions are
  - Metrics   : what we track and print each epoch (accuracy)

Think of it as: architecture (network.py) defines WHAT the model is,
compilation defines HOW it learns.
"""

import os
import sys
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config
from training.callbacks import get_callbacks


def compile_model(model: tf.keras.Model) -> tf.keras.Model:
    """
    Compiles the model with Adam optimizer, Categorical Cross-Entropy loss,
    and accuracy metric.

    OPTIMIZER — Adam (Adaptive Moment Estimation):
    -----------------------------------------------
    Adam is an adaptive learning rate optimizer. Unlike basic SGD which
    uses the same learning rate for all parameters, Adam:
      - Tracks the mean of past gradients (momentum) → smoother updates
      - Tracks the variance of past gradients → scales step size per param
      - Parameters that see sparse gradients (rare features) get larger updates
      - Parameters with consistent gradients get smaller, stable updates

    This makes Adam ideal for image classification tasks where different
    convolutional filters activate at very different frequencies.

    LOSS — Categorical Cross-Entropy:
    ----------------------------------
    For multi-class classification (38 classes), we use categorical
    cross-entropy. Given true label y and predicted probabilities p:

        Loss = -sum(y_i * log(p_i))  for all 38 classes

    Since y is one-hot (only one class is 1, rest are 0), this simplifies to:
        Loss = -log(p_correct_class)

    Interpretation:
      - If the model predicts p=0.99 for the correct class → loss ≈ 0.01 (low)
      - If the model predicts p=0.01 for the correct class → loss ≈ 4.6 (high)
    The optimizer minimizes this value across all training images.

    Args:
        model (tf.keras.Model): Uncompiled model from build_model()

    Returns:
        tf.keras.Model: Same model, now compiled and ready to train
    """

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.LEARNING_RATE   # starts at 1e-3, reduced by callback
    )

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",    # correct loss for one-hot encoded labels
        metrics=["accuracy"],               # track accuracy each epoch
    )

    return model


def train(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
) -> tf.keras.callbacks.History:
    """
    Runs the full training loop.

    Each epoch:
      1. Forward pass  : images flow through the CNN → predictions
      2. Loss computed : categorical cross-entropy between predictions and labels
      3. Backward pass : gradients computed via backpropagation
      4. Weight update : Adam optimizer adjusts all weights
      5. Validation    : evaluate on val_ds (no weight updates here)
      6. Callbacks     : checkpoint, early stopping, LR reduction

    Args:
        model   : Compiled Keras model
        train_ds: Augmented training tf.data.Dataset
        val_ds  : Rescaled validation tf.data.Dataset

    Returns:
        history: Keras History object containing per-epoch metrics
                 (used by visualization/plots.py to draw training curves)
    """

    print("\n" + "=" * 55)
    print("  STARTING TRAINING")
    print(f"  Max epochs : {config.EPOCHS}")
    print(f"  Batch size : {config.BATCH_SIZE}")
    print(f"  Initial LR : {config.LEARNING_RATE}")
    print(f"  Early stop : patience={config.EARLY_STOPPING_PATIENCE}")
    print("=" * 55 + "\n")

    # Ensure output directory exists for any saved artifacts
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)

    history = model.fit(
        train_ds,                           # augmented training batches
        epochs=config.EPOCHS,              # maximum epochs (EarlyStopping may stop sooner)
        validation_data=val_ds,            # evaluate on validation set every epoch
        callbacks=get_callbacks(),         # checkpoint + early stopping + LR scheduler
        verbose=1,                         # print progress bar per epoch
    )

    print("\n[Trainer] Training complete.")
    print(f"[Trainer] Best model saved to: {config.MODEL_SAVE_PATH}")

    return history


def evaluate(
    model: tf.keras.Model,
    val_ds: tf.data.Dataset,
) -> tuple:
    """
    Evaluates the model on the validation dataset and prints results.

    Called after training to get the final performance numbers.

    Args:
        model : Trained Keras model
        val_ds: Validation tf.data.Dataset

    Returns:
        (val_loss, val_accuracy): Final validation metrics as floats
    """

    print("\n[Trainer] Evaluating on validation set...")
    val_loss, val_accuracy = model.evaluate(val_ds, verbose=1)

    print("\n" + "=" * 55)
    print("  FINAL EVALUATION RESULTS")
    print("=" * 55)
    print(f"  Validation Loss    : {val_loss:.4f}")
    print(f"  Validation Accuracy: {val_accuracy * 100:.2f}%")
    print("=" * 55 + "\n")

    return val_loss, val_accuracy
