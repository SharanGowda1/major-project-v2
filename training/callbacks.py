"""
training/callbacks.py — Smart Training Controls
=================================================
Callbacks are functions that Keras calls automatically at the end of
each epoch. They monitor training and take actions without us having
to manually intervene.

WHY CALLBACKS?
--------------
Training 54,000 images for 25 epochs takes time. Without callbacks:
  - We might overtrain and overfit (model memorizes training data)
  - We might miss the best model checkpoint
  - We might waste time when training has already plateaued

These 3 callbacks solve those problems automatically.
"""

import os
import sys
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config


def get_callbacks() -> list:
    """
    Returns a list of Keras callbacks for smart training control.

    Used as: model.fit(..., callbacks=get_callbacks())

    Returns:
        list: [ModelCheckpoint, EarlyStopping, ReduceLROnPlateau]
    """

    # Ensure the saved_models directory exists before trying to save to it
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # CALLBACK 1: ModelCheckpoint
    # ------------------------------------------------------------------
    # Saves the model weights to disk whenever val_accuracy improves.
    # This ensures we always keep the BEST version of the model,
    # even if later epochs overfit and performance drops.
    #
    # Example: If epoch 12 has the best val_accuracy=0.89, the checkpoint
    # saves that version. Even if epoch 13 drops to 0.85, we keep epoch 12.
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=config.MODEL_SAVE_PATH,   # where to save (e.g., saved_models/plant_disease_cnn_best.keras)
        monitor="val_accuracy",            # watch validation accuracy (not training!)
        save_best_only=True,               # only overwrite if this epoch is better
        save_weights_only=False,           # save full model (architecture + weights)
        mode="max",                        # "max" because higher accuracy = better
        verbose=1,                         # print a message when saving
    )

    # ------------------------------------------------------------------
    # CALLBACK 2: EarlyStopping
    # ------------------------------------------------------------------
    # Stops training automatically if validation accuracy hasn't improved
    # for `patience` consecutive epochs.
    #
    # WHY: On CPU, each epoch takes ~5-10 minutes. If the model stopped
    # learning at epoch 15, there is no point running to epoch 25.
    # EarlyStopping saves that time automatically.
    #
    # restore_best_weights=True: When stopping, reloads the weights from
    # the best epoch (not the last epoch which may have overfit).
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=config.EARLY_STOPPING_PATIENCE,    # stop after 5 non-improving epochs
        mode="max",
        restore_best_weights=True,                  # revert to best epoch weights
        verbose=1,
    )

    # ------------------------------------------------------------------
    # CALLBACK 3: ReduceLROnPlateau
    # ------------------------------------------------------------------
    # Reduces the learning rate when training gets stuck (loss plateaus).
    #
    # HOW IT WORKS:
    # If val_loss doesn't decrease for `patience` epochs, multiply the
    # current learning rate by `factor`.
    # Example: LR=0.001 → stuck for 3 epochs → LR becomes 0.0005
    #          LR=0.0005 → stuck for 3 epochs → LR becomes 0.00025
    #
    # WHY: A large learning rate helps at the start (big steps toward minimum).
    # As training progresses, large steps can "overshoot" the optimal point.
    # A smaller LR takes finer steps to reach the true minimum.
    #
    # This is like switching from coarse to fine-grained searching.
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=config.LR_REDUCE_FACTOR,     # multiply LR by 0.5 (halve it)
        patience=config.LR_REDUCE_PATIENCE, # wait 3 epochs before reducing
        min_lr=config.LR_MIN,               # never go below 1e-6
        mode="min",                         # "min" because lower loss = better
        verbose=1,
    )

    return [checkpoint, early_stopping, reduce_lr]
