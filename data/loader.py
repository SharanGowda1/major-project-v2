"""
data/loader.py — Dataset Loading Pipeline
==========================================
Loads the PlantVillage dataset from disk using tf.data API.

WHAT THIS FILE DOES:
--------------------
1. Reads all 54,305 images from raw/color/ using Keras utility
2. Splits into 80% training / 20% validation (deterministically)
3. Applies augmentation to training set ONLY
4. Applies performance optimizations (cache, prefetch) so the
   CPU data loading doesn't bottleneck GPU/CPU computation

WHY tf.data INSTEAD OF ImageDataGenerator?
-------------------------------------------
ImageDataGenerator (used in old main.py) is legacy and slow.
tf.data is TensorFlow's modern pipeline: it loads batches in
parallel while the model is training the previous batch.
This is called "prefetching" and is critical for CPU training speed.
"""

import tensorflow as tf
import json
import os
import sys

# Allow imports from project root
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config
from data.augmentation import build_augmentation_layer, build_rescaling_layer


def load_datasets():
    """
    Loads the PlantVillage dataset and returns train/validation tf.data pipelines.

    The dataset directory structure must be:
        raw/color/
            Apple___Apple_scab/   ← class folder name becomes the label
                img001.jpg
                img002.jpg
            Apple___Black_rot/
                ...

    Keras reads the folder names as class labels automatically.

    Returns:
        train_ds  (tf.data.Dataset): Augmented training dataset, batched & prefetched
        val_ds    (tf.data.Dataset): Rescaled validation dataset, batched & prefetched
        class_names (list[str])    : Sorted list of 38 class name strings
    """

    # Verify the dataset directory exists before loading
    if not os.path.exists(config.DATA_DIR):
        raise FileNotFoundError(
            f"\n[ERROR] Dataset not found at: {config.DATA_DIR}\n"
            f"Please paste your 'color' folder (38 subfolders) inside:\n"
            f"  plant-disease-cnn/raw/color/\n"
        )

    print(f"[Loader] Loading dataset from: {config.DATA_DIR}")
    print(f"[Loader] Image size: {config.IMG_SIZE} | Batch size: {config.BATCH_SIZE}")

    # ------------------------------------------------------------------
    # STEP 1: Load TRAINING split (80% of data)
    # ------------------------------------------------------------------
    # image_dataset_from_directory reads all subfolders as classes.
    # It automatically assigns integer labels 0..37 in alphabetical order.
    # seed= ensures the same images always go to train vs val.
    train_ds_raw = tf.keras.utils.image_dataset_from_directory(
        config.DATA_DIR,
        validation_split=config.VALIDATION_SPLIT,   # 20% goes to validation
        subset="training",
        seed=config.SEED,                            # reproducible split
        image_size=config.IMG_SIZE,                  # resize all images to 128x128
        batch_size=config.BATCH_SIZE,
        label_mode="categorical",                    # one-hot vectors e.g. [0,0,1,...,0]
        shuffle=True,                                # shuffle training data each epoch
    )

    # ------------------------------------------------------------------
    # STEP 2: Load VALIDATION split (20% of data)
    # ------------------------------------------------------------------
    val_ds_raw = tf.keras.utils.image_dataset_from_directory(
        config.DATA_DIR,
        validation_split=config.VALIDATION_SPLIT,
        subset="validation",
        seed=config.SEED,
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        label_mode="categorical",
        shuffle=False,                               # no shuffle for validation
    )

    # Extract class names from the directory structure (alphabetically sorted)
    # e.g. ['Apple___Apple_scab', 'Apple___Black_rot', ..., 'Tomato___Tomato_mosaic_virus']
    class_names = train_ds_raw.class_names
    print(f"[Loader] Found {len(class_names)} classes.")

    # ------------------------------------------------------------------
    # STEP 3: Apply augmentation pipeline
    # ------------------------------------------------------------------
    # Training: augment + rescale (random flips, rotations, zoom, brightness)
    # Validation: rescale ONLY — we never augment evaluation data
    augment = build_augmentation_layer()
    rescale = build_rescaling_layer()

    # map() applies the function to every batch in the dataset
    # num_parallel_calls=AUTOTUNE lets TensorFlow decide optimal parallelism
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds_raw.map(
        lambda images, labels: (augment(images, training=True), labels),
        num_parallel_calls=AUTOTUNE,
    )

    val_ds = val_ds_raw.map(
        lambda images, labels: (rescale(images, training=False), labels),
        num_parallel_calls=AUTOTUNE,
    )

    # ------------------------------------------------------------------
    # STEP 4: Performance optimizations
    # ------------------------------------------------------------------
    # cache(): stores preprocessed batches in memory after first epoch
    #          so subsequent epochs don't re-read from disk
    # prefetch(): prepares the next batch while the model trains on current
    #             batch — overlaps CPU data loading with model computation
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds   = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # ------------------------------------------------------------------
    # STEP 5: Save class names to disk (needed later for inference)
    # ------------------------------------------------------------------
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    with open(config.CLASS_NAMES_PATH, "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"[Loader] Class names saved to: {config.CLASS_NAMES_PATH}")

    return train_ds, val_ds, class_names


def get_dataset_info(class_names: list) -> None:
    """
    Prints a summary of the dataset for verification.

    Args:
        class_names (list[str]): The list of class names returned by load_datasets()
    """
    print("\n" + "=" * 55)
    print("  DATASET SUMMARY")
    print("=" * 55)
    print(f"  Total classes    : {len(class_names)}")
    print(f"  Image size       : {config.IMG_SIZE[0]} x {config.IMG_SIZE[1]} px")
    print(f"  Batch size       : {config.BATCH_SIZE}")
    print(f"  Train split      : {int((1 - config.VALIDATION_SPLIT) * 100)}%")
    print(f"  Validation split : {int(config.VALIDATION_SPLIT * 100)}%")
    print(f"  Approx train imgs: ~{int(54305 * (1 - config.VALIDATION_SPLIT)):,}")
    print(f"  Approx val imgs  : ~{int(54305 * config.VALIDATION_SPLIT):,}")
    print("=" * 55)
    print("  First 5 classes:")
    for i, name in enumerate(class_names[:5]):
        print(f"    [{i:2d}] {name}")
    print("  ...")
    print("=" * 55 + "\n")
