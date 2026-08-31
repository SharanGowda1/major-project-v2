"""
Plant Disease CNN — Kaggle Training Notebook
=============================================
Run this on Kaggle with GPU enabled.
Expected training time: ~30-60 minutes with T4 GPU.

INSTRUCTIONS:
1. Create a new Kaggle notebook
2. Add dataset: abdallahalidev/plantvillage-dataset
3. Set "Accelerator" to GPU T4 x2 (in notebook settings, right panel)
4. Paste this entire file into the notebook cell
5. Click Run All
6. Download from Output tab:
       /kaggle/working/saved_models/plant_disease_cnn_best.keras
       /kaggle/working/saved_models/class_names.json
"""

# ============================================================
# 0. INSTALL / CHECK
# ============================================================
import subprocess
subprocess.run(["pip", "install", "opencv-python-headless", "-q"])

import os, json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from sklearn.metrics import classification_report, f1_score, confusion_matrix

print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

# ============================================================
# 1. CONFIG — All settings in one place
# ============================================================

# Kaggle dataset path (added automatically when you add the dataset)
DATA_DIR = "/kaggle/input/datasets/abdallahalidev/plantvillage-dataset/color"

# Kaggle working directory — files saved here persist and can be downloaded
WORKING_DIR      = "/kaggle/working"
MODELS_DIR       = os.path.join(WORKING_DIR, "saved_models")
OUTPUTS_DIR      = os.path.join(WORKING_DIR, "outputs")
MODEL_SAVE_PATH  = os.path.join(MODELS_DIR, "plant_disease_cnn_best.keras")
CLASS_NAMES_PATH = os.path.join(MODELS_DIR, "class_names.json")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Image & training settings
IMG_SIZE          = (128, 128)
IMG_HEIGHT        = 128
IMG_WIDTH         = 128
NUM_CLASSES       = 38
BATCH_SIZE        = 64       # Kaggle GPU can handle larger batches
EPOCHS            = 30
LEARNING_RATE     = 1e-3
VALIDATION_SPLIT  = 0.2
SEED              = 42
DROPOUT_RATE      = 0.5
FC_UNITS          = 512

# Callback settings
EARLY_STOP_PATIENCE = 5
LR_REDUCE_PATIENCE  = 3
LR_REDUCE_FACTOR    = 0.5
LR_MIN              = 1e-6
TOP_K               = 3

print("Config loaded.")
print(f"Dataset path   : {DATA_DIR}")
print(f"Dataset exists : {os.path.exists(DATA_DIR)}")
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(
        f"\n[ERROR] Dataset not found at {DATA_DIR}\n"
        "Did you add the 'abdallahalidev/plantvillage-dataset' dataset to this notebook?\n"
        "Go to the right panel → '+ Add Data' → search 'plantvillage'\n"
    )
print(f"Class folders  : {len(os.listdir(DATA_DIR))}")


# ============================================================
# 2. DATA AUGMENTATION PIPELINE
# ============================================================

def build_augmentation_layer():
    """
    Augmentation applied to TRAINING data only.
    Rescaling(1/255) normalizes pixel values [0,255] → [0.0,1.0]
    """
    return tf.keras.Sequential([
        tf.keras.layers.Rescaling(1.0 / 255),
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2, fill_mode="reflect"),
        tf.keras.layers.RandomZoom(0.1, fill_mode="reflect"),
        tf.keras.layers.RandomBrightness(0.1, value_range=(0.0, 1.0)),
    ], name="augmentation_pipeline")


def build_rescaling_layer():
    """Rescaling only — for validation data (no augmentation)."""
    return tf.keras.Sequential([
        tf.keras.layers.Rescaling(1.0 / 255),
    ], name="rescaling_only")


# ============================================================
# 3. DATA LOADING
# ============================================================

def load_datasets():
    """
    Loads PlantVillage from disk.
    Returns train_ds, val_ds, class_names.
    """
    print(f"\n[Loader] Loading from: {DATA_DIR}")

    train_ds_raw = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=True,
    )

    val_ds_raw = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False,
    )

    class_names = train_ds_raw.class_names
    print(f"[Loader] Found {len(class_names)} classes.")

    augment = build_augmentation_layer()
    rescale = build_rescaling_layer()
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds_raw.map(
        lambda imgs, lbls: (augment(imgs, training=True), lbls),
        num_parallel_calls=AUTOTUNE,
    ).prefetch(AUTOTUNE)

    val_ds = val_ds_raw.map(
        lambda imgs, lbls: (rescale(imgs, training=False), lbls),
        num_parallel_calls=AUTOTUNE,
    ).prefetch(AUTOTUNE)

    # Save class names
    with open(CLASS_NAMES_PATH, "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"[Loader] Class names saved to {CLASS_NAMES_PATH}")

    approx_train = int(54305 * (1 - VALIDATION_SPLIT))
    approx_val   = int(54305 * VALIDATION_SPLIT)
    print(f"[Loader] Train: ~{approx_train:,} images | Val: ~{approx_val:,} images")

    return train_ds, val_ds, class_names


# ============================================================
# 4. CNN ARCHITECTURE — BUILT FROM SCRATCH
# ============================================================

def conv_block(inputs, filters, dropout_rate=0.25, block_name="block"):
    """
    One ConvBlock: Conv→BN→ReLU→Conv→BN→ReLU→MaxPool→Dropout

    Two stacked 3×3 convolutions = effectively 5×5 receptive field
    with fewer parameters. This is the VGG insight.

    BatchNormalization stabilizes training by normalizing activations.
    ReLU introduces non-linearity: f(x) = max(0, x)
    MaxPooling halves spatial dimensions, keeps dominant features.
    Dropout randomly zeros neurons to prevent overfitting.
    """
    # Conv 1
    x = tf.keras.layers.Conv2D(
        filters=filters, kernel_size=(3, 3), padding="same",
        use_bias=False, kernel_initializer="he_normal",
        name=f"{block_name}_conv1",
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name=f"{block_name}_bn1")(x)
    x = tf.keras.layers.ReLU(name=f"{block_name}_relu1")(x)

    # Conv 2
    x = tf.keras.layers.Conv2D(
        filters=filters, kernel_size=(3, 3), padding="same",
        use_bias=False, kernel_initializer="he_normal",
        name=f"{block_name}_conv2",
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f"{block_name}_bn2")(x)
    x = tf.keras.layers.ReLU(name=f"{block_name}_relu2")(x)

    # Downsample
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name=f"{block_name}_maxpool")(x)
    x = tf.keras.layers.Dropout(rate=dropout_rate, name=f"{block_name}_dropout")(x)

    return x


def build_model(num_classes=NUM_CLASSES):
    """
    Full Plant Disease CNN assembled from 4 ConvBlocks.

    Architecture:
        Input (128×128×3)
        → ConvBlock(32)   64×64×32   [edges, gradients]
        → ConvBlock(64)   32×32×64   [textures, patterns]
        → ConvBlock(128)  16×16×128  [spots, lesions]
        → ConvBlock(256)   8×8×256   [disease signatures]
        → GlobalAveragePooling2D → 256
        → Dense(512, relu)
        → Dropout(0.5)
        → Dense(38, softmax) → 38 class probabilities

    NO pre-trained weights. All weights initialized randomly.
    All learning happens on PlantVillage data.
    """
    inputs = tf.keras.Input(
        shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        name="input_layer",
    )

    x = conv_block(inputs, 32,  dropout_rate=0.25, block_name="block1")
    x = conv_block(x,      64,  dropout_rate=0.25, block_name="block2")
    x = conv_block(x,      128, dropout_rate=0.25, block_name="block3")
    x = conv_block(x,      256, dropout_rate=0.25, block_name="block4")

    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.Dense(FC_UNITS, activation="relu",
                               kernel_initializer="he_normal", name="dense_512")(x)
    x = tf.keras.layers.Dropout(DROPOUT_RATE, name="dropout_final")(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax",
                                     name="output_softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs,
                           name="PlantDiseaseCNN_FromScratch")
    return model


# ============================================================
# 5. TRAINING CALLBACKS
# ============================================================

def get_callbacks():
    return [
        # Save best model by val_accuracy
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_SAVE_PATH,
            monitor="val_accuracy", save_best_only=True,
            mode="max", verbose=1,
        ),
        # Stop early if no improvement for 5 epochs
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=EARLY_STOP_PATIENCE,
            mode="max", restore_best_weights=True, verbose=1,
        ),
        # Halve LR if val_loss plateaus for 3 epochs
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=LR_REDUCE_FACTOR,
            patience=LR_REDUCE_PATIENCE, min_lr=LR_MIN,
            mode="min", verbose=1,
        ),
    ]


# ============================================================
# 6. EVALUATION HELPERS
# ============================================================

def get_true_and_predicted(model, val_ds):
    y_true, y_pred = [], []
    for images, labels in val_ds:
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
    return np.array(y_true), np.array(y_pred)


def plot_training_curves(history):
    epochs_ran = range(1, len(history.history["accuracy"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs_ran, history.history["accuracy"],
                 label="Train Accuracy", color="#2196F3", marker="o", markersize=4)
    axes[0].plot(epochs_ran, history.history["val_accuracy"],
                 label="Val Accuracy", color="#4CAF50", marker="s",
                 markersize=4, linestyle="--")
    axes[0].set_title("Accuracy Over Epochs", fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[0].yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))

    axes[1].plot(epochs_ran, history.history["loss"],
                 label="Train Loss", color="#FF5722", marker="o", markersize=4)
    axes[1].plot(epochs_ran, history.history["val_loss"],
                 label="Val Loss", color="#9C27B0", marker="s",
                 markersize=4, linestyle="--")
    axes[1].set_title("Loss Over Epochs", fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.suptitle("Plant Disease CNN — Training History\n(Built from scratch · No pre-trained weights)",
                 fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plots] Training curves saved: {path}")


def plot_per_class_f1(y_true, y_pred, class_names):
    f1_scores = f1_score(y_true, y_pred, average=None, zero_division=0)
    sorted_idx = np.argsort(f1_scores)[::-1]
    sorted_f1  = f1_scores[sorted_idx]
    sorted_names = [class_names[i].replace("___", "\n").replace("_", " ")
                    for i in sorted_idx]
    n = len(sorted_f1)
    colors = ["#4CAF50" if i < 10 else ("#F44336" if i >= n-10 else "#90A4AE")
              for i in range(n)]

    fig, ax = plt.subplots(figsize=(10, 14))
    ax.barh(range(n), sorted_f1, color=colors, height=0.75)
    ax.set_yticks(range(n)); ax.set_yticklabels(sorted_names, fontsize=6.5)
    ax.set_xlabel("F1 Score"); ax.set_xlim(0, 1.1)
    ax.set_title("Per-Class F1 Score — Plant Disease CNN", fontweight="bold")
    ax.invert_yaxis(); ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, "per_class_f1.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plots] Per-class F1 chart saved: {path}")


def make_gradcam_heatmap(img_array, model, last_conv_layer_name="block4_conv2"):
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )
    with tf.GradientTape() as tape:
        img_tensor = tf.cast(img_array, tf.float32)
        conv_outputs, predictions = grad_model(img_tensor, training=False)
        top_class_score = predictions[:, tf.argmax(predictions[0])]
    gradients = tape.gradient(top_class_score, conv_outputs)
    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_gradients[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    hm_max = tf.reduce_max(heatmap)
    if hm_max > 0:
        heatmap = heatmap / hm_max
    return heatmap.numpy()


def generate_gradcam(image_path, model, class_names):
    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_norm  = img_array / 255.0
    img_batch = np.expand_dims(img_norm, axis=0)

    preds     = model.predict(img_batch, verbose=0)
    top_idx   = np.argmax(preds[0])
    top_class = class_names[top_idx]
    conf      = preds[0][top_idx] * 100

    heatmap      = make_gradcam_heatmap(img_batch, model)
    hm_uint8     = np.uint8(255 * heatmap)
    hm_resized   = cv2.resize(hm_uint8, (IMG_WIDTH, IMG_HEIGHT),
                               interpolation=cv2.INTER_LINEAR)
    hm_colored   = cv2.applyColorMap(hm_resized, cv2.COLORMAP_JET)
    hm_colored   = cv2.cvtColor(hm_colored, cv2.COLOR_BGR2RGB)
    original_img = img_array.astype(np.uint8)
    overlay      = cv2.addWeighted(original_img, 0.55, hm_colored, 0.45, 0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f"Grad-CAM: {top_class.replace('___',' — ')}\nConfidence: {conf:.1f}%",
                 fontweight="bold")
    axes[0].imshow(original_img); axes[0].set_title("Input"); axes[0].axis("off")
    axes[1].imshow(overlay);      axes[1].set_title("Grad-CAM Heatmap"); axes[1].axis("off")
    plt.tight_layout()

    safe = top_class.replace(" ", "_").replace("/", "-")
    path = os.path.join(OUTPUTS_DIR, f"gradcam_{safe}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[GradCAM] Saved: {path}")
    return path


# ============================================================
# 7. MAIN TRAINING RUN
# ============================================================

print("\n" + "="*58)
print("  PLANT DISEASE CNN — TRAINING PIPELINE")
print("  Custom Architecture · Built from Scratch")
print("  No pre-trained weights · No transfer learning")
print("="*58 + "\n")

# --- Load Data ---
train_ds, val_ds, class_names = load_datasets()

# --- Build Model ---
print("\n[Step 2] Building CNN from scratch...")
model = build_model(num_classes=len(class_names))
model.summary()

total_params = model.count_params()
print(f"\nTotal parameters: {total_params:,}")

# --- Compile ---
print("\n[Step 3] Compiling (Adam + CategoricalCrossEntropy)...")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# --- Train ---
print("\n[Step 4] Training...")
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=get_callbacks(),
    verbose=1,
)

# --- Evaluate ---
print("\n[Step 5] Evaluating...")
val_loss, val_acc = model.evaluate(val_ds, verbose=1)
print(f"\nFinal Validation Accuracy: {val_acc*100:.2f}%")
print(f"Final Validation Loss    : {val_loss:.4f}")

# --- Plots ---
print("\n[Step 6] Generating plots...")
plot_training_curves(history)
y_true, y_pred = get_true_and_predicted(model, val_ds)
plot_per_class_f1(y_true, y_pred, class_names)

# --- Classification Report ---
print("\n[Step 7] Classification Report:")
print(classification_report(
    y_true, y_pred,
    labels=list(range(len(class_names))),
    target_names=class_names,
    digits=3,
    zero_division=0,
))

# --- Grad-CAM on one sample image ---
print("\n[Step 8] Generating Grad-CAM sample...")
# Find one image from the first class to demo Grad-CAM
sample_class_dir = os.path.join(DATA_DIR, class_names[0])
sample_images = [f for f in os.listdir(sample_class_dir)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
if sample_images:
    sample_path = os.path.join(sample_class_dir, sample_images[0])
    generate_gradcam(sample_path, model, class_names)

# --- Summary ---
print("\n" + "="*58)
print("  TRAINING COMPLETE!")
print(f"  Best model : {MODEL_SAVE_PATH}")
print(f"  Classes    : {CLASS_NAMES_PATH}")
print(f"  Plots      : {OUTPUTS_DIR}/")
print("="*58)
print("\nDOWNLOAD THESE FILES:")
print(f"  {MODEL_SAVE_PATH}")
print(f"  {CLASS_NAMES_PATH}")
