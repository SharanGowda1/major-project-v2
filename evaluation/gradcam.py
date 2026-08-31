"""
evaluation/gradcam.py — Gradient-weighted Class Activation Mapping
===================================================================
Grad-CAM answers the question: "Where in the image did the model look
to make its prediction?"

HOW GRAD-CAM WORKS (step by step):
------------------------------------
1. Pick an image and run a forward pass → get class probabilities
2. Take the predicted class score (e.g., "Tomato Early Blight" = 0.87)
3. Compute gradients of THAT class score with respect to the feature maps
   of the LAST convolutional layer (block4_conv2 in our CNN)
4. These gradients tell us: "Which feature map pixels most influenced
   the final prediction for this class?"
5. Average the gradients across each feature map → importance weights
6. Weighted sum of feature maps → raw heatmap (low resolution: 8×8)
7. ReLU the heatmap → keep only positive influences
8. Upsample the 8×8 heatmap to 128×128 to match the original image
9. Overlay on the original image with a color map (blue→red)

WHY THE LAST CONV LAYER?
--------------------------
Earlier conv layers detect simple features (edges, textures).
The last conv layer (block4) has the richest, most class-specific
feature maps — it knows about disease spots, lesions, and patterns.
Grad-CAM on this layer shows the most meaningful visual explanation.

REFERENCE: Selvaraju et al., 2017 "Grad-CAM: Visual Explanations from
Deep Networks via Gradient-based Localization"
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for WSL/headless
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config


def _make_gradcam_heatmap(
    img_array: np.ndarray,
    model: tf.keras.Model,
    last_conv_layer_name: str,
) -> np.ndarray:
    """
    Core Grad-CAM computation. Computes a gradient-weighted
    class activation heatmap for the top predicted class.

    Args:
        img_array           : Preprocessed image, shape (1, H, W, 3), values in [0,1]
        model               : Trained Keras model (Functional API required)
        last_conv_layer_name: Name of the last conv layer to use for Grad-CAM
                              (must be a layer name in model.layers)

    Returns:
        heatmap (np.ndarray): Normalized heatmap, shape (H', W'), values in [0,1]
    """

    # Build a sub-model that outputs both:
    #   - The feature maps of the last conv layer (for the weighted sum)
    #   - The final model predictions (to get the class score to differentiate)
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,  # feature maps
            model.output,                                   # class probabilities
        ],
    )

    # Use GradientTape to record operations for automatic differentiation.
    # We need: d(class_score) / d(conv_layer_output)
    with tf.GradientTape() as tape:
        # Cast to float32 and watch the input through the tape
        img_tensor = tf.cast(img_array, tf.float32)

        # Forward pass: get conv feature maps and final predictions simultaneously
        conv_outputs, predictions = grad_model(img_tensor, training=False)

        # Get the score of the top predicted class (highest probability)
        top_class_index = tf.argmax(predictions[0])
        top_class_score = predictions[:, top_class_index]

    # Compute gradients: how does each conv feature map pixel
    # affect the top class score?
    # Shape: same as conv_outputs → (1, H', W', num_filters)
    gradients = tape.gradient(top_class_score, conv_outputs)

    # Global average pool the gradients over spatial dimensions (H', W')
    # This gives one importance weight per filter.
    # Shape: (num_filters,)
    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))

    # Weight each filter's feature map by its importance weight
    # Then average across all filters to get the heatmap
    conv_outputs = conv_outputs[0]                          # remove batch dim → (H', W', F)
    heatmap = conv_outputs @ pooled_gradients[..., tf.newaxis]  # (H', W', 1)
    heatmap = tf.squeeze(heatmap)                           # → (H', W')

    # ReLU: keep only positive influences (features that push toward this class)
    # Negative values mean "evidence against this class" — we ignore those
    heatmap = tf.maximum(heatmap, 0)

    # Normalize to [0, 1] for display
    heatmap_max = tf.reduce_max(heatmap)
    if heatmap_max > 0:
        heatmap = heatmap / heatmap_max

    return heatmap.numpy()


def generate_gradcam(
    image_path: str,
    model: tf.keras.Model,
    class_names: list,
    last_conv_layer_name: str = "block4_conv2",
    save_dir: str = None,
) -> str:
    """
    Full Grad-CAM pipeline: load image → compute heatmap → overlay → save.

    Args:
        image_path          : Path to the input leaf image (any jpg/png)
        model               : Trained Keras model
        class_names         : List of 38 class name strings
        last_conv_layer_name: Name of the last conv layer (default: block4_conv2)
        save_dir            : Where to save the output image (default: config.OUTPUTS_DIR)

    Returns:
        str: Path to the saved Grad-CAM visualization image
    """
    if save_dir is None:
        save_dir = config.OUTPUTS_DIR
    os.makedirs(save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # STEP 1: Load and preprocess image (same as training pipeline)
    # ------------------------------------------------------------------
    img = tf.keras.utils.load_img(image_path, target_size=config.IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)        # shape: (128, 128, 3), [0,255]
    img_array_normalized = img_array / 255.0             # normalize to [0, 1]
    img_batch = np.expand_dims(img_array_normalized, axis=0)  # add batch dim → (1, 128, 128, 3)

    # ------------------------------------------------------------------
    # STEP 2: Get the model's prediction
    # ------------------------------------------------------------------
    predictions = model.predict(img_batch, verbose=0)
    top_index = np.argmax(predictions[0])
    top_class = class_names[top_index]
    confidence = predictions[0][top_index] * 100

    print(f"[GradCAM] Predicted: {top_class} ({confidence:.1f}% confidence)")

    # ------------------------------------------------------------------
    # STEP 3: Compute Grad-CAM heatmap
    # ------------------------------------------------------------------
    heatmap = _make_gradcam_heatmap(img_batch, model, last_conv_layer_name)

    # ------------------------------------------------------------------
    # STEP 4: Resize heatmap from 8×8 → 128×128 using interpolation
    # ------------------------------------------------------------------
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_resized = cv2.resize(
        heatmap_uint8,
        (config.IMG_WIDTH, config.IMG_HEIGHT),
        interpolation=cv2.INTER_LINEAR,
    )

    # ------------------------------------------------------------------
    # STEP 5: Apply colormap (COLORMAP_JET: blue→green→yellow→red)
    # ------------------------------------------------------------------
    # Blue  = low activation (model didn't focus here)
    # Red   = high activation (model focused here = disease pattern location)
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # ------------------------------------------------------------------
    # STEP 6: Overlay heatmap on original image (alpha blending)
    # ------------------------------------------------------------------
    original_img = img_array.astype(np.uint8)   # original pixel values [0, 255]
    overlay = cv2.addWeighted(
        original_img,    0.55,   # 55% original image
        heatmap_colored, 0.45,   # 45% heatmap
        0,
    )

    # ------------------------------------------------------------------
    # STEP 7: Save side-by-side figure (Original | Grad-CAM)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(
        f"Grad-CAM Explanation\nPredicted: {top_class.replace('___', ' — ')}\nConfidence: {confidence:.1f}%",
        fontsize=11,
        fontweight="bold",
    )

    axes[0].imshow(original_img)
    axes[0].set_title("Input Image", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title("Grad-CAM Heatmap\n(Red = where model focused)", fontsize=10)
    axes[1].axis("off")

    # Add a colorbar legend
    sm = plt.cm.ScalarMappable(
        cmap="jet",
        norm=plt.Normalize(vmin=0, vmax=1),
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("Activation Intensity", fontsize=8)
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.set_ticklabels(["Low", "Medium", "High"])

    plt.tight_layout()

    # Save file named after the predicted class
    safe_name = top_class.replace(" ", "_").replace("/", "-")
    save_path = os.path.join(save_dir, f"gradcam_{safe_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[GradCAM] Heatmap saved to: {save_path}")
    return save_path
