"""
model/network.py — Full CNN Architecture
=========================================
Assembles the complete Plant Disease CNN from scratch using ConvBlocks.

THIS IS THE HEART OF THE PROJECT.
-----------------------------------
No MobileNetV2. No pre-trained weights. No transfer learning.
Every parameter starts as random noise and is learned from the
PlantVillage dataset through backpropagation.

ARCHITECTURE OVERVIEW:
-----------------------
Input (128 × 128 × 3)
    ↓
ConvBlock 1 — 32 filters  → output: 64 × 64 × 32
    ↓
ConvBlock 2 — 64 filters  → output: 32 × 32 × 64
    ↓
ConvBlock 3 — 128 filters → output: 16 × 16 × 128
    ↓
ConvBlock 4 — 256 filters → output:  8 ×  8 × 256
    ↓
GlobalAveragePooling2D    → output: 256
    ↓
Dense(512, ReLU)          → output: 512
    ↓
Dropout(0.5)
    ↓
Dense(38, Softmax)        → output: 38 (one probability per disease class)

DESIGN RATIONALE:
-----------------
- Filter counts double at each block (32→64→128→256): standard practice.
  Early layers learn simple features (edges), later layers combine them
  into complex patterns (disease lesions, spots, textures).
  Doubling filters compensates for spatial size being halved by MaxPool.

- GlobalAveragePooling instead of Flatten:
  Flatten(8×8×256) = 16,384 values → Dense(512) = 8.4M parameters (huge!)
  GAP(8×8×256) = 256 values → Dense(512) = 131K parameters (efficient)
  GAP also provides spatial averaging which reduces overfitting.

- Dense(512) before output:
  Combines the 256 global features into 512 higher-level combinations
  before making the final 38-class decision.

- Softmax on final layer:
  Converts 38 raw scores (logits) into probabilities that sum to 1.0
  e.g. [0.72, 0.15, 0.03, ...] — readable confidence scores.
"""

import tensorflow as tf
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config
from model.blocks import conv_block


def build_model(num_classes: int = config.NUM_CLASSES) -> tf.keras.Model:
    """
    Constructs and returns the full Plant Disease CNN.

    Uses the Keras Functional API (not Sequential) because it gives us
    explicit control over the data flow and allows inspection of
    intermediate layer outputs (needed for Grad-CAM).

    Args:
        num_classes (int): Number of output classes (38 for PlantVillage)

    Returns:
        tf.keras.Model: Uncompiled CNN model ready for training
    """

    # ------------------------------------------------------------------
    # INPUT LAYER
    # ------------------------------------------------------------------
    # Explicitly defines the shape of one image: (height, width, channels)
    # The batch dimension is NOT included here — Keras handles batching.
    inputs = tf.keras.Input(
        shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS),
        name="input_layer",
    )

    # ------------------------------------------------------------------
    # CONVOLUTIONAL BLOCK 1 — 32 filters
    # ------------------------------------------------------------------
    # Input : 128 × 128 × 3   (RGB image)
    # Output:  64 ×  64 × 32  (after MaxPool halves spatial size)
    #
    # At this scale, kernels detect low-level features:
    # horizontal edges, vertical edges, color transitions, simple gradients.
    x = conv_block(
        inputs,
        filters=32,
        dropout_rate=0.25,
        block_name="block1",
    )

    # ------------------------------------------------------------------
    # CONVOLUTIONAL BLOCK 2 — 64 filters
    # ------------------------------------------------------------------
    # Input :  64 ×  64 × 32
    # Output:  32 ×  32 × 64
    #
    # Features detected: textures, repeated patterns, basic shapes.
    # The model starts combining edges into more complex structures.
    x = conv_block(
        x,
        filters=64,
        dropout_rate=0.25,
        block_name="block2",
    )

    # ------------------------------------------------------------------
    # CONVOLUTIONAL BLOCK 3 — 128 filters
    # ------------------------------------------------------------------
    # Input :  32 ×  32 ×  64
    # Output:  16 ×  16 × 128
    #
    # Features detected: spots, lesions, irregular patches, color blotches.
    # These are the intermediate-level disease markers.
    x = conv_block(
        x,
        filters=128,
        dropout_rate=0.25,
        block_name="block3",
    )

    # ------------------------------------------------------------------
    # CONVOLUTIONAL BLOCK 4 — 256 filters
    # ------------------------------------------------------------------
    # Input :  16 ×  16 × 128
    # Output:   8 ×   8 × 256
    #
    # Features detected: high-level disease signatures — the combination
    # of spot patterns, colors, textures, and shapes that uniquely
    # identify a specific plant disease.
    x = conv_block(
        x,
        filters=256,
        dropout_rate=0.25,
        block_name="block4",
    )

    # ------------------------------------------------------------------
    # GLOBAL AVERAGE POOLING
    # ------------------------------------------------------------------
    # Takes the 8×8×256 feature maps and averages each of the 256 feature
    # maps into a single number. Output: a vector of 256 values.
    #
    # Each value in this vector represents "how strongly was this feature
    # (e.g., dark brown spots) present anywhere in the image?"
    #
    # This layer is also the key to Grad-CAM visualization — we use
    # gradients flowing back through this layer to find where in the
    # image the model was focusing its attention.
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    # ------------------------------------------------------------------
    # FULLY CONNECTED LAYER
    # ------------------------------------------------------------------
    # Dense(512): Takes the 256 global feature values and creates 512
    # learned combinations of them. Each combination might represent
    # something like "this is a combination of brown spots + yellow halo
    # pattern typical of Early Blight".
    x = tf.keras.layers.Dense(
        units=config.FC_UNITS,   # 512
        activation="relu",
        kernel_initializer="he_normal",
        name="dense_512",
    )(x)

    # ------------------------------------------------------------------
    # DROPOUT (Heavy regularization before output)
    # ------------------------------------------------------------------
    # 50% dropout here is standard for the fully connected layer.
    # Prevents the Dense layer from over-relying on any single feature.
    x = tf.keras.layers.Dropout(
        rate=config.DROPOUT_RATE,  # 0.5
        name="dropout_final",
    )(x)

    # ------------------------------------------------------------------
    # OUTPUT LAYER — Softmax classification
    # ------------------------------------------------------------------
    # Dense(38): 38 neurons, one per disease class.
    # Each neuron computes a raw score (logit) for its class.
    #
    # Softmax converts these 38 logits into probabilities:
    #   softmax(z_i) = exp(z_i) / sum(exp(z_j) for all j)
    #
    # Properties of Softmax output:
    #   - All 38 values are between 0.0 and 1.0
    #   - All 38 values sum to exactly 1.0
    #   - The highest value is the model's predicted class
    outputs = tf.keras.layers.Dense(
        units=num_classes,        # 38
        activation="softmax",
        name="output_softmax",
    )(x)

    # ------------------------------------------------------------------
    # BUILD THE MODEL
    # ------------------------------------------------------------------
    # The Keras Functional API requires explicitly declaring the
    # inputs and outputs to create the Model object.
    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="PlantDiseaseCNN_FromScratch",
    )

    return model


def print_model_summary(model: tf.keras.Model) -> None:
    """
    Prints a detailed summary of the model architecture.
    Shows every layer, its output shape, and parameter count.

    Args:
        model: The built Keras model
    """
    print("\n" + "=" * 65)
    print("  PLANT DISEASE CNN — ARCHITECTURE SUMMARY")
    print("  (Built from scratch — no pre-trained weights)")
    print("=" * 65)
    model.summary(line_length=65)
    print("=" * 65)

    total_params = model.count_params()
    trainable = sum(
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    )
    print(f"  Total parameters    : {total_params:,}")
    print(f"  Trainable parameters: {trainable:,}")
    print(f"  Non-trainable (BN)  : {total_params - trainable:,}")
    print("=" * 65 + "\n")
