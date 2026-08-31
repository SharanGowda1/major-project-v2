"""
model/blocks.py — Reusable CNN Building Blocks
===============================================
This file defines the fundamental building blocks of our CNN.
The architecture is inspired by VGG-style networks but designed by us.

WHY BLOCKS?
-----------
Instead of writing one giant model, we define a ConvBlock function
that we reuse 4 times with increasing filter counts. This is how
professional CNN architectures are designed — modular and readable.

WHAT HAPPENS INSIDE A ConvBlock?
---------------------------------
Each block takes a feature map, applies two rounds of convolution,
then halves the spatial size with MaxPooling. The number of filters
doubles at each block to learn increasingly complex features:

  Block 1 (32 filters)  → detects simple edges and color gradients
  Block 2 (64 filters)  → detects textures and basic patterns
  Block 3 (128 filters) → detects spots, lesions, shapes
  Block 4 (256 filters) → detects complex disease signatures
"""

import tensorflow as tf


def conv_block(
    inputs: tf.Tensor,
    filters: int,
    dropout_rate: float = 0.25,
    block_name: str = "conv_block",
) -> tf.Tensor:
    """
    One convolutional block: Conv → BN → ReLU → Conv → BN → ReLU → MaxPool → Dropout

    Why TWO convolutions per block?
    --------------------------------
    A single 3×3 conv sees a 3×3 neighborhood.
    Two stacked 3×3 convs effectively see a 5×5 neighborhood,
    with FEWER parameters than one 5×5 conv. This is the VGG insight.
    More depth = more representational power at lower cost.

    Args:
        inputs       : Input tensor from the previous layer
        filters      : Number of convolutional filters (32, 64, 128, or 256)
        dropout_rate : Fraction of neurons to randomly drop (prevents overfitting)
        block_name   : Name prefix for all layers in this block (for readability)

    Returns:
        tf.Tensor: Output feature map after convolution, normalization,
                   pooling and dropout.
    """

    # --- CONVOLUTION 1 ---
    # Conv2D slides `filters` learnable kernels across the input.
    # Each kernel is a 3×3 matrix of weights initialized randomly.
    # padding='same' keeps spatial dimensions unchanged (no border shrinkage).
    # The network learns what these kernels should detect during training.
    x = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=(3, 3),
        padding="same",
        use_bias=False,        # bias is redundant when BatchNorm follows
        kernel_initializer="he_normal",  # best init for ReLU activations
        name=f"{block_name}_conv1",
    )(inputs)

    # --- BATCH NORMALIZATION 1 ---
    # After convolution, raw outputs (called "pre-activations") can have
    # very different scales across filters. BatchNorm normalizes each
    # feature map to zero mean and unit variance, then applies learnable
    # scale (gamma) and shift (beta) parameters.
    # Effect: faster training, higher accuracy, acts as regularization.
    x = tf.keras.layers.BatchNormalization(name=f"{block_name}_bn1")(x)

    # --- RELU ACTIVATION 1 ---
    # ReLU(x) = max(0, x)
    # Replaces all negative values with 0. This is the non-linearity that
    # allows the network to learn non-linear decision boundaries.
    # Without activation functions, stacking layers = just one linear transform.
    x = tf.keras.layers.ReLU(name=f"{block_name}_relu1")(x)

    # --- CONVOLUTION 2 ---
    # A second convolution in the same block allows the network to combine
    # the features detected by the first conv into higher-level patterns.
    x = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=(3, 3),
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        name=f"{block_name}_conv2",
    )(x)

    # --- BATCH NORMALIZATION 2 ---
    x = tf.keras.layers.BatchNormalization(name=f"{block_name}_bn2")(x)

    # --- RELU ACTIVATION 2 ---
    x = tf.keras.layers.ReLU(name=f"{block_name}_relu2")(x)

    # --- MAX POOLING ---
    # Slides a 2×2 window across the feature map and keeps only the maximum
    # value in each window. This halves both height and width.
    # Example: 128×128 → 64×64 after MaxPool(2×2)
    #
    # WHY MAX (not average)?
    # Max pooling keeps the strongest activation — if a disease spot is
    # detected anywhere in a 2×2 region, we keep that strong signal.
    # This also makes the model slightly translation-invariant: a feature
    # slightly shifted still produces the same pooled output.
    x = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        name=f"{block_name}_maxpool",
    )(x)

    # --- DROPOUT ---
    # During training ONLY, randomly sets `dropout_rate` fraction of
    # neurons to zero in each forward pass.
    # This prevents co-adaptation: neurons can't rely on specific other
    # neurons, so each must learn independently useful features.
    # At inference (prediction time), dropout is automatically disabled.
    x = tf.keras.layers.Dropout(
        rate=dropout_rate,
        name=f"{block_name}_dropout",
    )(x)

    return x
