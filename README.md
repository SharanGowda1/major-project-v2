# 🌿 Plant Disease Detection CNN — Built From Scratch

> **A custom Convolutional Neural Network that detects 38 plant diseases from leaf images.**
> Built entirely from scratch — no borrowed pre-trained model, no shortcuts.
> Every layer, every weight, every decision is explained below.

---

## 📌 Table of Contents

1. [What This Project Does](#what-this-project-does)
2. [The Dataset](#the-dataset)
3. [Project Folder Structure](#project-folder-structure)
4. [Complete Flow — How Everything Connects](#complete-flow)
5. [Step-by-Step Deep Dive](#step-by-step-deep-dive)
   - [Step 1 — Data Loading](#step-1--data-loading-dataloarderpy)
   - [Step 2 — Data Augmentation](#step-2--data-augmentation-dataaugmentationpy)
   - [Step 3 — CNN Architecture](#step-3--cnn-architecture-the-heart-of-the-project)
   - [Step 4 — Training Pipeline](#step-4--training-pipeline)
   - [Step 5 — Evaluation & Grad-CAM](#step-5--evaluation--grad-cam)
   - [Step 6 — Visualization](#step-6--visualization)
   - [Step 7 — Prediction / Inference](#step-7--prediction--inference)
6. [Why We Chose Each Thing](#why-we-chose-each-thing)
7. [Outputs You Get](#outputs-you-get)
8. [How to Run](#how-to-run)
9. [Glossary](#glossary)

---

## What This Project Does

A farmer looks at a leaf — they can tell if it's diseased just by looking at the spots,
color patterns, and texture. This project teaches a computer to do the same thing.

We build a **CNN (Convolutional Neural Network)** that:
- Takes a photo of a plant leaf as input
- Analyses the visual patterns (spots, lesions, colors, textures)
- Outputs which of **38 possible diseases** the leaf has (or says it's healthy)
- Shows you **exactly where on the leaf** it found the disease (Grad-CAM)

### What makes this different from a basic project?

Most students just do this:
```python
base_model = MobileNetV2(weights='imagenet')  # download someone else's brain
predictions = Dense(38)(base_model.output)    # stick an output layer on top
```

**We did NOT do that.** We built the entire neural network ourselves — every
Conv2D layer, every BatchNormalization, every MaxPool — designed by us, trained
from random initialization on our dataset. We understand every single line.

---

## The Dataset

**Name:** PlantVillage Dataset
**Source:** [frontiersin.org paper](https://doi.org/10.3389/fpls.2016.01419) by Mohanty, Hughes & Salathé (2016)

| Fact | Value |
|---|---|
| Total images | **54,305** |
| Number of classes | **38** (diseases + healthy) |
| Plant species covered | Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato |
| Image format | JPEG, color |
| Image size (after resize) | 128 × 128 pixels |

### Folder structure of the dataset
```
raw/color/
├── Apple___Apple_scab/          (class 0)
├── Apple___Black_rot/           (class 1)
├── Apple___Cedar_apple_rust/    (class 2)
├── Apple___healthy/             (class 3)
├── Blueberry___healthy/         (class 4)
├── ... (38 folders total)
└── Tomato___Tomato_mosaic_virus/(class 37)
```
Each subfolder = one class. The folder name IS the label.

---

## Project Folder Structure

```
plant-disease-cnn/
│
├── 📄 config.py                 ← ALL settings live here (one place to change anything)
├── 📄 train.py                  ← Run this to train the model
├── 📄 predict.py                ← Run this to diagnose a leaf image
├── 📄 requirements.txt          ← Python packages needed
│
├── 📁 raw/color/                ← Your dataset (54,305 images, 38 folders)
├── 📁 saved_models/             ← Created after training (model saved here)
├── 📁 outputs/                  ← Created after training (plots, Grad-CAM images)
│
├── 📁 data/
│   ├── loader.py                ← Loads images from disk using tf.data
│   └── augmentation.py         ← Augmentation pipeline (flip, rotate, zoom, brightness)
│
├── 📁 model/
│   ├── blocks.py                ← The ConvBlock: one reusable building block
│   └── network.py               ← Full CNN assembled from 4 ConvBlocks
│
├── 📁 training/
│   ├── callbacks.py             ← Smart training controls (save, stop, adjust LR)
│   └── trainer.py               ← compile_model(), train(), evaluate()
│
├── 📁 evaluation/
│   ├── metrics.py               ← Per-class precision, recall, F1, confusion matrix
│   └── gradcam.py               ← Grad-CAM heatmap (where did the model look?)
│
├── 📁 visualization/
│   └── plots.py                 ← Training curves + per-class accuracy bar chart
│
└── 📁 inference/
    └── predictor.py             ← Load model, preprocess image, get top-3 predictions
```

---

## Complete Flow

This is how data moves through the entire system when you run `python train.py`:

```
DISK (54,305 images in 38 folders)
         │
         │  loader.py reads folders, assigns labels 0–37
         ▼
RAW DATASET (tf.data pipeline)
         │
         │  80% → TRAIN SET     20% → VALIDATION SET
         │
         ▼                          ▼
  augmentation.py            rescaling only
  (flip, rotate, zoom,       (no augmentation —
   brightness, rescale)       we evaluate on
                               real conditions)
         │                          │
         └──────────┬───────────────┘
                    ▼
         model/network.py
    ┌────────────────────────────────────────┐
    │  Input Layer  (128 × 128 × 3)          │
    │       ↓                                │
    │  ConvBlock 1  (32 filters)  64×64×32   │
    │       ↓                                │
    │  ConvBlock 2  (64 filters)  32×32×64   │
    │       ↓                                │
    │  ConvBlock 3  (128 filters) 16×16×128  │
    │       ↓                                │
    │  ConvBlock 4  (256 filters)  8×8×256   │
    │       ↓                                │
    │  GlobalAveragePooling2D     → 256      │
    │       ↓                                │
    │  Dense(512, ReLU)           → 512      │
    │       ↓                                │
    │  Dropout(0.5)                          │
    │       ↓                                │
    │  Dense(38, Softmax)         → 38       │
    └────────────────────────────────────────┘
                    │
                    ▼
         38 probabilities (e.g., [0.02, 0.01, 0.87, ...])
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
   training/trainer.py    evaluation/
   (loss computed,         gradcam.py
    backprop,              metrics.py
    weights updated)       plots.py
```

---

## Step-by-Step Deep Dive

---

### Step 1 — Data Loading (`data/loader.py`)

**What it does:**
Reads all 54,305 images from `raw/color/` and splits them into a training set and validation set.

**How it works:**
```python
train_ds = image_dataset_from_directory(
    "raw/color/",
    validation_split=0.2,   # 80% train, 20% validation
    subset="training",
    seed=42,                 # same split every run
    image_size=(128, 128),   # resize every image to this
    batch_size=16,
    label_mode="categorical" # one-hot labels: [0,0,1,0,...,0]
)
```

**What is one-hot encoding?**
Instead of label = `2` (a single number), one-hot gives:
```
class 0 → [1, 0, 0, 0, ..., 0]
class 2 → [0, 0, 1, 0, ..., 0]
class 37 → [0, 0, 0, ..., 0, 1]
```
This is required for Categorical Cross-Entropy loss.

**Performance trick — Prefetch:**
```python
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
```
While the model is training on batch #5, the CPU is already loading batch #6 in the background.
This prevents the GPU/CPU from sitting idle waiting for data.

---

### Step 2 — Data Augmentation (`data/augmentation.py`)

**What it does:**
Artificially creates variations of training images so the model doesn't memorize exact photos.

**Why is this needed?**
A model that sees the same 54,305 images repeatedly will memorize them (overfitting).
In real life, a farmer's photo might be at a different angle, lighting, or zoom than the training data.
Augmentation prepares the model for these variations.

| Augmentation | What it does | Why |
|---|---|---|
| `RandomFlip` | Mirror image horizontally/vertically | A diseased leaf looks the same upside-down |
| `RandomRotation(0.2)` | Rotate up to ±72° | Leaves photographed at random angles |
| `RandomZoom(0.1)` | Zoom in/out up to 10% | Camera distance varies |
| `RandomBrightness(0.1)` | Brighten/darken up to 10% | Different lighting conditions |
| `Rescaling(1/255)` | Divide pixels by 255 | Normalize [0,255] → [0.0,1.0] for neural nets |

> **Important:** Augmentation is applied to TRAINING data only.
> Validation images are only rescaled — never augmented.
> (We evaluate on realistic, unmodified images.)

---

### Step 3 — CNN Architecture (The Heart of the Project)

This is what the teacher was asking for. We designed this ourselves.

#### What is a ConvBlock? (`model/blocks.py`)

A ConvBlock is one repeatable unit. We use it 4 times:

```
ConvBlock(filters):
    Conv2D(filters, 3×3)     ← slide a filter across the image
    BatchNormalization()      ← stabilize the values
    ReLU()                    ← kill negatives, add non-linearity
    Conv2D(filters, 3×3)     ← second convolution (deeper features)
    BatchNormalization()
    ReLU()
    MaxPooling2D(2×2)        ← shrink spatial size by half
    Dropout(0.25)            ← randomly turn off 25% of neurons
```

**Why two Conv2D layers per block?**
One 3×3 filter sees a 3×3 region. Two stacked 3×3 filters effectively see a 5×5 region,
but with fewer parameters than a single 5×5 filter. More depth = more power, lower cost.

#### Each Layer Explained

| Layer | Math / Behaviour | Purpose |
|---|---|---|
| **Conv2D** | Slides a learned filter (3×3 matrix) over the image. At each position: element-wise multiply + sum = one output value | Detects patterns (edges, spots, textures) |
| **BatchNormalization** | For each feature map: subtract mean, divide by std dev, then scale+shift with learned γ, β | Prevents values from exploding/vanishing; speeds up training |
| **ReLU** | `f(x) = max(0, x)` — sets negatives to zero | Adds non-linearity so the network isn't just linear math |
| **MaxPooling2D** | Splits feature map into 2×2 tiles, keeps only the maximum in each | Shrinks image, retains dominant features, slight translation invariance |
| **Dropout** | During training: randomly zeros out `rate` fraction of neurons | Forces neurons to learn independently; prevents co-adaptation (overfitting) |

#### Full Architecture (`model/network.py`)

```
Layer                    Output Shape      What it learns
─────────────────────────────────────────────────────────────
Input                    128 × 128 × 3     Raw RGB pixel values

ConvBlock 1 (32 filters)  64 × 64 × 32    Simple edges, color gradients
ConvBlock 2 (64 filters)  32 × 32 × 64    Textures, repeated patterns
ConvBlock 3 (128 filters) 16 × 16 × 128   Spots, lesions, shapes
ConvBlock 4 (256 filters)  8 ×  8 × 256   Complex disease signatures

GlobalAveragePooling2D        256          "How present is each feature globally?"
Dense(512, ReLU)              512          Learned combinations of features
Dropout(0.5)                  512          Regularization
Dense(38, Softmax)             38          One probability per disease class
─────────────────────────────────────────────────────────────
Total trainable parameters: ~1.3 Million
```

**Why GlobalAveragePooling instead of Flatten?**

| Method | Parameters added | Risk |
|---|---|---|
| Flatten(8×8×256) → Dense(512) | **8.4 Million** | Very prone to overfitting |
| GlobalAveragePooling → Dense(512) | **131 Thousand** | Much more efficient |

GAP averages each of the 256 feature maps into a single number — "how strongly was this feature (e.g., brown spots) present anywhere in the image?"

**What does Softmax output look like?**
```
Input image: Tomato leaf with early blight

Output: [0.01, 0.02, 0.00, ..., 0.87, 0.04, ...]
                                 ↑
                    "Tomato___Early_blight" = 87% confidence
All 38 values sum to exactly 1.0
```

---

### Step 4 — Training Pipeline

#### Compilation (`training/trainer.py → compile_model()`)

Before training, we attach three things to the model:

**1. Optimizer: Adam**
Adam (Adaptive Moment Estimation) adjusts the learning rate automatically per parameter.
- Weights that don't update often get a bigger learning rate (they need more push)
- Weights that update constantly get a smaller one (they're already converging)
- This makes training much faster than basic SGD (Stochastic Gradient Descent)

**2. Loss: Categorical Cross-Entropy**
Measures how wrong the model's predictions are.

For a correct class with probability `p`:
```
Loss = -log(p)

If model predicts 99% correct → Loss = -log(0.99) ≈ 0.01  (very low, good)
If model predicts  1% correct → Loss = -log(0.01) ≈ 4.6   (very high, bad)
```
The optimizer's job is to minimize this value across all 54,305 training images.

**3. Metric: Accuracy**
Printed each epoch so we can see progress.

#### The Training Loop (`training/trainer.py → train()`)

Each epoch (one complete pass over training data):

```
For each batch of 16 images:
    ┌─────────────────────────────────────────────────────┐
    │ 1. FORWARD PASS                                     │
    │    Images → CNN → 38 probabilities                  │
    │                                                     │
    │ 2. COMPUTE LOSS                                     │
    │    How wrong were the predictions? (cross-entropy)  │
    │                                                     │
    │ 3. BACKWARD PASS (Backpropagation)                  │
    │    Compute gradient: d(loss)/d(every weight)        │
    │    "If I increase this weight, does loss go up/down?"│
    │                                                     │
    │ 4. UPDATE WEIGHTS (Adam)                            │
    │    Nudge each weight in the direction that          │
    │    reduces the loss                                 │
    └─────────────────────────────────────────────────────┘

After all batches → Evaluate on validation set (no weight updates here)
Print: Epoch 1/25 - loss: 3.21 - accuracy: 0.42 - val_accuracy: 0.61
```

#### Smart Callbacks (`training/callbacks.py`)

Three automatic helpers that run at the end of each epoch:

| Callback | What it does | Why |
|---|---|---|
| `ModelCheckpoint` | Saves model if val_accuracy improved | Keep the best version, even if later epochs overfit |
| `EarlyStopping(patience=5)` | Stops training if no improvement for 5 epochs | Saves hours of wasted computation |
| `ReduceLROnPlateau(patience=3)` | Halves the learning rate when stuck | Fine-tunes after coarse learning |

---

### Step 5 — Evaluation & Grad-CAM

#### Metrics (`evaluation/metrics.py`)

After training, we evaluate on the validation set (images the model never saw during training).

**Classification Report:**
```
                                    precision  recall  f1-score
Apple___Apple_scab                    0.91      0.89     0.90
Apple___Black_rot                     0.95      0.93     0.94
...
Tomato___Tomato_mosaic_virus          0.88      0.91     0.89
```

- **Precision:** Of all images predicted as "Apple Scab", what % actually were?
- **Recall:** Of all real "Apple Scab" images, what % did we correctly identify?
- **F1 Score:** Harmonic mean of precision and recall (the most balanced single metric)

**Confusion Matrix:**
A 38×38 grid. Row = true class, Column = predicted class.
Bright diagonal = correct predictions. Off-diagonal = what was confused with what.

#### Grad-CAM (`evaluation/gradcam.py`)

This is the most impressive visual output. It answers:
**"WHERE in the image did the CNN look to make its decision?"**

**How it works mathematically:**
```
1. Run forward pass → get prediction (e.g., "Tomato Early Blight")
2. Take the score of that class
3. Compute gradient of that score w.r.t. the last conv layer's feature maps
   → "Which pixels in the feature map most influenced this prediction?"
4. Average the gradients over each feature map → importance weight per filter
5. Weighted sum of feature maps → raw 8×8 heatmap
6. Apply ReLU → keep only positive influences
7. Upsample 8×8 → 128×128 using bilinear interpolation
8. Apply JET colormap → Blue=low, Green=medium, Red=high attention
9. Blend with original image (55% original + 45% heatmap)
```

**Result:** Red = "the model focused here to detect disease"

If red zones align with disease spots on the leaf → the model genuinely learned
to detect diseases, not just memorize image backgrounds.

---

### Step 6 — Visualization (`visualization/plots.py`)

**Training Curves (`outputs/training_curves.png`):**

Two subplots side by side:
- Left: Train Accuracy vs Validation Accuracy over epochs
- Right: Train Loss vs Validation Loss over epochs

Reading the curves tells you:
- Train accuracy rising = model is learning ✅
- Val accuracy close to train = no overfitting ✅
- Val accuracy much lower than train = overfitting ❌
- Both curves plateau = EarlyStopping fired, training stopped

**Per-Class F1 Chart (`outputs/per_class_accuracy.png`):**

Horizontal bar chart of 38 disease classes sorted best → worst.
- 🟢 Green bars = top 10 easiest classes
- 🔴 Red bars = bottom 10 hardest classes
- This tells you which diseases your model struggles with most

---

### Step 7 — Prediction / Inference

#### Predictor (`inference/predictor.py`)

When you run `python predict.py --image leaf.jpg`, here's what happens:

```
1. Load trained model from saved_models/plant_disease_cnn_best.keras
2. Load class names from saved_models/class_names.json
3. Load the image from disk
4. Resize to 128×128 pixels
5. Normalize: divide by 255.0 → values in [0.0, 1.0]
6. Add batch dimension: (128,128,3) → (1,128,128,3)
7. model.predict() → array of 38 probabilities
8. Take top 3 highest probabilities
9. Print diagnosis report
```

**Sample output:**
```
==========================================================
   🌿  PLANT DISEASE DIAGNOSIS REPORT  🌿
   Built-from-scratch CNN · PlantVillage Dataset
==========================================================
   Image   : my_tomato_leaf.jpg
   Status  : DISEASE DETECTED ⚠️
----------------------------------------------------------
   Plant   : Tomato
   Disease : Early blight
   Confidence : 87.3%
----------------------------------------------------------
   Top 3 Predictions:
   [1] ████████████████░░░░  87.3%  Tomato — Early blight
   [2] ██░░░░░░░░░░░░░░░░░░   9.1%  Tomato — Late blight
   [3] █░░░░░░░░░░░░░░░░░░░   2.1%  Tomato — healthy
==========================================================
```

---

## Why We Chose Each Thing

| Decision | What we chose | Why not the alternative? |
|---|---|---|
| **Image size** | 128×128 | 224×224 takes 3× more memory and is too slow on CPU |
| **4 ConvBlocks** | 32→64→128→256 filters | Fewer blocks = not enough feature depth; more = too slow on CPU |
| **GlobalAveragePooling** | Instead of Flatten | Flatten creates 8.4M extra parameters → massive overfitting risk |
| **BatchNormalization** | Before ReLU | Without it, activations grow unbounded → training becomes unstable |
| **Adam optimizer** | Instead of SGD | Adam is self-tuning; SGD needs careful manual LR tuning |
| **Dropout(0.5)** | After Dense(512) | Standard for FC layers; forces robust feature learning |
| **EarlyStopping** | patience=5 | Prevents wasted time and overfitting after the model peaks |
| **he_normal init** | For Conv2D weights | Best weight initialization for layers followed by ReLU |

---

## Outputs You Get

After running `python train.py`, these files are created:

```
saved_models/
├── plant_disease_cnn_best.keras   ← the trained model (use this for predictions)
└── class_names.json               ← the 38 class names in order

outputs/
├── training_curves.png            ← accuracy & loss over epochs
├── per_class_accuracy.png         ← F1 bar chart for all 38 classes
├── classification_report.txt      ← precision/recall/F1 for every class
├── confusion_matrix.png           ← 38×38 mistake analysis heatmap
└── gradcam_<classname>.png        ← Grad-CAM heatmap overlay
```

---

## How to Run

### 1. Install dependencies (one time only)

```bash
cd "/home/sharan/major project/PlantVillage-Dataset/plant-disease-cnn"
../venv/bin/pip install -r requirements.txt
```

### 2. Train the model

```bash
../venv/bin/python train.py
```

> ⏱️ **Expected time:** Each epoch takes ~5–15 minutes on CPU.
> EarlyStopping will likely stop around epoch 15–20.
> Total training time: **2–4 hours** (let it run overnight).

### 3. Diagnose a leaf image

```bash
# Basic prediction
../venv/bin/python predict.py --image /path/to/leaf.jpg

# With Grad-CAM heatmap
../venv/bin/python predict.py --image /path/to/leaf.jpg --gradcam
```

---

## Glossary

| Term | Simple Explanation |
|---|---|
| **CNN** | A neural network specifically designed for images; uses filters to detect visual patterns |
| **Filter / Kernel** | A small matrix (e.g., 3×3) of learnable numbers that slides across the image to detect one specific pattern |
| **Feature Map** | The output of applying one filter to an image; highlights where that pattern was found |
| **Epoch** | One complete pass of the model over all training images |
| **Batch** | A small group of images processed together before one weight update (batch size = 16) |
| **Backpropagation** | The algorithm that computes how much each weight contributed to the error, so we can adjust it |
| **Gradient** | The direction and magnitude in which a weight should change to reduce loss |
| **Overfitting** | Model memorizes training data perfectly but fails on new images |
| **Dropout** | Randomly turning off neurons during training to prevent overfitting |
| **BatchNorm** | Normalizing layer outputs to stabilize and speed up training |
| **Softmax** | Converts raw scores → probabilities that sum to 1.0 |
| **Cross-Entropy** | Loss function that measures how wrong probability predictions are |
| **Adam** | Smart optimizer that adjusts learning rate per weight automatically |
| **Grad-CAM** | Technique to visualize which pixels in the image influenced the model's decision |
| **Transfer Learning** | Using a model pre-trained on one task for a different task (what we did NOT do) |
| **Pre-trained weights** | Weights already learned from another dataset (e.g., ImageNet) — we started from random instead |

---

*Built with TensorFlow 2.20 · PlantVillage Dataset · Custom CNN Architecture*
*Authors: Sharan & team — Major Project*
