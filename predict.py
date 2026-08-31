"""
predict.py — Interactive Plant Disease Diagnosis
=================================================
Run this file and it will ask you for the image path step by step.

Usage (interactive — recommended):
    python predict.py

Usage (direct — for advanced use):
    python predict.py --image path/to/leaf.jpg --gradcam
"""

import os
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import config
from inference.predictor import (
    load_model_and_classes,
    predict,
    format_diagnosis_report,
)
from evaluation.gradcam import generate_gradcam


def ask_image_path() -> str:
    """
    Interactively asks the user for an image path.
    Keeps asking until a valid image file is given.
    """
    print("\n" + "=" * 55)
    print("  🌿  PLANT DISEASE DIAGNOSIS — CNN From Scratch")
    print("=" * 55)
    print("\nEnter the path to your leaf image.")
    print("Examples:")
    print("  raw/color/Apple___Black_rot/<image_name>.JPG")
    print("  /full/path/to/your/leaf.jpg")
    print()

    while True:
        image_path = input("📷  Image path: ").strip().strip("'\"")

        if not image_path:
            print("[!] Path cannot be empty. Please try again.\n")
            continue

        if not os.path.exists(image_path):
            print(f"[!] File not found: {image_path}")
            print("    Check the path and try again.\n")
            continue

        # Check it's an image file
        ext = os.path.splitext(image_path)[1].lower()
        if ext not in (".jpg", ".jpeg", ".png", ".bmp"):
            print(f"[!] '{ext}' is not a supported image format.")
            print("    Supported: .jpg, .jpeg, .png, .bmp\n")
            continue

        return image_path


def ask_gradcam() -> bool:
    """Asks the user if they want a Grad-CAM heatmap."""
    print()
    while True:
        answer = input("🔍  Generate Grad-CAM heatmap? (yes/no): ").strip().lower()
        if answer in ("yes", "y"):
            return True
        elif answer in ("no", "n", ""):
            return False
        else:
            print("    Please type 'yes' or 'no'.")


def parse_args():
    """Parse command-line args (for direct/advanced usage)."""
    parser = argparse.ArgumentParser(
        description="Plant Disease Diagnosis — Custom CNN from Scratch",
        add_help=True,
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to the leaf image (optional — will ask interactively if not given)",
    )
    parser.add_argument(
        "--gradcam", action="store_true", default=False,
        help="Generate Grad-CAM heatmap",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Get image path ─────────────────────────────────────────
    if args.image:
        # Provided via command line
        image_path = args.image.strip().strip("'\"")
        if not os.path.exists(image_path):
            print(f"\n[ERROR] Image not found: {image_path}\n")
            sys.exit(1)
        use_gradcam = args.gradcam
    else:
        # Interactive mode — ask the user
        image_path = ask_image_path()
        use_gradcam = ask_gradcam()

    # ── Load model ─────────────────────────────────────────────
    print("\n[*] Loading model (this takes ~30 seconds on first run)...")
    model, class_names = load_model_and_classes()

    # ── Predict ────────────────────────────────────────────────
    print("[*] Running diagnosis...")
    predictions = predict(image_path, model, class_names)

    # ── Print report ───────────────────────────────────────────
    report = format_diagnosis_report(image_path, predictions)
    print(report)

    # ── Grad-CAM ───────────────────────────────────────────────
    if use_gradcam:
        print("[GradCAM] Generating heatmap...")
        gradcam_path = generate_gradcam(
            image_path=image_path,
            model=model,
            class_names=class_names,
            last_conv_layer_name="block4_conv2",
            save_dir=config.OUTPUTS_DIR,
        )
        print(f"\n✅  Heatmap saved to:\n    {gradcam_path}\n")
    else:
        print("Tip: Run again and type 'yes' to generate a Grad-CAM heatmap.\n")

    # ── Ask to predict another image ───────────────────────────
    if not args.image:   # only in interactive mode
        print()
        again = input("🔁  Predict another image? (yes/no): ").strip().lower()
        if again in ("yes", "y"):
            main()


if __name__ == "__main__":
    main()
