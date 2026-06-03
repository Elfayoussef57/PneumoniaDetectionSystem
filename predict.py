"""
=============================================================================
PNEUMONIA PREDICTOR — Single Image Inference
  Takes one X-ray image as input and returns predictions from:
    • SVM + HOG
    • Random Forest + HOG
    • KNN + HOG
    • CNN (deep learning)

Usage:
    python predict.py --image path/to/xray.jpeg
    python predict.py --image path/to/xray.jpeg --model cnn
    python predict.py --image path/to/xray.jpeg --model all
=============================================================================
"""

import argparse
import sys
import warnings
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Config (must match training scripts) ──────────────────────────────────
IMG_SIZE_ML  = 64
IMG_SIZE_CNN = 128
CLASSES      = ["NORMAL", "PNEUMONIA"]
MODEL_DIR    = Path("models")

# Colors for display
COLOR_NORMAL    = "#2196F3"   # blue
COLOR_PNEUMONIA = "#F44336"   # red
COLOR_UNSURE    = "#FF9800"   # orange (50-60% confidence)


# ─────────────────────────────────────────────
# IMAGE PREPROCESSING
# ─────────────────────────────────────────────

def load_and_preprocess(image_path: str, img_size: int, grayscale: bool = True) -> np.ndarray:
    """
    Read an image from disk, resize, and normalize to [0, 1].
    Returns shape (img_size, img_size) for grayscale.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img  = cv2.imread(str(path), flag)
    if img is None:
        raise ValueError(f"Could not read image: {image_path} — unsupported format?")

    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    return img.astype(np.float32) / 255.0


def extract_hog_features_single(img: np.ndarray) -> np.ndarray:
    """Extract HOG features from a single (H, W) float image."""
    from skimage.feature import hog
    uint8_img = (img * 255).astype(np.uint8)
    fd = hog(
        uint8_img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return fd.reshape(1, -1).astype(np.float32)


# ─────────────────────────────────────────────
# MODEL LOADERS
# ─────────────────────────────────────────────

def load_ml_models() -> dict:
    """Load saved scikit-learn models from disk."""
    import joblib
    models = {}
    files  = {
        "SVM + HOG": MODEL_DIR / "svm_hog.pkl",
        "RF + HOG":  MODEL_DIR / "rf_hog.pkl",
        "KNN + HOG": MODEL_DIR / "knn_hog.pkl",
    }
    for name, path in files.items():
        if path.exists():
            models[name] = joblib.load(str(path))
            print(f"  ✓ Loaded {name}  ({path})")
        else:
            print(f"  ✗ Not found: {path}  (skipping {name})")
    return models


def load_cnn_model():
    """Load saved Keras CNN model from disk."""
    import tensorflow as tf
    path = MODEL_DIR / "best_cnn.keras"
    if not path.exists():
        # fallback to older .h5 format
        path = MODEL_DIR / "best_cnn.h5"
    if not path.exists():
        print(f"  ✗ CNN model not found at {MODEL_DIR}/best_cnn.keras")
        return None
    model = tf.keras.models.load_model(str(path))
    print(f"  ✓ Loaded CNN  ({path})")
    return model


# ─────────────────────────────────────────────
# PREDICTION FUNCTIONS
# ─────────────────────────────────────────────

def predict_ml(models: dict, image_path: str) -> dict:
    """Run all ML models on a single image."""
    img  = load_and_preprocess(image_path, IMG_SIZE_ML)
    feat = extract_hog_features_single(img)

    results = {}
    for name, model in models.items():
        prob      = model.predict_proba(feat)[0]          # [p_normal, p_pneumonia]
        label_idx = int(np.argmax(prob))
        results[name] = {
            "label":       CLASSES[label_idx],
            "confidence":  float(prob[label_idx]),
            "p_normal":    float(prob[0]),
            "p_pneumonia": float(prob[1]),
        }
    return results


def predict_cnn(model, image_path: str) -> dict:
    """Run CNN on a single image."""
    import tensorflow as tf
    img = load_and_preprocess(image_path, IMG_SIZE_CNN)
    # CNN expects (batch, H, W, channels)
    img_tensor = img[np.newaxis, :, :, np.newaxis]        # (1, H, W, 1)

    prob_pneumonia = float(model.predict(img_tensor, verbose=0)[0][0])
    prob_normal    = 1.0 - prob_pneumonia
    label_idx      = int(prob_pneumonia >= 0.5)

    return {
        "label":       CLASSES[label_idx],
        "confidence":  prob_pneumonia if label_idx == 1 else prob_normal,
        "p_normal":    prob_normal,
        "p_pneumonia": prob_pneumonia,
    }


# ─────────────────────────────────────────────
# DISPLAY
# ─────────────────────────────────────────────

def print_result(name: str, res: dict):
    label = res["label"]
    conf  = res["confidence"] * 100
    bar   = "█" * int(conf / 5) + "░" * (20 - int(conf / 5))
    color_tag = "🔴" if label == "PNEUMONIA" else "🟢"
    print(f"  {color_tag}  {name:<18}  →  {label:<10}  {conf:5.1f}%  [{bar}]")
    print(f"              Normal: {res['p_normal']*100:5.1f}%   Pneumonia: {res['p_pneumonia']*100:5.1f}%")


def plot_results(image_path: str, all_results: dict):
    """Display the X-ray image alongside a bar chart of predictions."""
    img_display = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img_display is None:
        return

    n_models = len(all_results)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                              gridspec_kw={"width_ratios": [1, 1.6]})

    # Left panel — X-ray image
    axes[0].imshow(img_display, cmap="gray")
    axes[0].set_title("Input X-Ray", fontsize=13, fontweight="bold")
    axes[0].axis("off")

    # Right panel — confidence bars
    model_names   = list(all_results.keys())
    p_pneumonias  = [all_results[n]["p_pneumonia"] for n in model_names]
    bar_colors    = [COLOR_PNEUMONIA if p >= 0.5 else COLOR_NORMAL for p in p_pneumonias]

    y_pos = np.arange(len(model_names))
    bars  = axes[1].barh(y_pos, p_pneumonias, color=bar_colors,
                         edgecolor="white", height=0.5, linewidth=1.5)

    # Confidence labels
    for bar, p in zip(bars, p_pneumonias):
        axes[1].text(min(p + 0.02, 0.92), bar.get_y() + bar.get_height() / 2,
                     f"{p*100:.1f}%", va="center", fontsize=10, fontweight="bold")

    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(model_names, fontsize=11)
    axes[1].set_xlim(0, 1.05)
    axes[1].set_xlabel("Pneumonia Probability", fontsize=11)
    axes[1].set_title("Model Predictions", fontsize=13, fontweight="bold")
    axes[1].axvline(0.5, color="black", linestyle="--", linewidth=1.2, alpha=0.6,
                    label="Decision threshold (0.5)")
    axes[1].legend(fontsize=9, loc="lower right")
    axes[1].grid(axis="x", alpha=0.3)

    # Legend patches
    patch_normal    = mpatches.Patch(color=COLOR_NORMAL,    label="NORMAL")
    patch_pneumonia = mpatches.Patch(color=COLOR_PNEUMONIA, label="PNEUMONIA")
    axes[1].legend(handles=[patch_normal, patch_pneumonia, 
                             mpatches.Patch(color="none", label="")],
                   fontsize=9, loc="lower right")

    plt.suptitle(f"Pneumonia Detection — {Path(image_path).name}",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("prediction_result.png", dpi=130, bbox_inches="tight")
    plt.show()
    print("\n  → Result plot saved: prediction_result.png")


# ─────────────────────────────────────────────
# MAJORITY VOTE
# ─────────────────────────────────────────────

def majority_vote(all_results: dict) -> str:
    votes = [1 if res["label"] == "PNEUMONIA" else 0
             for res in all_results.values()]
    return "PNEUMONIA" if sum(votes) > len(votes) / 2 else "NORMAL"


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Predict whether a chest X-ray shows pneumonia."
    )
    parser.add_argument(
        "--image", required=True,
        help="Path to the X-ray image (.jpeg / .jpg / .png)"
    )
    parser.add_argument(
        "--model", default="all",
        choices=["all", "ml", "cnn", "svm", "rf", "knn"],
        help="Which model(s) to use for prediction (default: all)"
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip the matplotlib result plot"
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  PNEUMONIA DETECTOR — Single Image Prediction")
    print("=" * 60)
    print(f"  Image : {args.image}")
    print(f"  Model : {args.model}")
    print()

    if not MODEL_DIR.exists():
        raise SystemExit(
            f"\n[ERROR] Model directory '{MODEL_DIR}' not found.\n"
            "  Train models first:\n"
            "    python ml_pipeline.py\n"
            "    python dl_pipeline.py\n"
        )

    all_results = {}
    use_ml  = args.model in ("all", "ml", "svm", "rf", "knn")
    use_cnn = args.model in ("all", "cnn")

    # ── ML models ────────────────────────────
    if use_ml:
        print("  Loading ML models …")
        ml_models = load_ml_models()

        # Filter to specific model if requested
        if args.model == "svm":
            ml_models = {k: v for k, v in ml_models.items() if "SVM" in k}
        elif args.model == "rf":
            ml_models = {k: v for k, v in ml_models.items() if "RF"  in k}
        elif args.model == "knn":
            ml_models = {k: v for k, v in ml_models.items() if "KNN" in k}

        if ml_models:
            print("\n  Running ML predictions …")
            ml_results = predict_ml(ml_models, args.image)
            all_results.update(ml_results)
        else:
            print("  No ML models available.")

    # ── CNN model ────────────────────────────
    if use_cnn:
        print("\n  Loading CNN model …")
        cnn_model = load_cnn_model()
        if cnn_model is not None:
            print("  Running CNN prediction …")
            cnn_result = predict_cnn(cnn_model, args.image)
            all_results["CNN"] = cnn_result

    if not all_results:
        raise SystemExit("\n[ERROR] No models could be loaded. Run the training scripts first.")

    # ── Print results ─────────────────────────
    print("\n" + "─" * 60)
    print("  RESULTS")
    print("─" * 60)
    for name, res in all_results.items():
        print_result(name, res)

    # ── Majority vote (if multiple models) ───
    if len(all_results) > 1:
        verdict = majority_vote(all_results)
        emoji   = "🔴" if verdict == "PNEUMONIA" else "🟢"
        print()
        print("─" * 60)
        print(f"  {emoji}  FINAL VERDICT (majority vote):  {verdict}")
        print("─" * 60)
    else:
        name, res = next(iter(all_results.items()))
        emoji = "🔴" if res["label"] == "PNEUMONIA" else "🟢"
        print()
        print("─" * 60)
        print(f"  {emoji}  PREDICTION:  {res['label']}  ({res['confidence']*100:.1f}% confidence)")
        print("─" * 60)

    print("\n  ⚠  This tool is for research purposes only.")
    print("     Always consult a qualified physician for diagnosis.\n")

    # ── Plot ──────────────────────────────────
    if not args.no_plot:
        plot_results(args.image, all_results)


if __name__ == "__main__":
    main()
