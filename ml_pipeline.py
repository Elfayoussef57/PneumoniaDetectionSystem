"""
=============================================================================
ML PIPELINE — Pneumonia Detection
  Models: SVM, Random Forest, KNN (with HOG features)
=============================================================================
"""

# ─────────────────────────────────────────────
# DEPENDENCIES
# ─────────────────────────────────────────────
import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve
)
from sklearn.utils.class_weight import compute_class_weight
from skimage.feature import hog
import joblib

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DATA_DIR    = Path("chest_xray")
IMG_SIZE_ML = 64
CLASSES     = ["NORMAL", "PNEUMONIA"]
LABEL_MAP   = {"NORMAL": 0, "PNEUMONIA": 1}


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def load_images(split: str, img_size: int, grayscale: bool = True):
    """Load all images from a split (train/val/test) into arrays."""
    images, labels = [], []
    for cls in CLASSES:
        folder = DATA_DIR / split / cls
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")
        for img_path in sorted(folder.glob("*.jpeg")):
            flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
            img  = cv2.imread(str(img_path), flag)
            if img is None:
                continue
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
            images.append(img)
            labels.append(LABEL_MAP[cls])
    return np.array(images, dtype=np.float32) / 255.0, np.array(labels, dtype=np.int32)


def extract_hog_features(images: np.ndarray) -> np.ndarray:
    """Extract HOG descriptors from a batch of grayscale images."""
    feats = []
    for img in images:
        uint8_img = (img * 255).astype(np.uint8)
        fd = hog(
            uint8_img,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            feature_vector=True,
        )
        feats.append(fd)
    return np.array(feats, dtype=np.float32)


def metrics_dict(y_true, y_pred, y_prob=None):
    """Return a dict of evaluation metrics."""
    d = {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "F1-Score":  f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        d["AUC-ROC"] = roc_auc_score(y_true, y_prob)
    return d


def plot_confusion(y_true, y_pred, title, ax):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASSES)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title, fontsize=12, fontweight="bold")


# ─────────────────────────────────────────────
# ML PIPELINE
# ─────────────────────────────────────────────

def run_ml_pipeline():
    print("\n" + "=" * 60)
    print("  MACHINE LEARNING MODELS  (SVM · RF · KNN + HOG)")
    print("=" * 60)

    # Load + preprocess
    print("  Loading training data …")
    X_train_raw, y_train = load_images("train", IMG_SIZE_ML)
    X_val_raw,   y_val   = load_images("val",   IMG_SIZE_ML)
    X_test_raw,  y_test  = load_images("test",  IMG_SIZE_ML)

    # Merge val into train (val set is very small)
    X_train_raw = np.concatenate([X_train_raw, X_val_raw])
    y_train     = np.concatenate([y_train, y_val])

    print(f"  Train: {X_train_raw.shape[0]} images | Test: {X_test_raw.shape[0]} images")

    # HOG feature extraction
    print("  Extracting HOG features …")
    t0 = time.time()
    X_train_hog = extract_hog_features(X_train_raw)
    X_test_hog  = extract_hog_features(X_test_raw)
    print(f"  HOG done in {time.time()-t0:.1f}s | feature dim: {X_train_hog.shape[1]}")

    # Class weights
    classes_          = np.unique(y_train)
    cw_values         = compute_class_weight("balanced", classes=classes_, y=y_train)
    class_weight_dict = dict(zip(classes_, cw_values))
    print(f"  Class weights: {class_weight_dict}")

    results_ml = {}

    # ── SVM (HOG features) ────────────────────
    print("\n  [SVM + HOG]  Training …")
    t0 = time.time()
    svm_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(kernel="rbf", class_weight="balanced",
                       probability=True, random_state=42)),
    ])
    param_grid_svm = {"svm__C": [0.1, 1, 10], "svm__gamma": ["scale", "auto"]}
    gs_svm = GridSearchCV(svm_pipe, param_grid_svm, cv=3, scoring="f1",
                          n_jobs=-1, verbose=0)
    gs_svm.fit(X_train_hog, y_train)
    best_svm = gs_svm.best_estimator_
    y_pred   = best_svm.predict(X_test_hog)
    y_prob   = best_svm.predict_proba(X_test_hog)[:, 1]
    results_ml["SVM + HOG"] = {"model": best_svm, "y_pred": y_pred, "y_prob": y_prob,
                                **metrics_dict(y_test, y_pred, y_prob)}
    print(f"  Best params: {gs_svm.best_params_} | Time: {time.time()-t0:.1f}s")
    print(f"  F1={results_ml['SVM + HOG']['F1-Score']:.4f} | "
          f"Recall={results_ml['SVM + HOG']['Recall']:.4f} | "
          f"AUC={results_ml['SVM + HOG']['AUC-ROC']:.4f}")

    # ── Random Forest (HOG features) ──────────
    print("\n  [Random Forest + HOG]  Training …")
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=200, max_depth=20,
                                class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_train_hog, y_train)
    y_pred = rf.predict(X_test_hog)
    y_prob = rf.predict_proba(X_test_hog)[:, 1]
    results_ml["RF + HOG"] = {"model": rf, "y_pred": y_pred, "y_prob": y_prob,
                               **metrics_dict(y_test, y_pred, y_prob)}
    print(f"  Time: {time.time()-t0:.1f}s")
    print(f"  F1={results_ml['RF + HOG']['F1-Score']:.4f} | "
          f"Recall={results_ml['RF + HOG']['Recall']:.4f} | "
          f"AUC={results_ml['RF + HOG']['AUC-ROC']:.4f}")

    # ── KNN (HOG features) ────────────────────
    print("\n  [KNN + HOG]  Training …")
    t0 = time.time()
    knn_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("knn",    KNeighborsClassifier(n_jobs=-1)),
    ])
    param_grid_knn = {"knn__n_neighbors": [3, 5, 7, 11]}
    gs_knn = GridSearchCV(knn_pipe, param_grid_knn, cv=3, scoring="f1",
                          n_jobs=-1, verbose=0)
    gs_knn.fit(X_train_hog, y_train)
    best_knn = gs_knn.best_estimator_
    y_pred   = best_knn.predict(X_test_hog)
    y_prob   = best_knn.predict_proba(X_test_hog)[:, 1]
    results_ml["KNN + HOG"] = {"model": best_knn, "y_pred": y_pred, "y_prob": y_prob,
                                **metrics_dict(y_test, y_pred, y_prob)}
    print(f"  Best k: {gs_knn.best_params_} | Time: {time.time()-t0:.1f}s")
    print(f"  F1={results_ml['KNN + HOG']['F1-Score']:.4f} | "
          f"Recall={results_ml['KNN + HOG']['Recall']:.4f} | "
          f"AUC={results_ml['KNN + HOG']['AUC-ROC']:.4f}")

    # ── Save models ───────────────────────────
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_svm,  "models/svm_hog.pkl")
    joblib.dump(rf,        "models/rf_hog.pkl")
    joblib.dump(best_knn,  "models/knn_hog.pkl")
    print("\n  → ML models saved to models/")

    # ── Visualisations ────────────────────────
    os.makedirs("Visualization", exist_ok=True)
    metric_keys = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]

    # Summary table
    rows = [{"Model": name, **{k: res[k] for k in metric_keys}}
            for name, res in results_ml.items()]
    df = pd.DataFrame(rows).set_index("Model")
    print("\n  ── Results ──")
    print(df.round(4).to_string())

    # Confusion matrices
    fig, axes = plt.subplots(1, len(results_ml), figsize=(5 * len(results_ml), 4))
    for ax, (name, res) in zip(axes, results_ml.items()):
        plot_confusion(y_test, res["y_pred"], name, ax)
    plt.suptitle("ML — Confusion Matrices", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("Visualization/ml_confusion_matrices.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("  → Saved: ml_confusion_matrices.png")

    # ROC curves
    fig, ax = plt.subplots(figsize=(7, 6))
    colors_roc = {"SVM + HOG": "#F44336", "RF + HOG": "#4CAF50", "KNN + HOG": "#FF9800"}
    for name, res in results_ml.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        ax.plot(fpr, tpr, label=f"{name}  (AUC={res['AUC-ROC']:.3f})",
                color=colors_roc.get(name, "gray"), linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ML — ROC Curves", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("Visualization/ml_roc_curves.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("  → Saved: ml_roc_curves.png")

    print("\n  ✅  Best ML model by Recall:")
    best = df["Recall"].idxmax()
    print(f"  → {best}  (Recall = {df.loc[best, 'Recall']:.4f})")

    return results_ml, y_test


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("Visualization", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    print("=" * 60)
    print("  PNEUMONIA DETECTION — ML PIPELINE")
    print("=" * 60)

    if not DATA_DIR.exists():
        raise SystemExit(
            f"\n[ERROR] Dataset directory '{DATA_DIR}' not found.\n"
            "  Download from: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia\n"
            "  Then set DATA_DIR at the top of this script."
        )

    results_ml, y_test = run_ml_pipeline()

    print("\n" + "=" * 60)
    print("  ML DONE ✓")
    print("=" * 60)
