"""
=============================================================================
Pipeline:
  1. Dataset exploration (EDA)
  2. Preprocessing
  3. ML Models: SVM, Random Forest, KNN (with HOG features)
  4. Deep Learning: CNN from scratch
  5. Evaluation & Comparison
=============================================================================
"""

# ─────────────────────────────────────────────
# 0. DEPENDENCIES
# ─────────────────────────────────────────────
import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import cv2
from pathlib import Path
from collections import Counter

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, classification_report, roc_curve
)
from sklearn.utils.class_weight import compute_class_weight
from skimage.feature import hog

import tensorflow
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

warnings.filterwarnings("ignore")
np.random.seed(42)
tensorflow.random.set_seed(42)

# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────
# Expected structure:
#   chest_xray/
#     train/NORMAL/*.jpeg
#     train/PNEUMONIA/*.jpeg
#     val/NORMAL/*.jpeg
#     val/PNEUMONIA/*.jpeg
#     test/NORMAL/*.jpeg
#     test/PNEUMONIA/*.jpeg

DATA_DIR   = Path("chest_xray")
IMG_SIZE_ML  = 64    # for ML models
IMG_SIZE_CNN = 128   # for CNN  (use 224 if GPU available)
BATCH_SIZE   = 32
EPOCHS       = 20
CLASSES      = ["NORMAL", "PNEUMONIA"]
LABEL_MAP    = {"NORMAL": 0, "PNEUMONIA": 1}


# ─────────────────────────────────────────────
# 2. HELPER FUNCTIONS
# ─────────────────────────────────────────────

def load_images(split: str, img_size: int, grayscale: bool = True):
    """Load all images from a split (train/val/test) into arrays."""
    images, labels = [], []
    for cls in CLASSES:
        folder = DATA_DIR / split / cls
        if not folder.exists():
            raise FileNotensorflowoundError(f"Folder not found: {folder}")
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
        # img shape: (H, W), values in [0,1]
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
# 3. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────

def run_eda():
    print("\n" + "=" * 60)
    print("  3. EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    split_counts = {}
    for split in ["train", "val", "test"]:
        counts = {}
        for cls in CLASSES:
            folder = DATA_DIR / split / cls
            counts[cls] = len(list(folder.glob("*.jpeg"))) if folder.exists() else 0
        split_counts[split] = counts
        total = sum(counts.values())
        print(f"  {split:6s}: {counts['NORMAL']:5d} NORMAL | {counts['PNEUMONIA']:5d} PNEUMONIA | Total: {total}")

    # Class distribution bar chart
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, split in zip(axes, ["train", "val", "test"]):
        vals = [split_counts[split][c] for c in CLASSES]
        colors = ["#2196F3", "#F44336"]
        ax.bar(CLASSES, vals, color=colors, edgecolor="white", linewidth=1.5)
        ax.set_title(f"{split.capitalize()} set", fontweight="bold")
        ax.set_ylabel("Count")
        for i, v in enumerate(vals):
            ax.text(i, v + 20, str(v), ha="center", fontsize=10)
    fig.suptitle("Class Distribution per Split", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("Visualization/eda_class_distribution.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("  → Saved: eda_class_distribution.png")

    # Sample images
    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    for row, cls in enumerate(CLASSES):
        folder = DATA_DIR / "train" / cls
        paths  = sorted(folder.glob("*.jpeg"))[:5]
        for col, path in enumerate(paths):
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            axes[row, col].imshow(img, cmap="gray")
            axes[row, col].set_title(cls, fontsize=9)
            axes[row, col].axis("off")
    fig.suptitle("Sample X-Ray Images (train)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("Visualization/eda_sample_images.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("  → Saved: eda_sample_images.png")

    # Pixel statistics per class (on a 200-image sample)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for i, cls in enumerate(CLASSES):
        folder = DATA_DIR / "train" / cls
        paths  = sorted(folder.glob("*.jpeg"))[:200]
        means  = []
        for p in paths:
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                means.append(img.mean())
        axes[i].hist(means, bins=40, color=["#2196F3", "#F44336"][i], edgecolor="white")
        axes[i].set_title(f"Mean pixel intensity — {cls}", fontweight="bold")
        axes[i].set_xlabel("Mean pixel value (0-255)")
        axes[i].set_ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("Visualization/eda_pixel_stats.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("  → Saved: eda_pixel_stats.png")

    return split_counts


# ─────────────────────────────────────────────
# 4. ML PIPELINE
# ─────────────────────────────────────────────

def run_ml_pipeline():
    print("\n" + "=" * 60)
    print("  4. MACHINE LEARNING MODELS")
    print("=" * 60)

    # 4.1 Load + preprocess
    print("  Loading training data …")
    X_train_raw, y_train = load_images("train", IMG_SIZE_ML)
    X_val_raw,   y_val   = load_images("val",   IMG_SIZE_ML)
    X_test_raw,  y_test  = load_images("test",  IMG_SIZE_ML)

    # Merge val into train (val is tiny: 16 images)
    X_train_raw = np.concatenate([X_train_raw, X_val_raw])
    y_train     = np.concatenate([y_train, y_val])

    print(f"  Train: {X_train_raw.shape[0]} images | Test: {X_test_raw.shape[0]} images")

    # 4.2 Feature extraction
    print("  Extracting HOG features …")
    t0 = time.time()
    X_train_hog = extract_hog_features(X_train_raw)
    X_test_hog  = extract_hog_features(X_test_raw)
    print(f"  HOG done in {time.time()-t0:.1f}s | feature dim: {X_train_hog.shape[1]}")

    # Also flatten for baseline
    X_train_flat = X_train_raw.reshape(len(X_train_raw), -1)
    X_test_flat  = X_test_raw.reshape(len(X_test_raw),  -1)

    # Class weights
    classes_   = np.unique(y_train)
    cw_values  = compute_class_weight("balanced", classes=classes_, y=y_train)
    class_weight_dict = dict(zip(classes_, cw_values))
    print(f"  Class weights: {class_weight_dict}")

    # ── 4.3 Define models ──────────────────────────────────────
    results_ml = {}

    # 4.3.1 SVM (HOG features) — with GridSearchCV
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
    best_svm  = gs_svm.best_estimator_
    y_pred    = best_svm.predict(X_test_hog)
    y_prob    = best_svm.predict_proba(X_test_hog)[:, 1]
    results_ml["SVM + HOG"] = {"model": best_svm, "y_pred": y_pred, "y_prob": y_prob,
                                **metrics_dict(y_test, y_pred, y_prob)}
    print(f"  Best params: {gs_svm.best_params_} | Time: {time.time()-t0:.1f}s")
    print(f"  F1={results_ml['SVM + HOG']['F1-Score']:.4f} | Recall={results_ml['SVM + HOG']['Recall']:.4f} | AUC={results_ml['SVM + HOG']['AUC-ROC']:.4f}")

    # 4.3.2 Random Forest (HOG features)
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
    print(f"  F1={results_ml['RF + HOG']['F1-Score']:.4f} | Recall={results_ml['RF + HOG']['Recall']:.4f} | AUC={results_ml['RF + HOG']['AUC-ROC']:.4f}")

    # 4.3.3 KNN (HOG features)
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
    print(f"  F1={results_ml['KNN + HOG']['F1-Score']:.4f} | Recall={results_ml['KNN + HOG']['Recall']:.4f} | AUC={results_ml['KNN + HOG']['AUC-ROC']:.4f}")

    return results_ml, y_test


# ─────────────────────────────────────────────
# 5. CNN PIPELINE
# ─────────────────────────────────────────────

def build_cnn(img_size: int) -> keras.Model:
    """
    Lightweight CNN trained from scratch.
    Block 1 : Conv(32, 3×3, ReLU) → MaxPool(2×2)
    Block 2 : Conv(64, 3×3, ReLU) → MaxPool(2×2)
    Block 3 : Conv(128, 3×3, ReLU) → MaxPool(2×2)
    Flatten → Dense(128, ReLU) → Dropout(0.5) → Dense(1, Sigmoid)
    """
    inp = keras.Input(shape=(img_size, img_size, 1))
    x   = inp

    for filters in [32, 64, 128]:
        x = layers.Conv2D(filters, (3, 3), padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inp, out, name="PneumoniaCNN")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model


def run_cnn_pipeline():
    print("\n" + "=" * 60)
    print("  5. DEEP LEARNING — CNN")
    print("=" * 60)

    # Data generators
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=True,
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.05,
        height_shift_range=0.05,
    )
    val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        DATA_DIR / "train",
        target_size=(IMG_SIZE_CNN, IMG_SIZE_CNN),
        color_mode="grayscale",
        class_mode="binary",
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42,
    )
    val_gen = val_test_datagen.flow_from_directory(
        DATA_DIR / "val",
        target_size=(IMG_SIZE_CNN, IMG_SIZE_CNN),
        color_mode="grayscale",
        class_mode="binary",
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    test_gen = val_test_datagen.flow_from_directory(
        DATA_DIR / "test",
        target_size=(IMG_SIZE_CNN, IMG_SIZE_CNN),
        color_mode="grayscale",
        class_mode="binary",
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # Class weights
    y_train_labels = train_gen.classes
    cw_vals = compute_class_weight("balanced", classes=np.unique(y_train_labels),
                                   y=y_train_labels)
    cw = dict(enumerate(cw_vals))
    print(f"  Class weights: {cw}")

    # Build & summarize
    model = build_cnn(IMG_SIZE_CNN)
    model.summary()

    # Callbacks
    cb_list = [
        callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                restore_best_weights=True, verbose=1),
        callbacks.ModelCheckpoint("" \
        "models/best_cnn.keras", monitor="val_loss",
                                  save_best_only=True, verbose=0),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                    patience=3, min_lr=1e-6, verbose=1),
    ]

    print("\n  Training CNN …")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        class_weight=cw,
        callbacks=cb_list,
        verbose=1,
    )

    # Plot training curves
    plot_training_history(history)

    # Evaluate on test set
    print("\n  Evaluating on test set …")
    y_true = test_gen.classes
    y_prob = model.predict(test_gen, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    results_cnn = {"y_pred": y_pred, "y_prob": y_prob,
                   **metrics_dict(y_true, y_pred, y_prob)}
    print(f"  Accuracy={results_cnn['Accuracy']:.4f} | Recall={results_cnn['Recall']:.4f} | AUC={results_cnn['AUC-ROC']:.4f}")

    return model, history, results_cnn, y_true


def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    metrics_pairs = [("loss", "val_loss"), ("accuracy", "val_accuracy")]
    titles = ["Loss", "Accuracy"]
    colors = [("#F44336", "#FF8A80"), ("#2196F3", "#82B1FF")]

    for ax, (tr_m, val_m), title, (c1, c2) in zip(axes, metrics_pairs, titles, colors):
        ep = range(1, len(history.history[tr_m]) + 1)
        ax.plot(ep, history.history[tr_m],   color=c1, linewidth=2, label="Train")
        ax.plot(ep, history.history[val_m],  color=c2, linewidth=2, linestyle="--", label="Val")
        ax.set_title(f"CNN {title}", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("Visualization/cnn_training_curves.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("  → Saved: cnn_training_curves.png")


# ─────────────────────────────────────────────
# 6. COMPARISON & VISUALISATION
# ─────────────────────────────────────────────

def run_comparison(results_ml: dict, results_cnn: dict, y_test: np.ndarray):
    print("\n" + "=" * 60)
    print("  6. EVALUATION & COMPARISON")
    print("=" * 60)

    metric_keys = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
    all_results = {**results_ml, "CNN": results_cnn}

    # Summary table
    rows = []
    for name, res in all_results.items():
        rows.append({"Model": name, **{k: res[k] for k in metric_keys}})
    df = pd.DataFrame(rows).set_index("Model")
    print("\n", df.round(4).to_string())

    # Bar chart comparison
    fig, ax = plt.subplots(figsize=(12, 5))
    x     = np.arange(len(df))
    width = 0.15
    palette = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"]

    for i, (metric, color) in enumerate(zip(metric_keys, palette)):
        ax.bar(x + i * width, df[metric], width, label=metric, color=color, alpha=0.85)

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(df.index, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — All Metrics", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("Visualization/comparison_bar.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("  → Saved: comparison_bar.png")

    # Confusion matrices
    fig, axes = plt.subplots(1, len(all_results), figsize=(5 * len(all_results), 4))
    for ax, (name, res) in zip(axes, all_results.items()):
        yt = y_test if name != "CNN" else res.get("y_true", y_test)
        plot_confusion(yt, res["y_pred"], name, ax)
    plt.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("Visualization/confusion_matrices.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("  → Saved: confusion_matrices.png")

    # ROC curves
    fig, ax = plt.subplots(figsize=(7, 6))
    colors_roc = {"SVM + HOG": "#F44336", "RF + HOG": "#4CAF50",
                  "KNN + HOG": "#FF9800", "CNN": "#2196F3"}
    for name, res in all_results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        ax.plot(fpr, tpr, label=f"{name}  (AUC={res['AUC-ROC']:.3f})",
                color=colors_roc.get(name, "gray"), linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("Visualization/roc_curves.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("  → Saved: roc_curves.png")

    print("\n  ✅  Best model by Recall (priority metric in medical context):")
    best = df["Recall"].idxmax()
    print(f"  → {best}  (Recall = {df.loc[best, 'Recall']:.4f})")

    return df


# ─────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Add at the beginning of main (after checking DATA_DIR):
    os.makedirs("Visualization", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    print("=" * 60)
    print("  PNEUMONIA DETECTION SYSTEM — ENSA Tanger 2025-2026")
    print("=" * 60)

    # Check dataset
    if not DATA_DIR.exists():
        raise SystemExit(
            f"\n[ERROR] Dataset directory '{DATA_DIR}' not found.\n"
            "  Download from: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia\n"
            "  Then set DATA_DIR at the top of this script."
        )

    # Step 3 — EDA
    split_counts = run_eda()

    # Step 4 — ML
    results_ml, y_test = run_ml_pipeline()

    # Step 5 — CNN
    model, history, results_cnn, y_test_cnn = run_cnn_pipeline()
    # Attach y_true for confusion matrix
    results_cnn["y_true"] = y_test_cnn

    # Step 6 — Comparison (use CNN's y_test for alignment)
    # Note: y_test from ML and y_test_cnn should match (same test/ folder)
    df_results = run_comparison(results_ml, results_cnn, y_test_cnn)

    print("\n" + "=" * 60)
    print("  ALL DONE ✓")
    print("=" * 60)
    print("  Generated files:")
    for f in ["Visualization/eda_class_distribution.png", "Visualization/eda_sample_images.png",
              "Visualization/eda_pixel_stats.png", "Visualization/cnn_training_curves.png",
              "Visualization/comparison_bar.png", "Visualization/confusion_matrices.png", "Visualization/roc_curves.png",
              "models/best_cnn.keras"]:
        print(f"    • {f}")