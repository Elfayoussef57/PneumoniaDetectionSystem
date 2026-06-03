"""
=============================================================================
DEEP LEARNING PIPELINE — Pneumonia Detection
  Model: CNN from scratch
=============================================================================
"""

# ─────────────────────────────────────────────
# DEPENDENCIES
# ─────────────────────────────────────────────
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve
)
from sklearn.utils.class_weight import compute_class_weight

import tensorflow
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

warnings.filterwarnings("ignore")
np.random.seed(42)
tensorflow.random.set_seed(42)

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DATA_DIR     = Path("chest_xray")
IMG_SIZE_CNN = 128   # increase to 224 if GPU is available
BATCH_SIZE   = 32
EPOCHS       = 20
CLASSES      = ["NORMAL", "PNEUMONIA"]


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def metrics_dict(y_true, y_pred, y_prob=None):
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
# CNN ARCHITECTURE
# ─────────────────────────────────────────────

def build_cnn(img_size: int) -> keras.Model:
    """
    Lightweight CNN trained from scratch.
    Block 1 : Conv(32, 3×3, ReLU) → BN → MaxPool(2×2)
    Block 2 : Conv(64, 3×3, ReLU) → BN → MaxPool(2×2)
    Block 3 : Conv(128, 3×3, ReLU) → BN → MaxPool(2×2)
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


# ─────────────────────────────────────────────
# TRAINING HISTORY PLOT
# ─────────────────────────────────────────────

def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    metrics_pairs = [("loss", "val_loss"), ("accuracy", "val_accuracy")]
    titles = ["Loss", "Accuracy"]
    colors = [("#F44336", "#FF8A80"), ("#2196F3", "#82B1FF")]

    for ax, (tr_m, val_m), title, (c1, c2) in zip(axes, metrics_pairs, titles, colors):
        ep = range(1, len(history.history[tr_m]) + 1)
        ax.plot(ep, history.history[tr_m],  color=c1, linewidth=2, label="Train")
        ax.plot(ep, history.history[val_m], color=c2, linewidth=2, linestyle="--", label="Val")
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
# CNN PIPELINE
# ─────────────────────────────────────────────

def run_cnn_pipeline():
    print("\n" + "=" * 60)
    print("  DEEP LEARNING — CNN")
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
        callbacks.ModelCheckpoint("models/best_cnn.keras", monitor="val_loss",
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

    results_cnn = {
        "y_pred": y_pred,
        "y_prob": y_prob,
        "y_true": y_true,
        **metrics_dict(y_true, y_pred, y_prob)
    }
    print(f"  Accuracy={results_cnn['Accuracy']:.4f} | "
          f"Recall={results_cnn['Recall']:.4f} | "
          f"AUC={results_cnn['AUC-ROC']:.4f}")

    # Confusion matrix
    os.makedirs("Visualization", exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    plot_confusion(y_true, y_pred, "CNN", ax)
    plt.tight_layout()
    plt.savefig("Visualization/cnn_confusion_matrix.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("  → Saved: cnn_confusion_matrix.png")

    # ROC curve
    fig, ax = plt.subplots(figsize=(6, 5))
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ax.plot(fpr, tpr, color="#2196F3", linewidth=2,
            label=f"CNN  (AUC={results_cnn['AUC-ROC']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("CNN — ROC Curve", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("Visualization/cnn_roc_curve.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("  → Saved: cnn_roc_curve.png")

    print("\n  → CNN model saved to models/best_cnn.keras")
    return model, history, results_cnn


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("Visualization", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    print("=" * 60)
    print("  PNEUMONIA DETECTION — DL PIPELINE")
    print("=" * 60)

    if not DATA_DIR.exists():
        raise SystemExit(
            f"\n[ERROR] Dataset directory '{DATA_DIR}' not found.\n"
            "  Download from: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia\n"
            "  Then set DATA_DIR at the top of this script."
        )

    model, history, results_cnn = run_cnn_pipeline()

    print("\n" + "=" * 60)
    print("  DL DONE ✓")
    print("=" * 60)
