# Pneumonia Detection System — ENSA Tanger 2025-2026

## Membres
El Fajri Youssef · El Alami Hassoun Mohamed · El Amrani Alae · El Aouzi Walid

---

## Description

Pipeline complet de détection de pneumonie sur des radiographies thoraciques,
comparant des approches de Machine Learning classique et de Deep Learning (CNN).

---

## Structure du projet

```
pneumonia_detection.py   ← script principal (tout le pipeline)
requirements.txt         ← dépendances Python
```

---

## Installation

```bash
# 1. Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

# 2. Installer les dépendances
pip install -r requirements.txt
```

---

## Dataset

Télécharger depuis Kaggle :
**https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia**

Décompresser dans le même dossier que le script. La structure attendue :

```
chest_xray/
  train/
    NORMAL/       ← 1 341 images
    PNEUMONIA/    ← 3 875 images
  val/
    NORMAL/       ←     8 images
    PNEUMONIA/    ←     8 images
  test/
    NORMAL/       ←   234 images
    PNEUMONIA/    ←   390 images
```

Si vous placez le dataset ailleurs, modifiez `DATA_DIR` en haut du script.

---

## Exécution

```bash
# Entraîner d'abord
python ml_pipeline.py
python dl_pipeline.py

# Prédire avec tous les modèles
python predict.py --image chest_xray/test/PNEUMONIA/person1_bacteria_1.jpeg

# Prédire avec le CNN seulement
python predict.py --image mon_xray.jpeg --model cnn

# Sans afficher le plot
python predict.py --image mon_xray.jpeg --no-plot
```

Ou dans Jupyter :

```bash
jupyter lab
# Puis copiez le contenu dans un notebook .ipynb
```

---

## Pipeline détaillé

### 1. EDA (Analyse exploratoire)
- Distribution des classes par split
- Visualisation d'images exemples
- Statistiques de pixels par classe
- Fichiers générés : `eda_*.png`

### 2. Preprocessing
- **ML** : resize 64×64, grayscale, normalisation [0,1]
- **CNN** : resize 128×128, grayscale, normalisation [0,1]
- Augmentation (CNN seulement) : flip, rotation ±10°, zoom ±10%, shift ±5%
- Gestion du déséquilibre : `class_weight="balanced"`

### 3. Modèles ML (features HOG)
| Modèle | Hyperparamètres optimisés |
|--------|--------------------------|
| SVM (RBF) | C ∈ {0.1, 1, 10}, gamma ∈ {scale, auto} |
| Random Forest | 200 arbres, max_depth=20 |
| KNN | k ∈ {3, 5, 7, 11} |

### 4. CNN (from scratch)
```
Conv(32) → BN → MaxPool
Conv(64) → BN → MaxPool
Conv(128) → BN → MaxPool
Flatten → Dense(128, ReLU) → Dropout(0.5) → Dense(1, Sigmoid)
```
- Optimizer : Adam (lr=1e-3)
- Loss : binary_crossentropy
- EarlyStopping (patience=5) + ReduceLROnPlateau

### 5. Métriques d'évaluation
- Accuracy, Precision, **Recall (priorité)**, F1-Score, AUC-ROC
- Matrices de confusion
- Courbes ROC

---

## Fichiers générés

| Fichier | Description |
|---------|-------------|
| `eda_class_distribution.png` | Distribution des classes |
| `eda_sample_images.png` | Exemples de radiographies |
| `eda_pixel_stats.png` | Statistiques de pixels |
| `cnn_training_curves.png` | Courbes d'apprentissage CNN |
| `comparison_bar.png` | Comparaison des modèles (bar chart) |
| `confusion_matrices.png` | Matrices de confusion de tous les modèles |
| `roc_curves.png` | Courbes ROC comparatives |
| `best_cnn.keras` | Poids du meilleur modèle CNN sauvegardé |

---

## Références
- Kermany et al. (Cell 2018) — dataset original
- Rajpurkar et al. (CheXNet, Stanford 2017) — référence état de l'art
- Scikit-learn, TensorFlow/Keras, OpenCV