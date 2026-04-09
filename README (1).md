# 🔬 Embryo Stage Classification — 2-Phase Transfer Learning

Automated classification of human embryo development stages from time-lapse microscopy images using deep learning. Four CNN architectures are trained with a custom biologically-aware loss function and a 2-phase transfer learning strategy.

---

## 📌 Overview

| Item | Detail |
|------|--------|
| **Task** | 16-class embryo stage classification |
| **Models** | MobileNetV2, InceptionV3, VGG16, VGG19 |
| **Framework** | TensorFlow / Keras |
| **Dataset** | [Embryo Dataset – Kaggle](https://www.kaggle.com/datasets/abhishekbuddiga06/embryo-dataset) |
| **Input** | Grayscale JPEG frames from time-lapse embryo videos |
| **Loss** | OrdinalFocalWeightedLoss (Focal CE + Ordinal Distance) |

---

## 🧬 Embryo Development Stages

The model classifies 16 sequential developmental stages in chronological order:

```
pPB2 → pPNa → pPNf → p2 → p3 → p4 → p5 → p6 → p7 → p8 → p9+ → pM → pSB → pB → pEB → pHB
```

---

## 🏗️ Architecture & Design Choices

### Model Head (all architectures)
- **GAP + GMP concatenation** — Global Average Pooling and Global Max Pooling outputs are concatenated for richer feature representation
- **FC layers**: 256 → 128 units with ReLU, L2 regularisation, BatchNorm, and Dropout (0.60 / 0.50)
- **Output**: 16-class logits (float32)

### 2-Phase Transfer Learning

| Phase | Trainable Layers | LR | Epochs | Purpose |
|-------|-----------------|-----|--------|---------|
| Phase 1 | Head only | 1e-4 | 5 | Fast stabilisation |
| Phase 2 | Head + last 2 conv blocks | 1e-5 – 5e-5 | 7–20 | Accuracy fine-tuning |

Per-model schedule via `PHASE_CONFIG`:

| Model | Phase 1 | Phase 2 | Total |
|-------|---------|---------|-------|
| MobileNetV2 | 5 ep @ 1e-4 | 20 ep @ 5e-5 | 25 ep |
| VGG16 | 5 ep @ 1e-4 | 7 ep @ 1e-5 | 12 ep |
| VGG19 | 5 ep @ 1e-4 | 7 ep @ 1e-5 | 12 ep |
| InceptionV3 | 5 ep @ 1e-4 | 7 ep @ 5e-5 | 12 ep |

---

## 📉 Loss Function — OrdinalFocalWeightedLoss

A custom biologically-aware loss combining three components:

```
L_total = λ_ce × L_focal_weighted  +  λ_ord × L_ordinal
```

| Component | Description | Weight |
|-----------|-------------|--------|
| **Focal CE** | Class-weighted focal cross-entropy with label smoothing | λ_ce = 0.6 |
| **Ordinal Distance** | Penalises predictions far from true stage in developmental order | λ_ord = 0.4 |

- **Focal gamma**: 2.0
- **Label smoothing**: 0.1
- Stage ranks encoded in a normalised `RANK_MATRIX_TF` (16×16)

---

## 📦 Dataset & Preprocessing

### Data Split (by video — no leakage)
| Split | Fraction |
|-------|----------|
| Train | 70% |
| Validation | 15% |
| Test | 15% |

### Class Rebalancing
- **Cap**: 15,000 samples per class (downsampling)
- **Floor**: 500 samples per class (oversampling with replacement)
- **Class weights**: clipped to [0.3, 10.0] for focal loss

### Augmentation (training only)
- Random horizontal & vertical flip
- Random brightness and contrast jitter
- Random 90° rotations
- Random crop (85%) + resize back
- **Cutout** (random erasing, 50% probability, 12.5% patch size)

### Image Pipeline
- Grayscale JPEG → RGB conversion
- Resize to 224×224 (MobileNetV2, VGG16, VGG19) or 299×299 (InceptionV3)
- Model-specific preprocessing (ImageNet normalisation)

---

## 📁 Project Structure

```
📦 embryo-classification
 ├── embryo-classification-anuja.ipynb   # Main notebook
 ├── /kaggle/working/
 │   ├── {ModelName}_best.keras          # Best checkpoint per model
 │   ├── {ModelName}_log.csv             # Per-epoch training log
 │   ├── training_curves.png             # Accuracy & loss curves
 │   ├── val_acc_overlay.png             # All models validation accuracy
 │   └── model_summary.csv              # Final comparison table
 └── /kaggle/input/
     └── embryo-dataset/                 # Raw frames + annotation CSVs
```

---

## ▶️ How to Run

1. **Add the dataset** as a Kaggle input: `abhishekbuddiga06/embryo-dataset`
2. Enable **GPU accelerator** (T4 recommended)
3. Run cells in order: **Cell 1 → Cell 15**
   - Cells 1–8: Setup, data loading, model definitions
   - Cells 9–10: LR schedule and training pipeline
   - Cells 11–14: Train each model individually
   - Cell 15: Final comparison plots and summary table

> **To skip retraining**: Run Cells 1–8, load saved `.keras` checkpoints, then run Cell 15 directly.

---

## 📊 Output

- Per-model classification report (precision, recall, F1 per stage)
- Confusion matrices
- Training curves with phase boundary markers
- Validation accuracy overlay (all 4 models)
- `model_summary.csv` with best val accuracy and test accuracy per model

---

## ⚙️ Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Batch size | 32 |
| Optimizer | Adam with CosineAnnealing LR |
| L2 regularisation | 1e-4 |
| Dropout | 0.60 (FC1), 0.50 (FC2) |
| FC1 / FC2 units | 256 / 128 |
| Mixed precision | float16 (if GPU available) |
| Random seed | 42 |

---

## 📋 Requirements

```
tensorflow >= 2.x
numpy
pandas
scikit-learn
matplotlib
seaborn
```

All dependencies are pre-installed in the Kaggle environment.

---

## 👤 Author

Anuja — Kaggle Notebook: `embryo-classification-anuja`
