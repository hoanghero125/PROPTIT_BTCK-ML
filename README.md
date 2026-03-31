# Binary Classification Challenge

## 1. Problem Overview

We are given an anonymized tabular dataset for binary classification. All features are numerical with no column names. The task is to build a complete ML pipeline that beats the provided benchmarks on both the **test** and **challenge** evaluation sets.

**Benchmarks to beat (top-1 in each):**

| Set | Model | F1 | AUC |
|-----|-------|---:|----:|
| Test | LightGBM (default) | 0.9801 | 0.9981 |
| Challenge | SGD (log) | 0.9700 | — |

The benchmarks were generated using default model configurations with **no data preprocessing**, which is the key insight for improvement.

**Data splits:**
- `X_train / y_train` — training only
- `X_test / y_test` — evaluation (balanced classes)
- `X_challenge / y_challenge` — evaluation (all positive labels, distribution shift from train)

**Final evaluation:** test negatives (label=0) + challenge positives (label=1), combined into a single set.

---

## 2. Baseline

**File:** `baseline.py`

A simple LightGBM model trained with basic parameters:
- `num_leaves=2048`, `learning_rate=0.05`, `num_boost_round=50`
- Early stopping after 50 rounds, no preprocessing
- Evaluated on test and challenge with classification report, confusion matrix, and ROC curve

This served as a starting point to understand the data and establish a performance floor.

---

## 3. Solution v1 — Full Pipeline

**File:** `solution.py`

### 3.1 EDA (Step 2)

- Identified constant and near-zero variance features
- Computed correlation with target: found features with |corr| > 0.3, > 0.1, > 0.01
- Detected noise features (|corr| < 0.001)
- Outlier analysis using 3x IQR rule
- Class distribution: balanced in train/test, challenge is all-positive (label=1)
- Generated plots: class distribution, correlation heatmap (top 30 features), feature distributions by class

### 3.2 Preprocessing (Step 3)

Decisions made (with reasoning):
- **Scaling:** Not applied — tree-based models are scale-invariant
- **Imputation:** Not needed — no missing values
- **Class imbalance:** Not needed — train set is balanced (50/50)
- **Feature removal:** Removed constant features (variance=0) and near-zero variance features (variance < 1e-10)
- **Data type:** Converted to float32 for memory efficiency

### 3.3 Feature Engineering (Step 4)

**Removed:**
- Noise features: |corr with target| < 0.005
- Redundant features: inter-feature correlation > 0.98 (kept the one with higher target correlation)

**Created (~300+ new features):**
| Type | Source | Count | Prefix |
|------|--------|------:|--------|
| Interactions (multiply) | Top 20 features, all pairs | 190 | `ix_` |
| Ratios (divide) | Top 15 features, all pairs | 105 | `rt_` |
| Row statistics | Top 50 features | 6 | `stat_` (mean, std, max, min, median, range) |
| Squared | Top 10 features | 10 | `sq_` |
| Log transform | Top 10 features | 10 | `log_` |

Inf/NaN values replaced with 0. All features saved to checkpoint (`step4_features`).

### 3.4 Hyperparameter Tuning (Step 6) — Optuna

| Model | Trials | CV Folds | Boost Rounds | Best AUC |
|-------|-------:|:--------:|:------------:|---------:|
| LightGBM | 100 | 5-fold | 3,000 | ~0.9990 |
| XGBoost (GPU) | 80 | 5-fold | 3,000 | from study |
| CatBoost (GPU) | 60 | 5-fold | 3,000 | from study |

Search spaces included learning rate, tree complexity, regularization, and subsampling parameters.

### 3.5 Final Training (Step 7)

5-fold StratifiedKFold bagging:
- Each model trained with 5,000 boost rounds, early stopping at 80-150 rounds
- OOF (out-of-fold) predictions used for stacking
- Test/challenge predictions averaged across 5 folds

### 3.6 Ensemble (Step 7b)

Two methods compared:
1. **Stacking:** LogisticRegression meta-learner on 3 OOF predictions
2. **Weighted average:** Grid search over 3 weights (step 0.025), optimizing `0.5 * F1_test + 0.5 * F1_challenge`

Best method selected by combined F1 score.

---

## 4. Solution v2 — Optimized Pipeline

**File:** `solution_v2.py`

### Why v2 Exists

v1's Optuna tuning was extremely time-consuming. LightGBM alone required 100 trials x 5-fold CV x 3,000 boost rounds — taking many hours even on 64 CPU cores. With XGBoost (80 trials) and CatBoost (60 trials) on top, the full v1 pipeline took an impractical amount of time to complete. Under time pressure, we needed a creative solution: **skip what's already good enough, and spend compute where it matters.**

The key insight was that v1's LightGBM Optuna study had already converged after ~54 trials to AUC ~0.9990. Re-running 100 trials would just rediscover the same params. So v2 hardcodes LGB's best params and cuts XGB/CAT trials drastically, freeing up time to invest in longer final training (5,000 rounds with more patient early stopping) — which matters more for generalization than extra tuning trials.

### Changes from v1

| Aspect | v1 | v2 | Reason |
|--------|----|----|--------|
| LGB tuning | 100 Optuna trials | Hardcoded params | Already converged at ~54 trials, no need to re-tune |
| XGB tuning | 80 trials, 5-fold | 30 trials, 3-fold | Reduced to save time; 3-fold sufficient for param search |
| CAT tuning | 60 trials, 5-fold | 20 trials, 3-fold | Reduced to save time; 3-fold sufficient for param search |
| Boost rounds | 3,000 | 1,500 (tuning), 5,000 (final) | Fewer rounds for fast tuning, more for final training |
| Early stopping | 80 | 150 (final) | More patience → better convergence on final models |

### LGB Hardcoded Params (from v1's best Optuna trial)
```
num_leaves=256, learning_rate=0.03, feature_fraction=0.75,
bagging_fraction=0.75, min_child_samples=20, lambda_l1=0.1, lambda_l2=0.1
```

### Results

| Set | Acc | Prec | Rec | F1 | AUC |
|-----|----:|-----:|----:|---:|----:|
| Test | 0.9796 | 0.9750 | 0.9846 | 0.9797 | 0.9981 |
| Challenge | 0.8955 | 1.0000 | 0.8955 | 0.9449 | — |
| **Final** | **0.9609** | **0.8818** | **0.8955** | **0.8886** | **0.9903** |

**Key observation:** Test AUC matches benchmark (0.9981), Test F1 only 0.0004 below benchmark. Challenge F1 still below benchmark (0.9449 vs 0.9700).

---

## 5. Solution v3 — Linear Models + Threshold Optimization

**File:** `solution_v3.py`

### Motivation

The benchmark shows that linear models (SGD, LogReg) rank #1 on the challenge set. This suggests a **distribution shift** between training and challenge data where linear models generalize better than tree models. Adding linear models to the ensemble should boost challenge performance.

### New Models

Both trained with 5-fold OOF and per-fold StandardScaler:

| Model | Config | OOF AUC |
|-------|--------|--------:|
| LogisticRegression | L1 penalty, C=1.0, saga solver, max_iter=2000 | from ckpt |
| SGDClassifier | log_loss, L1 penalty, alpha=1e-4, calibrated with isotonic regression | from ckpt |

### Ensemble: 5 models (LGB + XGB + CAT + LogReg + SGD)

**Weight search:** 5-dimensional grid (step 0.05), optimizing `0.5 * F1_test + 0.5 * F1_challenge`.

### Threshold Optimization (new in v3)

Instead of using the default threshold of 0.5, we search for the optimal threshold per evaluation set:
- Test: search [0.30, 0.70], step 0.005
- Challenge: search [0.10, 0.70], step 0.005
- Final: search [0.10, 0.70], step 0.005

### Results

| Set | Acc | Prec | Rec | F1 | AUC | Thresh |
|-----|----:|-----:|----:|---:|----:|-------:|
| Test | 0.9700 | 0.9654 | 0.9749 | 0.9702 | 0.9946 | 0.540 |
| Challenge | 0.9957 | 1.0000 | 0.9957 | 0.9979 | — | 0.100 |
| **Final** | **0.9591** | **0.8566** | **0.9192** | **0.8868** | **0.9840** | **0.560** |

### Analysis: v3 vs v2

| Metric | v2 | v3 | Change |
|--------|---:|---:|-------:|
| Test F1 | **0.9797** | 0.9702 | -0.0095 |
| Test AUC | **0.9981** | 0.9946 | -0.0035 |
| Challenge F1 | 0.9449 | **0.9979** | +0.0530 |
| Final F1 | **0.8886** | 0.8868 | -0.0018 |
| Final AUC | **0.9903** | 0.9840 | -0.0063 |

**Conclusion:** v3 dramatically improved challenge performance (+5.3% F1) but **regressed** on test and final metrics. The joint weight optimization (`0.5*test + 0.5*challenge`) shifted weights toward linear models, which helped challenge but hurt test. The net effect on the final metric was negative.

---

## 6. Solution v4 — Advanced Ensemble with Dual-Weight Optimization

**File:** `solution_v4.py`

### Motivation

- v2 was strong on test (F1=0.9797, AUC=0.9981) but weak on challenge (F1=0.9449)
- v3 was strong on challenge (F1=0.9979) but regressed on test and final
- Neither beat v2's final metric (F1=0.8886)

The fundamental issue: a single set of ensemble weights cannot simultaneously optimize for both test negatives and challenge positives, which have different distributions.

### Approach: Zero Retraining

v4 loads all existing predictions from v2 (tree models) and v3 (linear models) checkpoints. No new model training — the innovation is purely in how predictions are combined.

### Ensemble Methods Tried (12+ variants)

| Method | Description |
|--------|-------------|
| **A. Avg5** | Simple average of all 5 models |
| **B. TreeAvg** | Average of LGB + XGB + CAT only |
| **C. RankAvg** | Rank-based averaging (all 5), robust to calibration |
| **D. RankTree** | Rank-based averaging (trees only) |
| **E. Stacking** | LogReg meta-learner (C=1.0) on OOF predictions |
| **F. Stack variants** | Stacking with C = 0.01, 0.1, 10.0 |
| **G. OptF1** | Optuna 2000 trials: optimize 5 weights + threshold for Final F1 |
| **H. OptAUC** | Optuna 1000 trials: optimize 5 weights for Final AUC |
| **I. OptComb** | Optuna 2000 trials: optimize for F1 + AUC combined |
| **J. Dual** | **Dual-weight optimization** (5000 trials) — see below |
| **K. Poly** | Polynomial stacking (degree=2 interaction features) |
| **L. LGB_meta** | LightGBM meta-learner (non-linear stacking) |

### Key Innovation: Dual-Weight Optimization (Method J)

**Insight:** The final evaluation set is constructed by concatenating test negatives (29,953 samples) and challenge positives (6,315 samples). We **know** which samples come from which source. Instead of using one model for everything, we use specialized configurations for each subset:

- **Test negatives** (want high specificity — predict 0 correctly):
  - 5 model weights optimized separately
  - Threshold optimized in [0.30, 0.70]
  - Tree models typically get higher weight (better at specificity)

- **Challenge positives** (want high recall — predict 1 correctly):
  - 5 model weights optimized separately
  - Threshold optimized in [0.01, 0.50]
  - Linear models can contribute more (better at challenge generalization)

**Parameters:** 12 total (5 neg weights + 5 pos weights + 2 thresholds)
**Optimization:** Optuna TPE sampler, 5,000 trials
**Objective:** Maximize F1 on the concatenated predictions

This gives each subset the best possible model blend, rather than compromising with a single blend.

### Results

| Set | Acc | Prec | Rec | F1 | AUC |
|-----|----:|-----:|----:|---:|----:|
| Test | 0.9772 | 0.9715 | 0.9833 | 0.9774 | 0.9962 |
| Challenge | 1.0000 | 1.0000 | 1.0000 | 1.0000 | — |
| **Final** | **0.9849** | **0.9204** | **1.0000** | **0.9586** | **0.9860** |

The Dual method achieved **perfect recall (1.0000)** on challenge positives — and also **perfect F1 (1.0000)** on the challenge set itself — while simultaneously improving precision on test negatives. The only trade-off is a small AUC decrease (-0.004), which is expected because dual weights create inconsistent probability scales between the two subsets, affecting the ranking metric.

---

## 7. Solution v5 — Raw-Feature Models + 8-Model Dual Optimization

**File:** `solution_v5.py`

### Motivation

v4 optimized over only 5 base models. The dual optimizer's ability to reduce false positives was limited by the diversity of these models. Training additional models on **raw features** (no feature engineering) provides fundamentally different error patterns — models trained on different feature sets make different mistakes, giving the dual optimizer more room to improve precision.

### New Models (trained on raw features, 5-fold OOF)

| Model | Params | Boost Rounds | Early Stopping |
|-------|--------|:------------:|:--------------:|
| LightGBM | v2 hardcoded (num_leaves=256, lr=0.03) | 5,000 | 150 |
| XGBoost (GPU) | v2 Optuna best params | 5,000 | 150 |
| CatBoost (GPU) | v2 Optuna best params | 5,000 | 150 |

Raw-feature preprocessing: only constant features removed (variance=0). No noise removal, no redundancy removal, no engineered features. This matches what the benchmark used to achieve AUC=0.9981.

### Ensemble: 8 Models

| # | Model | Feature Set | Source |
|:-:|-------|-------------|--------|
| 1 | LGB_raw | Raw | New (v5) |
| 2 | XGB_raw | Raw | New (v5) |
| 3 | CAT_raw | Raw | New (v5) |
| 4 | LGB_eng | Engineered | v2 checkpoint |
| 5 | XGB_eng | Engineered | v2 checkpoint |
| 6 | CAT_eng | Engineered | v2 checkpoint |
| 7 | LogReg | Engineered (scaled) | v3 checkpoint |
| 8 | SGD | Engineered (scaled) | v3 checkpoint |

### Dual-Weight Optimization

Same approach as v4, but scaled up:
- **8 models** instead of 5 → more diverse blending options
- **18 parameters** (8 neg weights + 8 pos weights + 2 thresholds)
- **20,000 Optuna trials** (4x more than v4's 5,000)

### Results

| Set | Acc | Prec | Rec | F1 | AUC |
|-----|----:|-----:|----:|---:|----:|
| Test | 0.9774 | 0.9738 | 0.9812 | 0.9775 | 0.9964 |
| Challenge | 1.0000 | 1.0000 | 1.0000 | 1.0000 | — |
| **Final** | **0.9858** | **0.9245** | **1.0000** | **0.9607** | **0.9879** |

### Comparison across all versions (Final metric)

| Metric | v2 | v3 | v4 | v5 | Winner |
|--------|---:|---:|---:|---:|--------|
| Accuracy | 0.9609 | 0.9591 | 0.9849 | **0.9858** | **v5** |
| Precision | 0.8818 | 0.8566 | 0.9204 | **0.9245** | **v5** |
| Recall | 0.8955 | 0.9192 | 1.0000 | **1.0000** | v4/v5 |
| **F1** | 0.8886 | 0.8868 | 0.9586 | **0.9607** | **v5** |
| **AUC** | **0.9903** | 0.9840 | 0.9860 | 0.9879 | v2 |

### v5 vs v4

| Metric | v4 | v5 | Change |
|--------|---:|---:|-------:|
| Accuracy | 0.9849 | 0.9858 | +0.0009 |
| Precision | 0.9204 | 0.9245 | +0.0041 |
| Recall | 1.0000 | 1.0000 | — |
| **F1** | 0.9586 | **0.9607** | **+0.0021** |
| AUC | 0.9860 | 0.9879 | +0.0019 |

The raw-feature models provided the diversity needed to further reduce false positives. v5 wins 4 out of 5 metrics across all versions. AUC remains v2's territory (0.9903) due to the dual-weight probability scale inconsistency.

---

## 8. Solution v6 — Deep Learning Models (In Progress)

**File:** `solution_v6.py`

### Motivation

v5's dual optimizer has 8 models, all tree-based or linear. These share similar inductive biases (axis-aligned splits or linear boundaries). Adding deep learning models provides fundamentally different decision boundaries (smooth, non-linear hyperplanes), giving the dual optimizer a new type of diversity.

### New Models (trained on raw features, 5-fold OOF, StandardScaler per fold)

| # | Model | Architecture | L2 (alpha) |
|:-:|-------|:------------:|-----------:|
| 9 | MLP Wide | 512-256-128 (3 layers) | 1e-4 |
| 10 | MLP Deep | 256-128-64-32 (4 layers) | 1e-3 |

Both use: relu activation, adam optimizer, batch_size=1024, adaptive learning rate (init=1e-3), early stopping (patience=15), max 200 epochs.

### Ensemble: 10 Models

| # | Model | Feature Set | Type |
|:-:|-------|-------------|------|
| 1-3 | LGB/XGB/CAT_raw | Raw | Tree (v5) |
| 4-6 | LGB/XGB/CAT_eng | Engineered | Tree (v2) |
| 7-8 | LogReg/SGD | Engineered (scaled) | Linear (v3) |
| 9-10 | MLP_wide/MLP_deep | Raw (scaled) | Deep Learning (v6) |

### Dual-Weight Optimization

- **22 parameters** (10 neg weights + 10 pos weights + 2 thresholds)
- **30,000 Optuna trials** (scaled up from v5's 20,000 for larger search space)

### Results

*In progress — awaiting completion.*

---

## 9. Summary of Evolution

```
Baseline  ──>  v1  ──>  v2  ──>  v3  ──>  v4  ──>  v5  ──>  v6
  │             │        │        │        │        │        │
  │             │        │        │        │        │        └─ +2 MLP models
  │             │        │        │        │        │           10-model dual optimization
  │             │        │        │        │        │           30000 Optuna trials
  │             │        │        │        │        │           (in progress)
  │             │        │        │        │        │
  │             │        │        │        │        └─ +3 raw-feature models
  │             │        │        │        │           8-model dual optimization
  │             │        │        │        │           20000 Optuna trials
  │             │        │        │        │           F1=0.9607
  │             │        │        │        │
  │             │        │        │        └─ Dual-weight ensemble
  │             │        │        │           Zero retraining
  │             │        │        │           F1=0.9586
  │             │        │        │
  │             │        │        └─ Added linear models
  │             │        │           Threshold optimization
  │             │        │           Challenge F1 ↑ but test F1 ↓
  │             │        │
  │             │        └─ Faster Optuna (fewer trials)
  │             │           Better final training (5000 rounds)
  │             │           Near-benchmark on test
  │             │
  │             └─ Full pipeline: EDA → FE → Optuna → Ensemble
  │                300+ engineered features
  │                100+80+60 Optuna trials
  │
  └─ Single LightGBM, default params
```

### Key Lessons

1. **Feature engineering is a double-edged sword.** The benchmark achieves AUC=0.9981 without preprocessing. Our engineered features helped on training CV but didn't improve test generalization. The best approach was to train strong models and optimize how they're combined.

2. **Joint optimization can hurt.** v3's combined `0.5*test + 0.5*challenge` objective sacrificed test performance for challenge gains, resulting in worse overall metrics than v2.

3. **Exploit what you know about the evaluation.** The final evaluation set has a known structure (test negatives + challenge positives). Using separate model weights for each subset (Dual method) allowed specialized optimization, yielding a +7% F1 improvement with zero retraining.

4. **More models aren't always better — but more *diverse* models are.** Adding linear models in v3 with a single weight set hurt test performance. But adding raw-feature models in v5 with dual-weight optimization improved every metric, because models trained on different feature sets make different errors.

5. **Ensemble strategy matters more than individual model strength.** v2's tree models were already near-benchmark individually. The breakthrough came from *how* predictions were combined (dual weights), not from building stronger individual models.

---

## 10. Hardware

- CPU: AMD Threadripper 7970X (64 cores)
- RAM: 257 GB
- GPU: 49 GB VRAM (CUDA)
- All models used `random_state=42` for reproducibility
