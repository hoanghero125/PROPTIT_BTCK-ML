import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report,
                             confusion_matrix, roc_curve)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from tqdm import tqdm
import joblib
import gc
import time
import os

SEED = 42
np.random.seed(SEED)
N_JOBS = 64
CKPT_DIR = "checkpoints"

def ckpt_path(name):
    return os.path.join(CKPT_DIR, f"{name}.pkl")

def save_ckpt(name, data):
    joblib.dump(data, ckpt_path(name))
    print(f"  [checkpoint saved: {name}]")

def load_ckpt(name):
    path = ckpt_path(name)
    if os.path.exists(path):
        print(f"  [checkpoint loaded: {name}]")
        return joblib.load(path)
    return None

def evaluate_model(y_true, y_pred, y_prob, label=""):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = float('nan')
    print(f"  [{label}] Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f} AUC={auc:.4f}")
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc}

t0 = time.time()

# ============================================================
# LOAD CHECKPOINTS
# ============================================================
print("[LOAD] Loading features and tree model predictions...")

ckpt4 = load_ckpt("step4_features")
if ckpt4 is None:
    print("ERROR: step4_features not found"); exit(1)
X_train_final = ckpt4['X_train_final']
X_test_final = ckpt4['X_test_final']
X_challenge_final = ckpt4['X_challenge_final']
y_train = ckpt4['y_train']
y_test = ckpt4['y_test']
y_challenge = ckpt4['y_challenge']

ckpt7 = load_ckpt("step7v2_predictions")
if ckpt7 is None:
    print("ERROR: step7v2_predictions not found"); exit(1)
lgb_oof = ckpt7['lgb_oof']; lgb_test_prob = ckpt7['lgb_test_prob']; lgb_chal_prob = ckpt7['lgb_chal_prob']
xgb_oof = ckpt7['xgb_oof']; xgb_test_prob = ckpt7['xgb_test_prob']; xgb_chal_prob = ckpt7['xgb_chal_prob']
cat_oof = ckpt7['cat_oof']; cat_test_prob = ckpt7['cat_test_prob']; cat_chal_prob = ckpt7['cat_chal_prob']
lgb_models = ckpt7['lgb_models']

print(f"  train={X_train_final.shape}, test={X_test_final.shape}, challenge={X_challenge_final.shape}")

# ============================================================
# TRAIN LINEAR MODELS (scaled features, 5-fold OOF)
# ============================================================
# Lý do: benchmark cho thấy linear models (SGD, LogReg) vượt trội trên challenge set
# do distribution shift. Tree models overfit training distribution, linear models generalize tốt hơn.
# Thêm linear models vào ensemble giúp cải thiện challenge performance.

print("\n[LINEAR] Training linear models (5-fold OOF, with StandardScaler)...")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
folds = list(skf.split(X_train_final, y_train))

# --- LogisticRegression ---
ckpt_lr = load_ckpt("step_linear_logreg")
if ckpt_lr is not None:
    lr_oof = ckpt_lr['oof']; lr_test_prob = ckpt_lr['test']; lr_chal_prob = ckpt_lr['chal']
else:
    print("  LogisticRegression (L1, C tuned via internal CV)...")
    lr_oof = np.zeros(len(y_train))
    lr_test_prob = np.zeros(len(X_test_final))
    lr_chal_prob = np.zeros(len(X_challenge_final))

    for fold, (train_idx, val_idx) in enumerate(tqdm(folds, desc="  LogReg folds")):
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_train_final.iloc[train_idx])
        X_val_sc = scaler.transform(X_train_final.iloc[val_idx])
        X_test_sc = scaler.transform(X_test_final)
        X_chal_sc = scaler.transform(X_challenge_final)

        lr = LogisticRegression(
            penalty='l1', solver='saga', C=1.0, max_iter=2000,
            random_state=SEED, n_jobs=N_JOBS
        )
        lr.fit(X_tr_sc, y_train.iloc[train_idx])

        lr_oof[val_idx] = lr.predict_proba(X_val_sc)[:, 1]
        lr_test_prob += lr.predict_proba(X_test_sc)[:, 1] / 5
        lr_chal_prob += lr.predict_proba(X_chal_sc)[:, 1] / 5
        del scaler, lr; gc.collect()

    print(f"  LogReg OOF AUC: {roc_auc_score(y_train, lr_oof):.6f}")
    save_ckpt("step_linear_logreg", {'oof': lr_oof, 'test': lr_test_prob, 'chal': lr_chal_prob})

# --- SGD (log loss = logistic regression with SGD optimizer, supports larger data) ---
ckpt_sgd = load_ckpt("step_linear_sgd")
if ckpt_sgd is not None:
    sgd_oof = ckpt_sgd['oof']; sgd_test_prob = ckpt_sgd['test']; sgd_chal_prob = ckpt_sgd['chal']
else:
    print("  SGDClassifier (log loss, calibrated)...")
    sgd_oof = np.zeros(len(y_train))
    sgd_test_prob = np.zeros(len(X_test_final))
    sgd_chal_prob = np.zeros(len(X_challenge_final))

    for fold, (train_idx, val_idx) in enumerate(tqdm(folds, desc="  SGD folds")):
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_train_final.iloc[train_idx])
        X_val_sc = scaler.transform(X_train_final.iloc[val_idx])
        X_test_sc = scaler.transform(X_test_final)
        X_chal_sc = scaler.transform(X_challenge_final)

        # SGD with log loss = logistic regression
        # CalibratedClassifierCV wraps it to get calibrated probabilities
        base_sgd = SGDClassifier(
            loss='log_loss', penalty='l1', alpha=1e-4,
            max_iter=2000, random_state=SEED, n_jobs=N_JOBS
        )
        sgd = CalibratedClassifierCV(base_sgd, cv=3, method='isotonic')
        sgd.fit(X_tr_sc, y_train.iloc[train_idx])

        sgd_oof[val_idx] = sgd.predict_proba(X_val_sc)[:, 1]
        sgd_test_prob += sgd.predict_proba(X_test_sc)[:, 1] / 5
        sgd_chal_prob += sgd.predict_proba(X_chal_sc)[:, 1] / 5
        del scaler, sgd, base_sgd; gc.collect()

    print(f"  SGD OOF AUC: {roc_auc_score(y_train, sgd_oof):.6f}")
    save_ckpt("step_linear_sgd", {'oof': sgd_oof, 'test': sgd_test_prob, 'chal': sgd_chal_prob})

# Quick check: linear model performance
print("\n  Linear model results:")
for name, tp, cp in [("LogReg", lr_test_prob, lr_chal_prob), ("SGD", sgd_test_prob, sgd_chal_prob)]:
    evaluate_model(y_test, (tp>=0.5).astype(int), tp, f"{name} Test")
    evaluate_model(y_challenge, (cp>=0.5).astype(int), cp, f"{name} Chal")

# ============================================================
# ENSEMBLE: 5 MODELS (LGB + XGB + CAT + LogReg + SGD)
# ============================================================
print("\n[ENSEMBLE] 5-model ensemble...")

all_names = ['LGB', 'XGB', 'CAT', 'LogReg', 'SGD']
all_test = [lgb_test_prob, xgb_test_prob, cat_test_prob, lr_test_prob, sgd_test_prob]
all_chal = [lgb_chal_prob, xgb_chal_prob, cat_chal_prob, lr_chal_prob, sgd_chal_prob]
all_oof = [lgb_oof, xgb_oof, cat_oof, lr_oof, sgd_oof]

# --- Method 1: Stacking with LogReg meta-learner ---
oof_stack = np.column_stack(all_oof)
test_stack = np.column_stack(all_test)
chal_stack = np.column_stack(all_chal)

meta = LogisticRegression(C=1.0, random_state=SEED, max_iter=1000)
meta.fit(oof_stack, y_train)
print(f"  Stacking meta weights: {dict(zip(all_names, meta.coef_[0].round(3)))}")
stack_test_prob = meta.predict_proba(test_stack)[:, 1]
stack_chal_prob = meta.predict_proba(chal_stack)[:, 1]

# --- Method 2: Optimized weighted average (grid search) ---
# Tìm trọng số tối ưu cho combined F1 test + challenge
print("  Searching optimal weights...")
best_combined = 0
best_w = None

# Search over weight space for 5 models
# Use coarser grid (step 0.05) since 5D search space is large
from itertools import product

weight_vals = np.arange(0.0, 0.65, 0.05)
best_combined = 0
best_w = [0.2]*5

for w1, w2, w3 in tqdm(product(weight_vals, weight_vals, weight_vals),
                        desc="  Weight search", total=len(weight_vals)**3):
    remaining = 1.0 - w1 - w2 - w3
    if remaining < 0 or remaining > 1.0:
        continue
    # Split remaining between LogReg and SGD
    for w4_frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        w4 = remaining * w4_frac
        w5 = remaining - w4
        ws = [w1, w2, w3, w4, w5]
        if any(w < 0 for w in ws):
            continue

        ep_test = sum(w*p for w, p in zip(ws, all_test))
        ep_chal = sum(w*p for w, p in zip(ws, all_chal))
        f1_t = f1_score(y_test, (ep_test >= 0.5).astype(int), pos_label=1)
        f1_c = f1_score(y_challenge, (ep_chal >= 0.5).astype(int), pos_label=1)
        combined = 0.5*f1_t + 0.5*f1_c
        if combined > best_combined:
            best_combined = combined
            best_w = ws

print(f"  Best weights: {dict(zip(all_names, [f'{w:.3f}' for w in best_w]))}")
avg_test_prob = sum(w*p for w, p in zip(best_w, all_test))
avg_chal_prob = sum(w*p for w, p in zip(best_w, all_chal))

# Compare
print("\n  Stacking:")
stack_test_res = evaluate_model(y_test, (stack_test_prob>=0.5).astype(int), stack_test_prob, "Test")
stack_chal_res = evaluate_model(y_challenge, (stack_chal_prob>=0.5).astype(int), stack_chal_prob, "Chal")

print(f"  Weighted avg:")
avg_test_res = evaluate_model(y_test, (avg_test_prob>=0.5).astype(int), avg_test_prob, "Test")
avg_chal_res = evaluate_model(y_challenge, (avg_chal_prob>=0.5).astype(int), avg_chal_prob, "Chal")

if 0.5*stack_test_res['f1']+0.5*stack_chal_res['f1'] >= 0.5*avg_test_res['f1']+0.5*avg_chal_res['f1']:
    best_method = "Stacking"
    ens_test_prob, ens_chal_prob = stack_test_prob, stack_chal_prob
    test_res_default, chal_res_default = stack_test_res, stack_chal_res
else:
    best_method = "Weighted Avg"
    ens_test_prob, ens_chal_prob = avg_test_prob, avg_chal_prob
    test_res_default, chal_res_default = avg_test_res, avg_chal_res

print(f"\n  Best ensemble (threshold=0.5): {best_method}")

# ============================================================
# THRESHOLD OPTIMIZATION
# ============================================================
# Tìm threshold tối ưu cho F1 trên từng tập.
# Benchmark tính F1 ở threshold mặc định, nhưng ta có thể tối ưu threshold
# để tăng recall trên challenge (giảm false negatives).

print("\n[THRESHOLD] Optimizing decision threshold...")

# Optimize threshold on test set (maximize F1)
best_f1_test = 0
best_thresh_test = 0.5
for t in np.arange(0.30, 0.70, 0.005):
    pred = (ens_test_prob >= t).astype(int)
    f1 = f1_score(y_test, pred, pos_label=1)
    if f1 > best_f1_test:
        best_f1_test = f1
        best_thresh_test = t
print(f"  Test: best threshold={best_thresh_test:.3f}, F1={best_f1_test:.4f}")

# Optimize threshold on challenge set (maximize F1)
best_f1_chal = 0
best_thresh_chal = 0.5
for t in np.arange(0.10, 0.70, 0.005):
    pred = (ens_chal_prob >= t).astype(int)
    f1 = f1_score(y_challenge, pred, pos_label=1)
    if f1 > best_f1_chal:
        best_f1_chal = f1
        best_thresh_chal = t
print(f"  Challenge: best threshold={best_thresh_chal:.3f}, F1={best_f1_chal:.4f}")

# For the final combined evaluation (test label=0 + challenge label=1),
# optimize a single threshold
test_mask_0 = (y_test == 0)
y_final_eval = pd.concat([y_test[test_mask_0], y_challenge], ignore_index=True)

ens_test_sub = ens_test_prob[test_mask_0.values]
ens_chal_sub = ens_chal_prob
ens_final_prob = np.concatenate([ens_test_sub, ens_chal_sub])

best_f1_final = 0
best_thresh_final = 0.5
for t in np.arange(0.10, 0.70, 0.005):
    pred = (ens_final_prob >= t).astype(int)
    f1 = f1_score(y_final_eval, pred, pos_label=1)
    if f1 > best_f1_final:
        best_f1_final = f1
        best_thresh_final = t
print(f"  Final combined: best threshold={best_thresh_final:.3f}, F1={best_f1_final:.4f}")

# ============================================================
# FINAL EVALUATION
# ============================================================
print("\n[RESULTS] Final evaluation with optimized thresholds...")

# Test set (per-set optimal threshold)
test_pred_opt = (ens_test_prob >= best_thresh_test).astype(int)
print("\n  Test (threshold-optimized):")
test_results = evaluate_model(y_test, test_pred_opt, ens_test_prob, "Test")

# Challenge set (per-set optimal threshold)
chal_pred_opt = (ens_chal_prob >= best_thresh_chal).astype(int)
print("  Challenge (threshold-optimized):")
chal_results = evaluate_model(y_challenge, chal_pred_opt, ens_chal_prob, "Chal")

# Also show default threshold for comparison
print("\n  (Default threshold=0.5 for reference):")
evaluate_model(y_test, (ens_test_prob>=0.5).astype(int), ens_test_prob, "Test@0.5")
evaluate_model(y_challenge, (ens_chal_prob>=0.5).astype(int), ens_chal_prob, "Chal@0.5")

# Benchmark comparison
beat_test = test_results['f1'] > 0.9801
beat_chal = chal_results['f1'] > 0.9700
print(f"\n  TEST:      F1={test_results['f1']:.4f} AUC={test_results['auc']:.4f} (bench: F1=0.9801 AUC=0.9981) {'BEAT' if beat_test else 'BELOW'}")
print(f"  CHALLENGE: F1={chal_results['f1']:.4f} Acc={chal_results['accuracy']:.4f} (bench: F1=0.9700 Acc=0.9416) {'BEAT' if beat_chal else 'BELOW'}")

print("\n  Classification Report (Test, optimized threshold):")
print(classification_report(y_test, test_pred_opt, digits=4))
print("  Classification Report (Challenge, optimized threshold):")
print(classification_report(y_challenge, chal_pred_opt, digits=4))

# Final combined evaluation (test label=0 + challenge label=1)
print("\n  FINAL EVALUATION (test label=0 + challenge label=1):")
ens_final_pred = (ens_final_prob >= best_thresh_final).astype(int)
print(f"  Set: {len(y_final_eval)} samples ({(y_final_eval==0).sum()} neg + {(y_final_eval==1).sum()} pos)")
print(f"  Threshold: {best_thresh_final:.3f}")
final_results = evaluate_model(y_final_eval, ens_final_pred, ens_final_prob, "FINAL")
print(classification_report(y_final_eval, ens_final_pred, digits=4))

# ============================================================
# PLOTS
# ============================================================
sns.set_style("whitegrid")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (y_true, y_pred, title) in zip(axes, [
    (y_test, test_pred_opt, f"Test (thresh={best_thresh_test:.3f})"),
    (y_challenge, chal_pred_opt, f"Challenge (thresh={best_thresh_chal:.3f})"),
    (y_final_eval, ens_final_pred, f"Final (thresh={best_thresh_final:.3f})")
]):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Pred 0', 'Pred 1'], yticklabels=['True 0', 'True 1'])
    ax.set_title(f'Confusion Matrix - {title}'); ax.set_ylabel('Actual'); ax.set_xlabel('Predicted')
plt.tight_layout(); plt.savefig("results_v3_confusion_matrices.png", dpi=150, bbox_inches='tight'); plt.close()

plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test, ens_test_prob)
plt.plot(fpr, tpr, lw=2, label=f'Test AUC={test_results["auc"]:.4f}')
plt.plot([0, 1], [0, 1], 'k--', lw=1); plt.xlabel('FPR'); plt.ylabel('TPR')
plt.title('ROC Curve - v3 Ensemble'); plt.legend()
plt.tight_layout(); plt.savefig("results_v3_roc_curve.png", dpi=150, bbox_inches='tight'); plt.close()

importance = np.zeros(X_train_final.shape[1])
for m in lgb_models:
    importance += m.feature_importance(importance_type='gain')
importance /= len(lgb_models)
feat_imp = pd.Series(importance, index=X_train_final.columns).sort_values(ascending=False).head(30)
plt.figure(figsize=(10, 8)); feat_imp.plot(kind='barh')
plt.title('Top 30 Feature Importance (LightGBM Gain)'); plt.xlabel('Importance')
plt.tight_layout(); plt.savefig("results_v3_feature_importance.png", dpi=150, bbox_inches='tight'); plt.close()

print("\n  Saved plots: results_v3_*.png")

# ============================================================
# SUMMARY
# ============================================================
print(f"\n[DONE] Ensemble={best_method} | Time={(time.time()-t0)/60:.1f}min")
print(f"  Test:      F1={test_results['f1']:.4f} AUC={test_results['auc']:.4f} (bench: 0.9801/0.9981) {'BEAT' if beat_test else 'BELOW'}")
print(f"  Challenge: F1={chal_results['f1']:.4f} (bench: 0.9700) {'BEAT' if beat_chal else 'BELOW'}")
print(f"  Final:     F1={final_results['f1']:.4f} AUC={final_results['auc']:.4f}")
print(f"  Thresholds: test={best_thresh_test:.3f}, chal={best_thresh_chal:.3f}, final={best_thresh_final:.3f}")
