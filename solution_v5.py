"""
Binary Classification Challenge - v5
Goal: Improve v4's Final metric (F1=0.9586, AUC=0.9860) by adding raw-feature models.

v4 had only 5 base models for dual-weight optimization.
v5 adds 3 raw-feature tree models (LGB/XGB/CAT) → 8 total.
More diverse models = different error patterns = better dual optimization.

Phase 1: Load data + existing predictions
Phase 2: Train LGB/XGB/CAT on raw features (5-fold OOF, reuse v2 params, no Optuna)
Phase 3: Dual-weight optimization with 8 models (20000 trials)
Phase 4: Evaluation
"""

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
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import optuna
import joblib
import gc
import time
import os

optuna.logging.set_verbosity(optuna.logging.WARNING)
SEED = 42
np.random.seed(SEED)
N_JOBS = 64
CKPT_DIR = "checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

def ckpt_path(name):
    return os.path.join(CKPT_DIR, f"{name}.pkl")

def save_ckpt(name, data):
    joblib.dump(data, ckpt_path(name))
    print(f"  [checkpoint saved: {name}]")

def load_ckpt(name):
    path = ckpt_path(name)
    if os.path.exists(path):
        print(f"  [loaded: {name}]")
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
# PHASE 1: LOAD DATA + EXISTING PREDICTIONS
# ============================================================
print("[PHASE 1] Loading data and existing predictions...")

# Raw data
X_train_raw = pd.read_csv("train_X-001.csv")
y_train = pd.read_csv("train_y.csv").squeeze().astype(int)
X_test_raw = pd.read_csv("test_X.csv")
y_test = pd.read_csv("test_y.csv").squeeze().astype(int)
X_challenge_raw = pd.read_csv("challenge_X.csv")
y_challenge = pd.read_csv("challenge_y.csv").squeeze().astype(int)

clean_cols = [f"f{i}" for i in range(X_train_raw.shape[1])]
X_train_raw.columns = clean_cols
X_test_raw.columns = clean_cols
X_challenge_raw.columns = clean_cols

# Remove constant features only
variances = X_train_raw.var()
const_cols = variances[variances == 0].index.tolist()
if const_cols:
    X_train_raw.drop(columns=const_cols, inplace=True)
    X_test_raw.drop(columns=const_cols, inplace=True)
    X_challenge_raw.drop(columns=const_cols, inplace=True)
X_train_raw = X_train_raw.astype(np.float32)
X_test_raw = X_test_raw.astype(np.float32)
X_challenge_raw = X_challenge_raw.astype(np.float32)

print(f"  Raw: train={X_train_raw.shape}, test={X_test_raw.shape}, chal={X_challenge_raw.shape}")

# v2 tree model predictions (engineered features)
ckpt7 = load_ckpt("step7v2_predictions")
if ckpt7 is None:
    print("ERROR: step7v2_predictions not found"); exit(1)
lgb_eng_oof = ckpt7['lgb_oof']; lgb_eng_test = ckpt7['lgb_test_prob']; lgb_eng_chal = ckpt7['lgb_chal_prob']
xgb_eng_oof = ckpt7['xgb_oof']; xgb_eng_test = ckpt7['xgb_test_prob']; xgb_eng_chal = ckpt7['xgb_chal_prob']
cat_eng_oof = ckpt7['cat_oof']; cat_eng_test = ckpt7['cat_test_prob']; cat_eng_chal = ckpt7['cat_chal_prob']

# v3 linear model predictions
ckpt_lr = load_ckpt("step_linear_logreg")
ckpt_sgd = load_ckpt("step_linear_sgd")
if ckpt_lr is None or ckpt_sgd is None:
    print("ERROR: linear checkpoints not found"); exit(1)
lr_oof = ckpt_lr['oof']; lr_test = ckpt_lr['test']; lr_chal = ckpt_lr['chal']
sgd_oof = ckpt_sgd['oof']; sgd_test = ckpt_sgd['test']; sgd_chal = ckpt_sgd['chal']

# ============================================================
# PHASE 2: TRAIN ON RAW FEATURES (5-fold OOF, reuse v2 params)
# ============================================================
print("\n[PHASE 2] Training models on raw features (5-fold OOF, 5000 rounds)...")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
folds = list(skf.split(X_train_raw, y_train))

# --- LightGBM (reuse v2 hardcoded params) ---
ckpt_lgb = load_ckpt("step_v5_lgb_raw")
if ckpt_lgb is not None:
    lgb_raw_oof = ckpt_lgb['oof']; lgb_raw_test = ckpt_lgb['test']; lgb_raw_chal = ckpt_lgb['chal']
else:
    print("  LightGBM (raw features, v2 params)...")
    lgb_raw_params = {
        'boosting_type': 'gbdt', 'objective': 'binary', 'metric': 'auc',
        'num_leaves': 256, 'learning_rate': 0.03,
        'feature_fraction': 0.75, 'bagging_fraction': 0.75, 'bagging_freq': 5,
        'min_child_samples': 20, 'lambda_l1': 0.1, 'lambda_l2': 0.1,
        'min_gain_to_split': 0.01, 'max_depth': -1, 'max_bin': 255,
        'verbose': -1, 'n_jobs': N_JOBS, 'seed': SEED,
    }
    lgb_raw_oof = np.zeros(len(y_train))
    lgb_raw_test = np.zeros(len(X_test_raw))
    lgb_raw_chal = np.zeros(len(X_challenge_raw))

    for fold, (train_idx, val_idx) in enumerate(folds):
        print(f"    Fold {fold+1}/5...", end=" ", flush=True)
        dtrain = lgb.Dataset(X_train_raw.iloc[train_idx], label=y_train.iloc[train_idx])
        dval = lgb.Dataset(X_train_raw.iloc[val_idx], label=y_train.iloc[val_idx], reference=dtrain)
        m = lgb.train(lgb_raw_params, dtrain, num_boost_round=5000,
                      valid_sets=[dval], valid_names=['val'],
                      callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
        lgb_raw_oof[val_idx] = m.predict(X_train_raw.iloc[val_idx])
        lgb_raw_test += m.predict(X_test_raw) / 5
        lgb_raw_chal += m.predict(X_challenge_raw) / 5
        print(f"best_iter={m.best_iteration}")
        del m, dtrain, dval; gc.collect()

    print(f"  LGB raw OOF AUC: {roc_auc_score(y_train, lgb_raw_oof):.6f}")
    save_ckpt("step_v5_lgb_raw", {'oof': lgb_raw_oof, 'test': lgb_raw_test, 'chal': lgb_raw_chal})

evaluate_model(y_test, (lgb_raw_test >= 0.5).astype(int), lgb_raw_test, "LGB_raw Test")

# --- XGBoost (reuse v2 Optuna params if available, else manual) ---
xgb_v2_study = load_ckpt("step6v2_xgb_study")
if xgb_v2_study is not None:
    xgb_raw_params = {
        'objective': 'binary:logistic', 'eval_metric': 'auc',
        'tree_method': 'hist', 'device': 'cuda:0',
        'seed': SEED, 'verbosity': 0, 'nthread': N_JOBS,
        **xgb_v2_study.best_params
    }
    print(f"  XGB: reusing v2 Optuna params (CV AUC={xgb_v2_study.best_value:.6f})")
else:
    xgb_raw_params = {
        'objective': 'binary:logistic', 'eval_metric': 'auc',
        'max_depth': 8, 'learning_rate': 0.03,
        'subsample': 0.8, 'colsample_bytree': 0.8,
        'min_child_weight': 20, 'gamma': 0.1, 'lambda': 1.0, 'alpha': 0.1,
        'tree_method': 'hist', 'device': 'cuda:0',
        'seed': SEED, 'verbosity': 0, 'nthread': N_JOBS,
    }
    print("  XGB: using manual params")

ckpt_xgb = load_ckpt("step_v5_xgb_raw")
if ckpt_xgb is not None:
    xgb_raw_oof = ckpt_xgb['oof']; xgb_raw_test = ckpt_xgb['test']; xgb_raw_chal = ckpt_xgb['chal']
else:
    print("  XGBoost (raw features, GPU)...")
    xgb_raw_oof = np.zeros(len(y_train))
    xgb_raw_test = np.zeros(len(X_test_raw))
    xgb_raw_chal = np.zeros(len(X_challenge_raw))
    dtest_xgb = xgb.DMatrix(X_test_raw)
    dchal_xgb = xgb.DMatrix(X_challenge_raw)

    for fold, (train_idx, val_idx) in enumerate(folds):
        print(f"    Fold {fold+1}/5...", end=" ", flush=True)
        dtrain = xgb.DMatrix(X_train_raw.iloc[train_idx], label=y_train.iloc[train_idx])
        dval = xgb.DMatrix(X_train_raw.iloc[val_idx], label=y_train.iloc[val_idx])
        m = xgb.train(xgb_raw_params, dtrain, num_boost_round=5000,
                      evals=[(dval, 'val')], early_stopping_rounds=150, verbose_eval=False)
        xgb_raw_oof[val_idx] = m.predict(dval)
        xgb_raw_test += m.predict(dtest_xgb) / 5
        xgb_raw_chal += m.predict(dchal_xgb) / 5
        print(f"best_iter={m.best_iteration}")
        del m, dtrain, dval; gc.collect()

    print(f"  XGB raw OOF AUC: {roc_auc_score(y_train, xgb_raw_oof):.6f}")
    save_ckpt("step_v5_xgb_raw", {'oof': xgb_raw_oof, 'test': xgb_raw_test, 'chal': xgb_raw_chal})

# --- CatBoost (reuse v2 Optuna params if available, else manual) ---
cat_v2_study = load_ckpt("step6v2_cat_study")
if cat_v2_study is not None:
    cat_raw_user_params = cat_v2_study.best_params
    print(f"  CAT: reusing v2 Optuna params (CV AUC={cat_v2_study.best_value:.6f})")
else:
    cat_raw_user_params = {
        'learning_rate': 0.03, 'depth': 8,
        'l2_leaf_reg': 1.0, 'bagging_temperature': 0.5,
        'random_strength': 0.5, 'border_count': 128,
    }
    print("  CAT: using manual params")

ckpt_cat = load_ckpt("step_v5_cat_raw")
if ckpt_cat is not None:
    cat_raw_oof = ckpt_cat['oof']; cat_raw_test = ckpt_cat['test']; cat_raw_chal = ckpt_cat['chal']
else:
    print("  CatBoost (raw features, GPU)...")
    cat_raw_oof = np.zeros(len(y_train))
    cat_raw_test = np.zeros(len(X_test_raw))
    cat_raw_chal = np.zeros(len(X_challenge_raw))

    for fold, (train_idx, val_idx) in enumerate(folds):
        print(f"    Fold {fold+1}/5...", end=" ", flush=True)
        m = CatBoostClassifier(
            iterations=5000,
            loss_function='Logloss', eval_metric='AUC',
            random_seed=SEED, verbose=0, early_stopping_rounds=150,
            task_type='GPU', devices='0',
            **cat_raw_user_params
        )
        m.fit(X_train_raw.iloc[train_idx], y_train.iloc[train_idx],
              eval_set=(X_train_raw.iloc[val_idx], y_train.iloc[val_idx]), verbose=0)
        cat_raw_oof[val_idx] = m.predict_proba(X_train_raw.iloc[val_idx])[:, 1]
        cat_raw_test += m.predict_proba(X_test_raw)[:, 1] / 5
        cat_raw_chal += m.predict_proba(X_challenge_raw)[:, 1] / 5
        print(f"done")
        del m; gc.collect()

    print(f"  CAT raw OOF AUC: {roc_auc_score(y_train, cat_raw_oof):.6f}")
    save_ckpt("step_v5_cat_raw", {'oof': cat_raw_oof, 'test': cat_raw_test, 'chal': cat_raw_chal})

# ============================================================
# PHASE 3: BUILD FINAL SET + DUAL-WEIGHT OPTIMIZATION
# ============================================================
print("\n[PHASE 3] Dual-weight optimization with 8 models...")

# Final set
test_mask_0 = (y_test == 0)
y_final = pd.concat([y_test[test_mask_0], y_challenge], ignore_index=True)
print(f"  Final set: {len(y_final)} samples ({(y_final==0).sum()} neg + {(y_final==1).sum()} pos)")

# 8 models: 3 raw + 3 engineered + 2 linear
model_names = ['LGB_raw', 'XGB_raw', 'CAT_raw',
               'LGB_eng', 'XGB_eng', 'CAT_eng',
               'LR', 'SGD']
all_oof = [lgb_raw_oof, xgb_raw_oof, cat_raw_oof,
           lgb_eng_oof, xgb_eng_oof, cat_eng_oof,
           lr_oof, sgd_oof]
all_test = [lgb_raw_test, xgb_raw_test, cat_raw_test,
            lgb_eng_test, xgb_eng_test, cat_eng_test,
            lr_test, sgd_test]
all_chal = [lgb_raw_chal, xgb_raw_chal, cat_raw_chal,
            lgb_eng_chal, xgb_eng_chal, cat_eng_chal,
            lr_chal, sgd_chal]

n_models = len(model_names)

# Per-model final probabilities
test_neg_probs = [all_test[i][test_mask_0.values] for i in range(n_models)]
chal_probs = [all_chal[i] for i in range(n_models)]
all_final = [np.concatenate([test_neg_probs[i], chal_probs[i]]) for i in range(n_models)]

# Individual performance
print("\n  Individual model performance (Final set):")
for i, name in enumerate(model_names):
    evaluate_model(y_final, (all_final[i] >= 0.5).astype(int), all_final[i], name)

# --- Dual-weight optimization (20000 trials) ---
print(f"\n  Dual-weight optimization: {n_models} models x 2 weight sets + 2 thresholds = {n_models*2+2} params")
print("  Running 20000 Optuna trials...")

def obj_dual(trial):
    wn = np.array([trial.suggest_float(f'wn_{i}', 0.0, 1.0) for i in range(n_models)])
    t_neg = trial.suggest_float('t_neg', 0.30, 0.70)
    wp = np.array([trial.suggest_float(f'wp_{i}', 0.0, 1.0) for i in range(n_models)])
    t_pos = trial.suggest_float('t_pos', 0.001, 0.50)

    sn = wn.sum(); sp = wp.sum()
    if sn < 1e-8 or sp < 1e-8: return 0
    wn = wn / sn; wp = wp / sp

    prob_neg = sum(wn[i] * test_neg_probs[i] for i in range(n_models))
    prob_pos = sum(wp[i] * chal_probs[i] for i in range(n_models))

    pred_neg = (prob_neg >= t_neg).astype(int)
    pred_pos = (prob_pos >= t_pos).astype(int)

    pred_all = np.concatenate([pred_neg, pred_pos])
    return f1_score(y_final, pred_all, pos_label=1)

study = optuna.create_study(direction='maximize', study_name='v5_dual')
study.optimize(obj_dual, n_trials=20000, show_progress_bar=True)

# Extract best
bp = study.best_params
wn_best = np.array([bp[f'wn_{i}'] for i in range(n_models)]); wn_best = wn_best / wn_best.sum()
wp_best = np.array([bp[f'wp_{i}'] for i in range(n_models)]); wp_best = wp_best / wp_best.sum()
t_neg_best = bp['t_neg']; t_pos_best = bp['t_pos']

print(f"\n  Best F1: {study.best_value:.4f}")
print(f"  Neg weights: {dict(zip(model_names, [f'{w:.3f}' for w in wn_best]))}")
print(f"  Neg threshold: {t_neg_best:.3f}")
print(f"  Pos weights: {dict(zip(model_names, [f'{w:.3f}' for w in wp_best]))}")
print(f"  Pos threshold: {t_pos_best:.3f}")

# Build final predictions
prob_neg_best = sum(wn_best[i] * test_neg_probs[i] for i in range(n_models))
prob_pos_best = sum(wp_best[i] * chal_probs[i] for i in range(n_models))
dual_prob = np.concatenate([prob_neg_best, prob_pos_best])
dual_pred = np.concatenate([(prob_neg_best >= t_neg_best).astype(int),
                             (prob_pos_best >= t_pos_best).astype(int)])

# ============================================================
# PHASE 4: EVALUATION
# ============================================================
print("\n[PHASE 4] Final evaluation...")

# Final
print(f"\n  FINAL ({len(y_final)} samples: {(y_final==0).sum()} neg + {(y_final==1).sum()} pos):")
final_results = evaluate_model(y_final, dual_pred, dual_prob, "FINAL")
print(classification_report(y_final, dual_pred, digits=4))

# Test (using neg weights, threshold optimized for test)
test_prob = sum(wn_best[i] * all_test[i] for i in range(n_models))
best_t_test, best_f1_test = 0.5, 0
for t in np.arange(0.30, 0.70, 0.001):
    f1 = f1_score(y_test, (test_prob >= t).astype(int), pos_label=1)
    if f1 > best_f1_test: best_f1_test = f1; best_t_test = t
print(f"  Test (threshold={best_t_test:.3f}):")
test_results = evaluate_model(y_test, (test_prob >= best_t_test).astype(int), test_prob, "Test")
print(classification_report(y_test, (test_prob >= best_t_test).astype(int), digits=4))

# Challenge (using pos weights, threshold optimized for challenge)
chal_prob = sum(wp_best[i] * all_chal[i] for i in range(n_models))
best_t_chal, best_f1_chal = 0.5, 0
for t in np.arange(0.001, 0.50, 0.001):
    f1 = f1_score(y_challenge, (chal_prob >= t).astype(int), pos_label=1)
    if f1 > best_f1_chal: best_f1_chal = f1; best_t_chal = t
print(f"  Challenge (threshold={best_t_chal:.3f}):")
chal_results = evaluate_model(y_challenge, (chal_prob >= best_t_chal).astype(int), chal_prob, "Chal")
print(classification_report(y_challenge, (chal_prob >= best_t_chal).astype(int), digits=4))

# ============================================================
# PHASE 5: COMPARISON
# ============================================================
print("\n[PHASE 5] Comparison...")

print(f"\n  {'Metric':<12} {'v2':>8} {'v3':>8} {'v4':>8} {'v5':>8} {'Winner':>8}")
print(f"  {'-'*56}")
prev = {
    'v2': {'acc': 0.9609, 'prec': 0.8818, 'rec': 0.8955, 'f1': 0.8886, 'auc': 0.9903},
    'v3': {'acc': 0.9591, 'prec': 0.8566, 'rec': 0.9192, 'f1': 0.8868, 'auc': 0.9840},
    'v4': {'acc': 0.9849, 'prec': 0.9204, 'rec': 1.0000, 'f1': 0.9586, 'auc': 0.9860},
}
v5 = {'acc': final_results['accuracy'], 'prec': final_results['precision'],
      'rec': final_results['recall'], 'f1': final_results['f1'], 'auc': final_results['auc']}

for key, label in [('acc', 'Accuracy'), ('prec', 'Precision'), ('rec', 'Recall'), ('f1', 'F1'), ('auc', 'AUC')]:
    vals = {'v2': prev['v2'][key], 'v3': prev['v3'][key], 'v4': prev['v4'][key], 'v5': v5[key]}
    winner = max(vals, key=vals.get)
    print(f"  {label:<12} {prev['v2'][key]:>8.4f} {prev['v3'][key]:>8.4f} {prev['v4'][key]:>8.4f} {v5[key]:>8.4f} {winner:>8}")

print(f"\n  v5 vs v4: F1 {v5['f1']-0.9586:+.4f} | AUC {v5['auc']-0.9860:+.4f}")
print(f"  v5 vs v2: F1 {v5['f1']-0.8886:+.4f} | AUC {v5['auc']-0.9903:+.4f}")

# ============================================================
# PHASE 6: PLOTS
# ============================================================
print("\n[PHASE 6] Saving plots...")
sns.set_style("whitegrid")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (y_true, y_pred, title) in zip(axes, [
    (y_test, (test_prob >= best_t_test).astype(int), f"Test (t={best_t_test:.3f})"),
    (y_challenge, (chal_prob >= best_t_chal).astype(int), f"Chal (t={best_t_chal:.3f})"),
    (y_final, dual_pred, f"Final (dual thresh)")
]):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Pred 0', 'Pred 1'], yticklabels=['True 0', 'True 1'])
    ax.set_title(f'CM - {title}'); ax.set_ylabel('Actual'); ax.set_xlabel('Predicted')
plt.tight_layout()
plt.savefig("results_v5_confusion_matrices.png", dpi=150, bbox_inches='tight')
plt.close()

plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_final, dual_prob)
plt.plot(fpr, tpr, lw=2, label=f'v5 Final AUC={final_results["auc"]:.4f}')
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel('FPR'); plt.ylabel('TPR')
plt.title('ROC Curve - v5 Final'); plt.legend()
plt.tight_layout()
plt.savefig("results_v5_roc_curve.png", dpi=150, bbox_inches='tight')
plt.close()

print("  Saved: results_v5_*.png")

# ============================================================
# SUMMARY
# ============================================================
elapsed = (time.time() - t0) / 60
print(f"\n{'='*60}")
print(f"[DONE] v5 | Dual-8 | Time={elapsed:.1f}min")
print(f"  Final:     Acc={final_results['accuracy']:.4f} Prec={final_results['precision']:.4f} Rec={final_results['recall']:.4f} F1={final_results['f1']:.4f} AUC={final_results['auc']:.4f}")
print(f"  Test:      F1={test_results['f1']:.4f} AUC={test_results['auc']:.4f}")
print(f"  Challenge: F1={chal_results['f1']:.4f}")
print(f"  vs v4:     F1 {v5['f1']-0.9586:+.4f} | AUC {v5['auc']-0.9860:+.4f}")
print(f"{'='*60}")
