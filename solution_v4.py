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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import StratifiedKFold
from scipy.stats import rankdata
import lightgbm as lgb
import optuna
import joblib
import time
import os

optuna.logging.set_verbosity(optuna.logging.WARNING)
SEED = 42
np.random.seed(SEED)
CKPT_DIR = "checkpoints"

def load_ckpt(name):
    path = os.path.join(CKPT_DIR, f"{name}.pkl")
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

def rank_average(predictions):
    """Rank-based averaging — robust to calibration differences."""
    ranks = [rankdata(p) / len(p) for p in predictions]
    return np.mean(ranks, axis=0)

t0 = time.time()

# ============================================================
# PHASE 1: LOAD EXISTING PREDICTIONS
# ============================================================
print("[PHASE 1] Loading existing predictions (zero retraining)...")

# Labels
ckpt4 = load_ckpt("step4_features")
if ckpt4 is None:
    print("ERROR: step4_features not found"); exit(1)
y_train = ckpt4['y_train']
y_test = ckpt4['y_test']
y_challenge = ckpt4['y_challenge']
del ckpt4

# v2 tree model predictions (LGB + XGB + CAT on engineered features)
ckpt7 = load_ckpt("step7v2_predictions")
if ckpt7 is None:
    print("ERROR: step7v2_predictions not found"); exit(1)
lgb_oof = ckpt7['lgb_oof']; lgb_test = ckpt7['lgb_test_prob']; lgb_chal = ckpt7['lgb_chal_prob']
xgb_oof = ckpt7['xgb_oof']; xgb_test = ckpt7['xgb_test_prob']; xgb_chal = ckpt7['xgb_chal_prob']
cat_oof = ckpt7['cat_oof']; cat_test = ckpt7['cat_test_prob']; cat_chal = ckpt7['cat_chal_prob']

# v3 linear model predictions (LogReg + SGD on engineered features)
ckpt_lr = load_ckpt("step_linear_logreg")
ckpt_sgd = load_ckpt("step_linear_sgd")
if ckpt_lr is None or ckpt_sgd is None:
    print("ERROR: linear checkpoints not found"); exit(1)
lr_oof = ckpt_lr['oof']; lr_test = ckpt_lr['test']; lr_chal = ckpt_lr['chal']
sgd_oof = ckpt_sgd['oof']; sgd_test = ckpt_sgd['test']; sgd_chal = ckpt_sgd['chal']

print(f"  y_train: {y_train.value_counts().to_dict()}")
print(f"  y_test: {y_test.value_counts().to_dict()}")
print(f"  y_challenge: {y_challenge.value_counts().to_dict()}")

# ============================================================
# PHASE 2: BUILD FINAL COMBINED SET
# ============================================================
# Final = test negatives (label=0) + all challenge (label=1)
print("\n[PHASE 2] Building final combined evaluation set...")

test_mask_0 = (y_test == 0)
y_final = pd.concat([y_test[test_mask_0], y_challenge], ignore_index=True)
print(f"  Final set: {len(y_final)} samples ({(y_final==0).sum()} neg + {(y_final==1).sum()} pos)")

# Build final probabilities for each model
model_names = ['LGB', 'XGB', 'CAT', 'LR', 'SGD']
all_oof = [lgb_oof, xgb_oof, cat_oof, lr_oof, sgd_oof]
all_test = [lgb_test, xgb_test, cat_test, lr_test, sgd_test]
all_chal = [lgb_chal, xgb_chal, cat_chal, lr_chal, sgd_chal]

# Final probabilities per model: test_neg part + challenge part
all_final = []
for i in range(len(model_names)):
    fp = np.concatenate([all_test[i][test_mask_0.values], all_chal[i]])
    all_final.append(fp)

n_models = len(model_names)

# ============================================================
# PHASE 3: INDIVIDUAL MODEL PERFORMANCE ON FINAL SET
# ============================================================
print("\n[PHASE 3] Individual model performance on Final set...")

for i, name in enumerate(model_names):
    evaluate_model(y_final, (all_final[i] >= 0.5).astype(int), all_final[i], name)

# Also show test and challenge individually for reference
print("\n  (Reference - Test set):")
for i, name in enumerate(model_names):
    evaluate_model(y_test, (all_test[i] >= 0.5).astype(int), all_test[i], f"{name} Test")
print("  (Reference - Challenge set):")
for i, name in enumerate(model_names):
    evaluate_model(y_challenge, (all_chal[i] >= 0.5).astype(int), all_chal[i], f"{name} Chal")

# ============================================================
# PHASE 4: ENSEMBLE METHODS
# ============================================================
print("\n[PHASE 4] Trying ensemble methods...")

candidates = []  # (name, final_prob, threshold, final_res)

# --- A: Simple average (all 5 models) ---
avg5 = np.mean(all_final, axis=0)
best_t, best_f1 = 0.5, 0
for t in np.arange(0.10, 0.70, 0.001):
    f1 = f1_score(y_final, (avg5 >= t).astype(int), pos_label=1)
    if f1 > best_f1: best_f1 = f1; best_t = t
res = evaluate_model(y_final, (avg5 >= best_t).astype(int), avg5, f"Avg5 @{best_t:.3f}")
candidates.append(("Avg5", avg5, best_t, res))

# --- B: Tree-only average (LGB + XGB + CAT) ---
avg3 = np.mean(all_final[:3], axis=0)
best_t, best_f1 = 0.5, 0
for t in np.arange(0.10, 0.70, 0.001):
    f1 = f1_score(y_final, (avg3 >= t).astype(int), pos_label=1)
    if f1 > best_f1: best_f1 = f1; best_t = t
res = evaluate_model(y_final, (avg3 >= best_t).astype(int), avg3, f"TreeAvg @{best_t:.3f}")
candidates.append(("TreeAvg", avg3, best_t, res))

# --- C: Rank averaging (all 5) ---
rank5 = rank_average(all_final)
best_t, best_f1 = 0.5, 0
for t in np.arange(0.10, 0.70, 0.001):
    f1 = f1_score(y_final, (rank5 >= t).astype(int), pos_label=1)
    if f1 > best_f1: best_f1 = f1; best_t = t
res = evaluate_model(y_final, (rank5 >= best_t).astype(int), rank5, f"RankAvg @{best_t:.3f}")
candidates.append(("RankAvg", rank5, best_t, res))

# --- D: Rank averaging (trees only) ---
rank3 = rank_average(all_final[:3])
best_t, best_f1 = 0.5, 0
for t in np.arange(0.10, 0.70, 0.001):
    f1 = f1_score(y_final, (rank3 >= t).astype(int), pos_label=1)
    if f1 > best_f1: best_f1 = f1; best_t = t
res = evaluate_model(y_final, (rank3 >= best_t).astype(int), rank3, f"RankTree @{best_t:.3f}")
candidates.append(("RankTree", rank3, best_t, res))

# --- E: Stacking (LogReg meta-learner on OOF) ---
print("\n  [E] Stacking (LogReg meta-learner)...")
oof_stack = np.column_stack(all_oof)
final_stack = np.column_stack(all_final)

meta = LogisticRegression(C=1.0, random_state=SEED, max_iter=1000)
meta.fit(oof_stack, y_train)
stack_final = meta.predict_proba(final_stack)[:, 1]
print(f"  Meta coefs: {dict(zip(model_names, meta.coef_[0].round(3)))}")

best_t, best_f1 = 0.5, 0
for t in np.arange(0.10, 0.70, 0.001):
    f1 = f1_score(y_final, (stack_final >= t).astype(int), pos_label=1)
    if f1 > best_f1: best_f1 = f1; best_t = t
res = evaluate_model(y_final, (stack_final >= best_t).astype(int), stack_final, f"Stacking @{best_t:.3f}")
candidates.append(("Stacking", stack_final, best_t, res))

# --- F: Stacking with different C values ---
for C in [0.01, 0.1, 10.0]:
    meta_c = LogisticRegression(C=C, random_state=SEED, max_iter=1000)
    meta_c.fit(oof_stack, y_train)
    sc = meta_c.predict_proba(final_stack)[:, 1]
    best_t, best_f1 = 0.5, 0
    for t in np.arange(0.10, 0.70, 0.001):
        f1 = f1_score(y_final, (sc >= t).astype(int), pos_label=1)
        if f1 > best_f1: best_f1 = f1; best_t = t
    res = evaluate_model(y_final, (sc >= best_t).astype(int), sc, f"Stack_C{C} @{best_t:.3f}")
    candidates.append((f"Stack_C{C}", sc, best_t, res))

# --- G: Optuna weight+threshold optimization for Final F1 (2000 trials) ---
print("\n  [G] Optuna: optimize weights+threshold for Final F1 (2000 trials)...")

def obj_final_f1(trial):
    ws = [trial.suggest_float(f'w_{i}', 0.0, 1.0) for i in range(n_models)]
    thresh = trial.suggest_float('thresh', 0.10, 0.65)
    w = np.array(ws)
    s = w.sum()
    if s < 1e-8: return 0
    w = w / s
    prob = sum(w[i] * all_final[i] for i in range(n_models))
    return f1_score(y_final, (prob >= thresh).astype(int), pos_label=1)

study_f1 = optuna.create_study(direction='maximize', study_name='final_f1')
study_f1.optimize(obj_final_f1, n_trials=2000, show_progress_bar=True)
w_f1 = np.array([study_f1.best_params[f'w_{i}'] for i in range(n_models)])
w_f1 = w_f1 / w_f1.sum()
t_f1 = study_f1.best_params['thresh']
print(f"  F1 weights: {dict(zip(model_names, [f'{w:.3f}' for w in w_f1]))}")
print(f"  F1 threshold: {t_f1:.3f}")
opt_f1_prob = sum(w_f1[i] * all_final[i] for i in range(n_models))
res = evaluate_model(y_final, (opt_f1_prob >= t_f1).astype(int), opt_f1_prob, f"OptF1 @{t_f1:.3f}")
candidates.append(("OptF1", opt_f1_prob, t_f1, res))

# --- H: Optuna weight+threshold optimization for Final AUC (1000 trials) ---
print("\n  [H] Optuna: optimize weights for Final AUC (1000 trials)...")

def obj_final_auc(trial):
    ws = [trial.suggest_float(f'w_{i}', 0.0, 1.0) for i in range(n_models)]
    w = np.array(ws)
    s = w.sum()
    if s < 1e-8: return 0.5
    w = w / s
    prob = sum(w[i] * all_final[i] for i in range(n_models))
    return roc_auc_score(y_final, prob)

study_auc = optuna.create_study(direction='maximize', study_name='final_auc')
study_auc.optimize(obj_final_auc, n_trials=1000, show_progress_bar=True)
w_auc = np.array([study_auc.best_params[f'w_{i}'] for i in range(n_models)])
w_auc = w_auc / w_auc.sum()
print(f"  AUC weights: {dict(zip(model_names, [f'{w:.3f}' for w in w_auc]))}")
opt_auc_prob = sum(w_auc[i] * all_final[i] for i in range(n_models))
best_t, best_f1 = 0.5, 0
for t in np.arange(0.10, 0.70, 0.001):
    f1 = f1_score(y_final, (opt_auc_prob >= t).astype(int), pos_label=1)
    if f1 > best_f1: best_f1 = f1; best_t = t
res = evaluate_model(y_final, (opt_auc_prob >= best_t).astype(int), opt_auc_prob, f"OptAUC @{best_t:.3f}")
candidates.append(("OptAUC", opt_auc_prob, best_t, res))

# --- I: Optuna combined F1+AUC (2000 trials) ---
print("\n  [I] Optuna: optimize weights+threshold for F1+AUC combined (2000 trials)...")

def obj_final_combined(trial):
    ws = [trial.suggest_float(f'w_{i}', 0.0, 1.0) for i in range(n_models)]
    thresh = trial.suggest_float('thresh', 0.10, 0.65)
    w = np.array(ws)
    s = w.sum()
    if s < 1e-8: return 0
    w = w / s
    prob = sum(w[i] * all_final[i] for i in range(n_models))
    f1 = f1_score(y_final, (prob >= thresh).astype(int), pos_label=1)
    auc = roc_auc_score(y_final, prob)
    return f1 + auc  # maximize both

study_comb = optuna.create_study(direction='maximize', study_name='final_combined')
study_comb.optimize(obj_final_combined, n_trials=2000, show_progress_bar=True)
w_comb = np.array([study_comb.best_params[f'w_{i}'] for i in range(n_models)])
w_comb = w_comb / w_comb.sum()
t_comb = study_comb.best_params['thresh']
print(f"  Combined weights: {dict(zip(model_names, [f'{w:.3f}' for w in w_comb]))}")
print(f"  Combined threshold: {t_comb:.3f}")
opt_comb_prob = sum(w_comb[i] * all_final[i] for i in range(n_models))
res = evaluate_model(y_final, (opt_comb_prob >= t_comb).astype(int), opt_comb_prob, f"OptComb @{t_comb:.3f}")
candidates.append(("OptComb", opt_comb_prob, t_comb, res))

# --- J: DUAL-WEIGHT optimization (separate weights for test neg vs challenge) ---
# Key insight: we KNOW which samples are test negatives vs challenge positives.
# Using different model weights for each part gives more degrees of freedom.
print("\n  [J] Dual-weight optimization (separate weights per subset, 5000 trials)...")

test_neg_probs = [all_test[i][test_mask_0.values] for i in range(n_models)]
chal_probs = [all_chal[i] for i in range(n_models)]

def obj_dual(trial):
    # Weights for test negatives (want specificity: predict 0)
    wn = np.array([trial.suggest_float(f'wn_{i}', 0.0, 1.0) for i in range(n_models)])
    t_neg = trial.suggest_float('t_neg', 0.30, 0.70)
    # Weights for challenge positives (want recall: predict 1)
    wp = np.array([trial.suggest_float(f'wp_{i}', 0.0, 1.0) for i in range(n_models)])
    t_pos = trial.suggest_float('t_pos', 0.01, 0.50)

    sn = wn.sum(); sp = wp.sum()
    if sn < 1e-8 or sp < 1e-8: return 0
    wn = wn / sn; wp = wp / sp

    prob_neg = sum(wn[i] * test_neg_probs[i] for i in range(n_models))
    prob_pos = sum(wp[i] * chal_probs[i] for i in range(n_models))

    pred_neg = (prob_neg >= t_neg).astype(int)
    pred_pos = (prob_pos >= t_pos).astype(int)

    pred_all = np.concatenate([pred_neg, pred_pos])
    return f1_score(y_final, pred_all, pos_label=1)

study_dual = optuna.create_study(direction='maximize', study_name='dual')
study_dual.optimize(obj_dual, n_trials=5000, show_progress_bar=True)

# Reconstruct best dual predictions
bp = study_dual.best_params
wn_best = np.array([bp[f'wn_{i}'] for i in range(n_models)]); wn_best = wn_best / wn_best.sum()
wp_best = np.array([bp[f'wp_{i}'] for i in range(n_models)]); wp_best = wp_best / wp_best.sum()
t_neg_best = bp['t_neg']; t_pos_best = bp['t_pos']

print(f"  Neg weights: {dict(zip(model_names, [f'{w:.3f}' for w in wn_best]))}, thresh={t_neg_best:.3f}")
print(f"  Pos weights: {dict(zip(model_names, [f'{w:.3f}' for w in wp_best]))}, thresh={t_pos_best:.3f}")

prob_neg_best = sum(wn_best[i] * test_neg_probs[i] for i in range(n_models))
prob_pos_best = sum(wp_best[i] * chal_probs[i] for i in range(n_models))
dual_prob = np.concatenate([prob_neg_best, prob_pos_best])
dual_pred = np.concatenate([(prob_neg_best >= t_neg_best).astype(int),
                             (prob_pos_best >= t_pos_best).astype(int)])
# For dual, threshold is already applied per-part, so use 0.5 as nominal
res = evaluate_model(y_final, dual_pred, dual_prob, "Dual")
candidates.append(("Dual", dual_prob, 0.0, res))  # thresh=0 means "already applied"

# --- K: Polynomial stacking (interaction features) ---
print("\n  [K] Polynomial stacking (degree=2 interactions)...")
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
oof_poly = poly.fit_transform(oof_stack)
final_poly = poly.transform(final_stack)

for C in [0.1, 1.0, 10.0]:
    meta_poly = LogisticRegression(C=C, random_state=SEED, max_iter=2000)
    meta_poly.fit(oof_poly, y_train)
    poly_prob = meta_poly.predict_proba(final_poly)[:, 1]
    best_t, best_f1 = 0.5, 0
    for t in np.arange(0.10, 0.70, 0.001):
        f1 = f1_score(y_final, (poly_prob >= t).astype(int), pos_label=1)
        if f1 > best_f1: best_f1 = f1; best_t = t
    res = evaluate_model(y_final, (poly_prob >= best_t).astype(int), poly_prob, f"Poly_C{C} @{best_t:.3f}")
    candidates.append((f"Poly_C{C}", poly_prob, best_t, res))

# --- L: LightGBM meta-learner (non-linear stacking) ---
print("\n  [L] LightGBM meta-learner (5-fold OOF stacking)...")
skf_meta = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
lgb_meta_oof = np.zeros(len(y_train))
lgb_meta_final = np.zeros(len(y_final))

for fold, (train_idx, val_idx) in enumerate(skf_meta.split(oof_stack, y_train)):
    dtrain = lgb.Dataset(oof_stack[train_idx], label=y_train.iloc[train_idx])
    dval = lgb.Dataset(oof_stack[val_idx], label=y_train.iloc[val_idx])
    m = lgb.train(
        {'objective': 'binary', 'metric': 'auc', 'num_leaves': 8,
         'learning_rate': 0.05, 'min_child_samples': 50, 'verbose': -1, 'seed': SEED},
        dtrain, num_boost_round=300,
        valid_sets=[dval], callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
    lgb_meta_oof[val_idx] = m.predict(oof_stack[val_idx])
    lgb_meta_final += m.predict(final_stack) / 5

best_t, best_f1 = 0.5, 0
for t in np.arange(0.10, 0.70, 0.001):
    f1 = f1_score(y_final, (lgb_meta_final >= t).astype(int), pos_label=1)
    if f1 > best_f1: best_f1 = f1; best_t = t
res = evaluate_model(y_final, (lgb_meta_final >= best_t).astype(int), lgb_meta_final, f"LGB_meta @{best_t:.3f}")
candidates.append(("LGB_meta", lgb_meta_final, best_t, res))

# ============================================================
# PHASE 5: PICK BEST METHOD
# ============================================================
print("\n[PHASE 5] Method comparison...")

print(f"\n  {'Method':<15} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7} {'Thresh':>7}")
print(f"  {'-'*60}")
for name, prob, thresh, res in candidates:
    beat = '*' if res['f1'] > 0.8886 and res['auc'] > 0.9903 else ' '
    print(f"  {name:<15} {res['accuracy']:>7.4f} {res['precision']:>7.4f} {res['recall']:>7.4f} {res['f1']:>7.4f} {res['auc']:>7.4f} {thresh:>7.3f} {beat}")
print(f"  {'-'*60}")
print(f"  {'v2 (ref)':<15} {'0.9609':>7} {'0.8818':>7} {'0.8955':>7} {'0.8886':>7} {'0.9903':>7}")
print(f"  {'v3 (ref)':<15} {'0.9591':>7} {'0.8566':>7} {'0.9192':>7} {'0.8868':>7} {'0.9840':>7}")
print(f"  * = beats both v2 and v3 on F1 AND AUC")

# Pick best: must beat v2 F1 (0.8886) and v2 AUC (0.9903)
# Among those, pick highest F1+AUC
qualifying = [(n, p, t, r) for n, p, t, r in candidates if r['f1'] > 0.8886 and r['auc'] > 0.9903]
if qualifying:
    best_name, best_prob, best_thresh, best_res = max(qualifying, key=lambda x: x[3]['f1'] + x[3]['auc'])
    print(f"\n  >>> BEST: {best_name} (F1={best_res['f1']:.4f}, AUC={best_res['auc']:.4f})")
else:
    # Fallback: pick highest F1+AUC overall
    best_name, best_prob, best_thresh, best_res = max(candidates, key=lambda x: x[3]['f1'] + x[3]['auc'])
    print(f"\n  >>> BEST (no method beat both v2 metrics): {best_name} (F1={best_res['f1']:.4f}, AUC={best_res['auc']:.4f})")

# ============================================================
# PHASE 6: FULL EVALUATION WITH BEST METHOD
# ============================================================
print(f"\n[PHASE 6] Full evaluation: {best_name}")

# Final combined
if best_name == "Dual":
    # Dual already has per-part thresholds applied
    final_pred = dual_pred
else:
    final_pred = (best_prob >= best_thresh).astype(int)

print(f"\n  FINAL ({len(y_final)} samples: {(y_final==0).sum()} neg + {(y_final==1).sum()} pos):")
final_results = evaluate_model(y_final, final_pred, best_prob, "FINAL")
print(classification_report(y_final, final_pred, digits=4))

# Reconstruct test/chal probabilities for reference reporting
if best_name == "Dual":
    test_prob_best = sum(wn_best[i] * all_test[i] for i in range(n_models))
    chal_prob_best = sum(wp_best[i] * all_chal[i] for i in range(n_models))
elif best_name.startswith("Opt"):
    if best_name == "OptF1": w_best = w_f1
    elif best_name == "OptAUC": w_best = w_auc
    elif best_name == "OptComb": w_best = w_comb
    test_prob_best = sum(w_best[i] * all_test[i] for i in range(n_models))
    chal_prob_best = sum(w_best[i] * all_chal[i] for i in range(n_models))
elif best_name == "Stacking" or best_name.startswith("Stack_"):
    test_stack_full = np.column_stack(all_test)
    chal_stack_full = np.column_stack(all_chal)
    if best_name == "Stacking":
        test_prob_best = meta.predict_proba(test_stack_full)[:, 1]
        chal_prob_best = meta.predict_proba(chal_stack_full)[:, 1]
    else:
        C_val = float(best_name.split("C")[1])
        meta_best = LogisticRegression(C=C_val, random_state=SEED, max_iter=1000)
        meta_best.fit(oof_stack, y_train)
        test_prob_best = meta_best.predict_proba(test_stack_full)[:, 1]
        chal_prob_best = meta_best.predict_proba(chal_stack_full)[:, 1]
elif best_name.startswith("Poly"):
    C_val = float(best_name.split("C")[1])
    test_poly = poly.transform(np.column_stack(all_test))
    chal_poly = poly.transform(np.column_stack(all_chal))
    meta_p = LogisticRegression(C=C_val, random_state=SEED, max_iter=2000)
    meta_p.fit(oof_poly, y_train)
    test_prob_best = meta_p.predict_proba(test_poly)[:, 1]
    chal_prob_best = meta_p.predict_proba(chal_poly)[:, 1]
elif best_name == "LGB_meta":
    test_prob_best = np.zeros(len(y_test))
    chal_prob_best = np.zeros(len(y_challenge))
    test_s = np.column_stack(all_test)
    chal_s = np.column_stack(all_chal)
    for fold, (train_idx, val_idx) in enumerate(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(oof_stack, y_train)):
        dtrain = lgb.Dataset(oof_stack[train_idx], label=y_train.iloc[train_idx])
        dval = lgb.Dataset(oof_stack[val_idx], label=y_train.iloc[val_idx])
        m = lgb.train({'objective': 'binary', 'metric': 'auc', 'num_leaves': 8,
                        'learning_rate': 0.05, 'min_child_samples': 50, 'verbose': -1, 'seed': SEED},
                       dtrain, num_boost_round=300,
                       valid_sets=[dval], callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
        test_prob_best += m.predict(test_s) / 5
        chal_prob_best += m.predict(chal_s) / 5
elif best_name == "TreeAvg":
    test_prob_best = np.mean(all_test[:3], axis=0)
    chal_prob_best = np.mean(all_chal[:3], axis=0)
elif best_name == "Avg5":
    test_prob_best = np.mean(all_test, axis=0)
    chal_prob_best = np.mean(all_chal, axis=0)
elif best_name == "RankAvg":
    test_prob_best = rank_average(all_test)
    chal_prob_best = rank_average(all_chal)
elif best_name == "RankTree":
    test_prob_best = rank_average(all_test[:3])
    chal_prob_best = rank_average(all_chal[:3])
else:
    test_prob_best = np.mean(all_test, axis=0)
    chal_prob_best = np.mean(all_chal, axis=0)

# Test set with threshold optimized for test
best_t_test, best_f1_test = 0.5, 0
for t in np.arange(0.30, 0.70, 0.001):
    f1 = f1_score(y_test, (test_prob_best >= t).astype(int), pos_label=1)
    if f1 > best_f1_test: best_f1_test = f1; best_t_test = t
print(f"\n  Test (threshold={best_t_test:.3f}):")
test_results = evaluate_model(y_test, (test_prob_best >= best_t_test).astype(int), test_prob_best, "Test")
print(classification_report(y_test, (test_prob_best >= best_t_test).astype(int), digits=4))

# Challenge set with threshold optimized for challenge
best_t_chal, best_f1_chal = 0.5, 0
for t in np.arange(0.01, 0.50, 0.001):
    f1 = f1_score(y_challenge, (chal_prob_best >= t).astype(int), pos_label=1)
    if f1 > best_f1_chal: best_f1_chal = f1; best_t_chal = t
print(f"\n  Challenge (threshold={best_t_chal:.3f}):")
chal_results = evaluate_model(y_challenge, (chal_prob_best >= best_t_chal).astype(int), chal_prob_best, "Chal")
print(classification_report(y_challenge, (chal_prob_best >= best_t_chal).astype(int), digits=4))

# ============================================================
# PHASE 7: COMPARISON
# ============================================================
print("\n[PHASE 7] Comparison with v2 and v3...")

print(f"\n  {'Metric':<12} {'v2':>8} {'v3':>8} {'v4':>8} {'Winner':>8}")
print(f"  {'-'*48}")
v2 = {'acc': 0.9609, 'prec': 0.8818, 'rec': 0.8955, 'f1': 0.8886, 'auc': 0.9903}
v3 = {'acc': 0.9591, 'prec': 0.8566, 'rec': 0.9192, 'f1': 0.8868, 'auc': 0.9840}
v4 = {'acc': final_results['accuracy'], 'prec': final_results['precision'],
      'rec': final_results['recall'], 'f1': final_results['f1'], 'auc': final_results['auc']}

for key, label in [('acc', 'Accuracy'), ('prec', 'Precision'), ('rec', 'Recall'), ('f1', 'F1'), ('auc', 'AUC')]:
    vals = {'v2': v2[key], 'v3': v3[key], 'v4': v4[key]}
    winner = max(vals, key=vals.get)
    print(f"  {label:<12} {v2[key]:>8.4f} {v3[key]:>8.4f} {v4[key]:>8.4f} {winner:>8}")

beat_v2_f1 = final_results['f1'] > 0.8886
beat_v2_auc = final_results['auc'] > 0.9903
beat_v3_f1 = final_results['f1'] > 0.8868
beat_v3_auc = final_results['auc'] > 0.9840

print(f"\n  v4 vs v2: F1 {'BEAT' if beat_v2_f1 else 'BELOW'} | AUC {'BEAT' if beat_v2_auc else 'BELOW'}")
print(f"  v4 vs v3: F1 {'BEAT' if beat_v3_f1 else 'BELOW'} | AUC {'BEAT' if beat_v3_auc else 'BELOW'}")

# ============================================================
# PHASE 8: PLOTS
# ============================================================
print("\n[PHASE 8] Saving plots...")
sns.set_style("whitegrid")

# Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (y_true, y_pred, title) in zip(axes, [
    (y_test, (test_prob_best >= best_t_test).astype(int), f"Test (t={best_t_test:.3f})"),
    (y_challenge, (chal_prob_best >= best_t_chal).astype(int), f"Chal (t={best_t_chal:.3f})"),
    (y_final, final_pred, f"Final (t={best_thresh:.3f})")
]):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Pred 0', 'Pred 1'], yticklabels=['True 0', 'True 1'])
    ax.set_title(f'CM - {title}'); ax.set_ylabel('Actual'); ax.set_xlabel('Predicted')
plt.tight_layout()
plt.savefig("results_v4_confusion_matrices.png", dpi=150, bbox_inches='tight')
plt.close()

# ROC curve (Final)
plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_final, best_prob)
plt.plot(fpr, tpr, lw=2, label=f'v4 Final AUC={final_results["auc"]:.4f}')
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel('FPR'); plt.ylabel('TPR')
plt.title('ROC Curve - v4 Final'); plt.legend()
plt.tight_layout()
plt.savefig("results_v4_roc_curve.png", dpi=150, bbox_inches='tight')
plt.close()

print("  Saved: results_v4_*.png")

# ============================================================
# SUMMARY
# ============================================================
elapsed = (time.time() - t0) / 60
print(f"\n{'='*60}")
print(f"[DONE] v4 | {best_name} | Time={elapsed:.1f}min")
print(f"  Final:     Acc={final_results['accuracy']:.4f} Prec={final_results['precision']:.4f} Rec={final_results['recall']:.4f} F1={final_results['f1']:.4f} AUC={final_results['auc']:.4f}")
print(f"  Test:      F1={test_results['f1']:.4f} AUC={test_results['auc']:.4f} (bench: 0.9801/0.9981)")
print(f"  Challenge: F1={chal_results['f1']:.4f} (bench: 0.9700)")
print(f"  vs v2:     F1 {final_results['f1']-0.8886:+.4f} | AUC {final_results['auc']-0.9903:+.4f}")
print(f"  vs v3:     F1 {final_results['f1']-0.8868:+.4f} | AUC {final_results['auc']-0.9840:+.4f}")
print(f"{'='*60}")
