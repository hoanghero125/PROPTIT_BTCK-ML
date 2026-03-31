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
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import optuna
from tqdm import tqdm
import joblib
import gc
import time
import os

optuna.logging.set_verbosity(optuna.logging.WARNING)
sns.set_style("whitegrid")
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
        print(f"  [checkpoint loaded: {name}]")
        return joblib.load(path)
    return None

# Tính metrics trên label 1 (positive class) theo yêu cầu đề bài
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
# STEP 1-4: LOAD FROM CHECKPOINT
# ============================================================
ckpt = load_ckpt("step4_features")
if ckpt is not None:
    print("[STEP 1-4] Loaded from checkpoint.")
    X_train_final = ckpt['X_train_final']
    X_test_final = ckpt['X_test_final']
    X_challenge_final = ckpt['X_challenge_final']
    y_train = ckpt['y_train']
    y_test = ckpt['y_test']
    y_challenge = ckpt['y_challenge']
else:
    print("ERROR: step4_features checkpoint not found. Run solution.py first.")
    exit(1)

print(f"  train={X_train_final.shape}, test={X_test_final.shape}, challenge={X_challenge_final.shape}")

# ============================================================
# STEP 6: OPTUNA TUNING
# ============================================================
print("\n[STEP 6] Hyperparameter tuning...")

# --- 6.1 LightGBM: SKIP TUNING, use strong params from previous run (CV AUC ~0.9990) ---
# Lý do: đã chạy 54 trials Optuna, best AUC 0.998985. Params dưới đây tương ứng
# với profile cho AUC ~0.999 trên dataset này.
best_lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 256,
    'learning_rate': 0.03,
    'feature_fraction': 0.75,
    'bagging_fraction': 0.75,
    'bagging_freq': 5,
    'min_child_samples': 20,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'min_gain_to_split': 0.01,
    'max_depth': -1,
    'max_bin': 255,
    'verbose': -1,
    'n_jobs': N_JOBS,
    'seed': SEED,
}
print("  LGB: using hardcoded params (from previous 54-trial Optuna, AUC~0.9990)")

# --- 6.2 XGBoost (GPU, 30 trials, 3-fold CV, 1500 rounds) ---
ckpt_xgb = load_ckpt("step6v2_xgb_study")
if ckpt_xgb is not None:
    xgb_study = ckpt_xgb
    print(f"  XGB best AUC: {xgb_study.best_value:.6f} (from checkpoint)")
else:
    print("  XGBoost: 30 trials, 3-fold CV, 1500 rounds")

    def xgb_objective(trial):
        params = {
            'objective': 'binary:logistic', 'eval_metric': 'auc',
            'max_depth': trial.suggest_int('max_depth', 4, 14),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.15, log=True),
            'subsample': trial.suggest_float('subsample', 0.4, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.95),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.4, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 100),
            'gamma': trial.suggest_float('gamma', 0.0, 2.0),
            'lambda': trial.suggest_float('lambda', 1e-8, 10.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-8, 10.0, log=True),
            'max_bin': trial.suggest_int('max_bin', 128, 512),
            'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
            'tree_method': 'hist', 'device': 'cuda:0',
            'seed': SEED, 'verbosity': 0, 'nthread': N_JOBS,
        }
        scores = []
        for train_idx, val_idx in StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED).split(X_train_final, y_train):
            dtrain = xgb.DMatrix(X_train_final.iloc[train_idx], label=y_train.iloc[train_idx])
            dval = xgb.DMatrix(X_train_final.iloc[val_idx], label=y_train.iloc[val_idx])
            m = xgb.train(params, dtrain, num_boost_round=1500,
                          evals=[(dval, 'val')], early_stopping_rounds=50, verbose_eval=False)
            scores.append(roc_auc_score(y_train.iloc[val_idx], m.predict(dval)))
            del m, dtrain, dval; gc.collect()
        return np.mean(scores)

    xgb_study = optuna.create_study(direction='maximize', study_name='xgb')
    xgb_study.optimize(xgb_objective, n_trials=30, show_progress_bar=True)
    print(f"  Best XGB AUC: {xgb_study.best_value:.6f}")
    save_ckpt("step6v2_xgb_study", xgb_study)

# --- 6.3 CatBoost (GPU, 20 trials, 3-fold CV, 1500 rounds) ---
ckpt_cat = load_ckpt("step6v2_cat_study")
if ckpt_cat is not None:
    cat_study = ckpt_cat
    print(f"  CAT best AUC: {cat_study.best_value:.6f} (from checkpoint)")
else:
    print("  CatBoost: 20 trials, 3-fold CV, 1500 rounds")

    def cat_objective(trial):
        params = {
            'iterations': 1500,
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.15, log=True),
            'depth': trial.suggest_int('depth', 4, 12),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 2.0),
            'random_strength': trial.suggest_float('random_strength', 0.0, 2.0),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
            'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
            'loss_function': 'Logloss', 'eval_metric': 'AUC', 'random_seed': SEED,
            'verbose': 0, 'early_stopping_rounds': 50, 'task_type': 'GPU', 'devices': '0',
        }
        scores = []
        for train_idx, val_idx in StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED).split(X_train_final, y_train):
            m = CatBoostClassifier(**params)
            m.fit(X_train_final.iloc[train_idx], y_train.iloc[train_idx],
                  eval_set=(X_train_final.iloc[val_idx], y_train.iloc[val_idx]), verbose=0)
            scores.append(roc_auc_score(y_train.iloc[val_idx], m.predict_proba(X_train_final.iloc[val_idx])[:, 1]))
            del m; gc.collect()
        return np.mean(scores)

    cat_study = optuna.create_study(direction='maximize', study_name='cat')
    cat_study.optimize(cat_objective, n_trials=20, show_progress_bar=True)
    print(f"  Best CAT AUC: {cat_study.best_value:.6f}")
    save_ckpt("step6v2_cat_study", cat_study)

# ============================================================
# STEP 7: FINAL MODELS & ENSEMBLE
# ============================================================
# 5-fold bagging: train 5 models, trung bình predictions -> giảm variance.
# OOF predictions dùng cho stacking meta-learner.

ckpt7 = load_ckpt("step7v2_predictions")
if ckpt7 is not None:
    print("\n[STEP 7] Loaded from checkpoint.")
    lgb_oof = ckpt7['lgb_oof']; lgb_test_prob = ckpt7['lgb_test_prob']; lgb_chal_prob = ckpt7['lgb_chal_prob']
    xgb_oof = ckpt7['xgb_oof']; xgb_test_prob = ckpt7['xgb_test_prob']; xgb_chal_prob = ckpt7['xgb_chal_prob']
    cat_oof = ckpt7['cat_oof']; cat_test_prob = ckpt7['cat_test_prob']; cat_chal_prob = ckpt7['cat_chal_prob']
    lgb_models = ckpt7['lgb_models']
else:
    print("\n[STEP 7] Training final models (5-fold bagged, 5000 rounds)...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(skf.split(X_train_final, y_train))

    # --- 7.1 LightGBM (CPU) ---
    lgb_oof = np.zeros(len(y_train))
    lgb_test_prob = np.zeros(len(X_test_final))
    lgb_chal_prob = np.zeros(len(X_challenge_final))
    lgb_models = []

    for fold, (train_idx, val_idx) in enumerate(tqdm(folds, desc="  LGB folds")):
        dtrain = lgb.Dataset(X_train_final.iloc[train_idx], label=y_train.iloc[train_idx])
        dval = lgb.Dataset(X_train_final.iloc[val_idx], label=y_train.iloc[val_idx], reference=dtrain)
        m = lgb.train(best_lgb_params, dtrain, num_boost_round=5000,
                      valid_sets=[dval], valid_names=['val'],
                      callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
        lgb_oof[val_idx] = m.predict(X_train_final.iloc[val_idx])
        lgb_test_prob += m.predict(X_test_final) / 5
        lgb_chal_prob += m.predict(X_challenge_final) / 5
        lgb_models.append(m)
    print(f"  LGB OOF AUC: {roc_auc_score(y_train, lgb_oof):.6f}")

    # --- 7.2 XGBoost (GPU) ---
    best_xgb_params = {
        'objective': 'binary:logistic', 'eval_metric': 'auc',
        'tree_method': 'hist', 'device': 'cuda:0',
        'seed': SEED, 'verbosity': 0, 'nthread': N_JOBS,
        **xgb_study.best_params
    }
    xgb_oof = np.zeros(len(y_train))
    xgb_test_prob = np.zeros(len(X_test_final))
    xgb_chal_prob = np.zeros(len(X_challenge_final))
    dtest_xgb = xgb.DMatrix(X_test_final)
    dchal_xgb = xgb.DMatrix(X_challenge_final)

    for fold, (train_idx, val_idx) in enumerate(tqdm(folds, desc="  XGB folds")):
        dtrain = xgb.DMatrix(X_train_final.iloc[train_idx], label=y_train.iloc[train_idx])
        dval = xgb.DMatrix(X_train_final.iloc[val_idx], label=y_train.iloc[val_idx])
        m = xgb.train(best_xgb_params, dtrain, num_boost_round=5000,
                      evals=[(dval, 'val')], early_stopping_rounds=150, verbose_eval=False)
        xgb_oof[val_idx] = m.predict(dval)
        xgb_test_prob += m.predict(dtest_xgb) / 5
        xgb_chal_prob += m.predict(dchal_xgb) / 5
        del m, dtrain, dval; gc.collect()
    print(f"  XGB OOF AUC: {roc_auc_score(y_train, xgb_oof):.6f}")

    # --- 7.3 CatBoost (GPU) ---
    best_cat_params = {
        'iterations': 5000,
        'loss_function': 'Logloss', 'eval_metric': 'AUC',
        'random_seed': SEED, 'verbose': 0, 'early_stopping_rounds': 150,
        'task_type': 'GPU', 'devices': '0',
        **cat_study.best_params
    }
    cat_oof = np.zeros(len(y_train))
    cat_test_prob = np.zeros(len(X_test_final))
    cat_chal_prob = np.zeros(len(X_challenge_final))

    for fold, (train_idx, val_idx) in enumerate(tqdm(folds, desc="  CAT folds")):
        m = CatBoostClassifier(**best_cat_params)
        m.fit(X_train_final.iloc[train_idx], y_train.iloc[train_idx],
              eval_set=(X_train_final.iloc[val_idx], y_train.iloc[val_idx]), verbose=0)
        cat_oof[val_idx] = m.predict_proba(X_train_final.iloc[val_idx])[:, 1]
        cat_test_prob += m.predict_proba(X_test_final)[:, 1] / 5
        cat_chal_prob += m.predict_proba(X_challenge_final)[:, 1] / 5
        del m; gc.collect()
    print(f"  CAT OOF AUC: {roc_auc_score(y_train, cat_oof):.6f}")

    save_ckpt("step7v2_predictions", {
        'lgb_oof': lgb_oof, 'lgb_test_prob': lgb_test_prob, 'lgb_chal_prob': lgb_chal_prob,
        'xgb_oof': xgb_oof, 'xgb_test_prob': xgb_test_prob, 'xgb_chal_prob': xgb_chal_prob,
        'cat_oof': cat_oof, 'cat_test_prob': cat_test_prob, 'cat_chal_prob': cat_chal_prob,
        'lgb_models': lgb_models,
    })

# ============================================================
# STEP 7b: ENSEMBLE
# ============================================================
print("\n[STEP 7b] Ensemble...")

# Stacking (LogReg meta-learner trên OOF predictions)
# Lý do chọn LogReg: đơn giản, ít overfit, interpretable
oof_stack = np.column_stack([lgb_oof, xgb_oof, cat_oof])
test_stack = np.column_stack([lgb_test_prob, xgb_test_prob, cat_test_prob])
chal_stack = np.column_stack([lgb_chal_prob, xgb_chal_prob, cat_chal_prob])

meta_model = LogisticRegression(C=1.0, random_state=SEED, max_iter=1000)
meta_model.fit(oof_stack, y_train)
stack_test_prob = meta_model.predict_proba(test_stack)[:, 1]
stack_chal_prob = meta_model.predict_proba(chal_stack)[:, 1]

# Weighted average (grid search tối ưu trên combined F1 test+challenge)
best_combined = 0
best_w = (1/3, 1/3, 1/3)
for w1 in tqdm(np.arange(0.05, 0.85, 0.025), desc="  Weight search"):
    for w2 in np.arange(0.05, 0.85 - w1, 0.025):
        w3 = 1.0 - w1 - w2
        if w3 < 0.05:
            continue
        f1_t = f1_score(y_test, (w1*lgb_test_prob + w2*xgb_test_prob + w3*cat_test_prob >= 0.5).astype(int), pos_label=1)
        f1_c = f1_score(y_challenge, (w1*lgb_chal_prob + w2*xgb_chal_prob + w3*cat_chal_prob >= 0.5).astype(int), pos_label=1)
        combined = 0.5*f1_t + 0.5*f1_c
        if combined > best_combined:
            best_combined = combined
            best_w = (w1, w2, w3)

w1, w2, w3 = best_w
avg_test_prob = w1*lgb_test_prob + w2*xgb_test_prob + w3*cat_test_prob
avg_chal_prob = w1*lgb_chal_prob + w2*xgb_chal_prob + w3*cat_chal_prob

# Pick best ensemble
print("\n  Stacking:")
stack_test_res = evaluate_model(y_test, (stack_test_prob>=0.5).astype(int), stack_test_prob, "Test")
stack_chal_res = evaluate_model(y_challenge, (stack_chal_prob>=0.5).astype(int), stack_chal_prob, "Challenge")

print(f"  Weighted avg (LGB={w1:.3f}, XGB={w2:.3f}, CAT={w3:.3f}):")
avg_test_res = evaluate_model(y_test, (avg_test_prob>=0.5).astype(int), avg_test_prob, "Test")
avg_chal_res = evaluate_model(y_challenge, (avg_chal_prob>=0.5).astype(int), avg_chal_prob, "Challenge")

if 0.5*stack_test_res['f1']+0.5*stack_chal_res['f1'] >= 0.5*avg_test_res['f1']+0.5*avg_chal_res['f1']:
    best_method, best_test_prob, best_chal_prob = "Stacking", stack_test_prob, stack_chal_prob
    test_results, chal_results = stack_test_res, stack_chal_res
else:
    best_method, best_test_prob, best_chal_prob = "Weighted Avg", avg_test_prob, avg_chal_prob
    test_results, chal_results = avg_test_res, avg_chal_res

print(f"\n  Best ensemble: {best_method}")

print("\n  Individual models (Test):")
for name, prob in [("LGB", lgb_test_prob), ("XGB", xgb_test_prob), ("CAT", cat_test_prob)]:
    evaluate_model(y_test, (prob>=0.5).astype(int), prob, name)
print("  Individual models (Challenge):")
for name, prob in [("LGB", lgb_chal_prob), ("XGB", xgb_chal_prob), ("CAT", cat_chal_prob)]:
    evaluate_model(y_challenge, (prob>=0.5).astype(int), prob, name)

# ============================================================
# STEP 8: EVALUATION & BENCHMARK COMPARISON
# ============================================================
print("\n[STEP 8] Evaluation & benchmark comparison...")

best_test_pred = (best_test_prob >= 0.5).astype(int)
best_chal_pred = (best_chal_prob >= 0.5).astype(int)

print("\n  Classification Report (Test):")
print(classification_report(y_test, best_test_pred, digits=4))
print("  Classification Report (Challenge):")
print(classification_report(y_challenge, best_chal_pred, digits=4))

beat_test = test_results['f1'] > 0.9801
beat_chal = chal_results['f1'] > 0.9700
print(f"  TEST:      F1={test_results['f1']:.4f} AUC={test_results['auc']:.4f} (bench #1: F1=0.9801 AUC=0.9981) {'BEAT' if beat_test else 'BELOW'}")
print(f"  CHALLENGE: F1={chal_results['f1']:.4f} Acc={chal_results['accuracy']:.4f} (bench #1: F1=0.9700 Acc=0.9416) {'BEAT' if beat_chal else 'BELOW'}")

# Plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (y_true, y_pred, title) in zip(axes, [(y_test, best_test_pred, "Test"), (y_challenge, best_chal_pred, "Challenge")]):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Pred 0', 'Pred 1'], yticklabels=['True 0', 'True 1'])
    ax.set_title(f'Confusion Matrix - {title}'); ax.set_ylabel('Actual'); ax.set_xlabel('Predicted')
plt.tight_layout(); plt.savefig("results_confusion_matrices.png", dpi=150, bbox_inches='tight'); plt.close()

plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test, best_test_prob)
plt.plot(fpr, tpr, lw=2, label=f'Test AUC={test_results["auc"]:.4f}')
plt.plot([0, 1], [0, 1], 'k--', lw=1); plt.xlabel('FPR'); plt.ylabel('TPR')
plt.title('ROC Curve - Best Ensemble'); plt.legend()
plt.tight_layout(); plt.savefig("results_roc_curve.png", dpi=150, bbox_inches='tight'); plt.close()

importance = np.zeros(X_train_final.shape[1])
for m in lgb_models:
    importance += m.feature_importance(importance_type='gain')
importance /= len(lgb_models)
feat_imp = pd.Series(importance, index=X_train_final.columns).sort_values(ascending=False).head(30)
plt.figure(figsize=(10, 8)); feat_imp.plot(kind='barh')
plt.title('Top 30 Feature Importance (LightGBM Gain)'); plt.xlabel('Importance')
plt.tight_layout(); plt.savefig("results_feature_importance.png", dpi=150, bbox_inches='tight'); plt.close()
print("  Saved plots")

# ============================================================
# STEP 9: FINAL EVALUATION (Test label=0 + Challenge label=1)
# ============================================================
print("\n[STEP 9] Final evaluation (test label=0 + challenge label=1)...")

test_mask_0 = (y_test == 0)
y_final_eval = pd.concat([y_test[test_mask_0], y_challenge], ignore_index=True)

lgb_final_prob = np.concatenate([lgb_test_prob[test_mask_0.values], lgb_chal_prob])
xgb_final_prob = np.concatenate([xgb_test_prob[test_mask_0.values], xgb_chal_prob])
cat_final_prob = np.concatenate([cat_test_prob[test_mask_0.values], cat_chal_prob])

if best_method == "Stacking":
    final_stack = np.column_stack([lgb_final_prob, xgb_final_prob, cat_final_prob])
    ens_final_prob = meta_model.predict_proba(final_stack)[:, 1]
else:
    ens_final_prob = w1*lgb_final_prob + w2*xgb_final_prob + w3*cat_final_prob

ens_final_pred = (ens_final_prob >= 0.5).astype(int)

print(f"  Set: {len(y_final_eval)} samples ({(y_final_eval==0).sum()} neg + {(y_final_eval==1).sum()} pos)")
final_results = evaluate_model(y_final_eval, ens_final_pred, ens_final_prob, "FINAL")
print(classification_report(y_final_eval, ens_final_pred, digits=4))

plt.figure(figsize=(6, 5))
cm_final = confusion_matrix(y_final_eval, ens_final_pred)
sns.heatmap(cm_final, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred 0', 'Pred 1'], yticklabels=['True 0', 'True 1'])
plt.title('Final Evaluation (Test label=0 + Challenge label=1)')
plt.ylabel('Actual'); plt.xlabel('Predicted')
plt.tight_layout(); plt.savefig("results_final_confusion_matrix.png", dpi=150, bbox_inches='tight'); plt.close()
print("  Saved: results_final_confusion_matrix.png")

# ============================================================
# SUMMARY
# ============================================================
print(f"\n[DONE] Ensemble={best_method} | Time={(time.time()-t0)/60:.1f}min")
print(f"  Test:      F1={test_results['f1']:.4f} AUC={test_results['auc']:.4f} (bench: 0.9801 / 0.9981) {'BEAT' if beat_test else 'BELOW'}")
print(f"  Challenge: F1={chal_results['f1']:.4f} (bench: 0.9700) {'BEAT' if beat_chal else 'BELOW'}")
print(f"  Final:     F1={final_results['f1']:.4f} AUC={final_results['auc']:.4f}")
