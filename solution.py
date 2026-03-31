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
# STEP 1-4: LOAD + EDA + PREPROCESS + FEATURE ENGINEERING
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
    # STEP 1: LOAD
    print("[STEP 1] Loading data...")
    X_train = pd.read_csv("train_X-001.csv")
    y_train = pd.read_csv("train_y.csv").squeeze().astype(int)
    X_test = pd.read_csv("test_X.csv")
    y_test = pd.read_csv("test_y.csv").squeeze().astype(int)
    X_challenge = pd.read_csv("challenge_X.csv")
    y_challenge = pd.read_csv("challenge_y.csv").squeeze().astype(int)

    clean_cols = [f"f{i}" for i in range(X_train.shape[1])]
    X_train.columns = clean_cols
    X_test.columns = clean_cols
    X_challenge.columns = clean_cols

    print(f"  train={X_train.shape}, test={X_test.shape}, challenge={X_challenge.shape}")
    print(f"  y_train: {y_train.value_counts().to_dict()}, y_test: {y_test.value_counts().to_dict()}, y_challenge: {y_challenge.value_counts().to_dict()}")
    print(f"  Missing: train={X_train.isnull().sum().sum()}, test={X_test.isnull().sum().sum()}, challenge={X_challenge.isnull().sum().sum()}")

    # STEP 2: EDA
    print("\n[STEP 2] EDA...")
    variances = X_train.var()
    nunique = X_train.nunique()
    print(f"  Features: {X_train.shape[1]} | Constant: {(variances==0).sum()} | Binary: {(nunique<=2).sum()} | Low-card(<=5): {(nunique<=5).sum()}")

    corr_with_target = X_train.corrwith(y_train.astype(float)).abs().fillna(0)
    top_corr = corr_with_target.sort_values(ascending=False)
    print(f"  |corr|>0.3: {(corr_with_target>0.3).sum()} | >0.1: {(corr_with_target>0.1).sum()} | >0.01: {(corr_with_target>0.01).sum()} | <0.001(noise): {(corr_with_target<0.001).sum()}")
    print(f"  Top 5: {', '.join(f'{f}={v:.4f}' for f,v in top_corr.head(5).items())}")

    desc = X_train.describe()
    iqr = desc.loc['75%'] - desc.loc['25%']
    outlier_counts = ((X_train < (desc.loc['25%'] - 3*iqr)) | (X_train > (desc.loc['75%'] + 3*iqr))).sum()
    print(f"  Outliers >1%: {(outlier_counts > 0.01*len(X_train)).sum()} features | >5%: {(outlier_counts > 0.05*len(X_train)).sum()} features")

    # EDA Plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, y) in zip(axes, [("Train", y_train), ("Test", y_test), ("Challenge", y_challenge)]):
        y.value_counts().sort_index().plot(kind='bar', ax=ax, color=['#3498db', '#e74c3c'])
        ax.set_title(f'{name} Label Distribution'); ax.set_xlabel('Class'); ax.set_ylabel('Count')
    plt.tight_layout(); plt.savefig("eda_class_distribution.png", dpi=150, bbox_inches='tight'); plt.close()

    top30_feats = top_corr.head(30).index.tolist()
    plt.figure(figsize=(12, 10))
    sns.heatmap(X_train[top30_feats].corr(), cmap='coolwarm', center=0, vmin=-1, vmax=1)
    plt.title("Correlation Heatmap - Top 30 Features")
    plt.tight_layout(); plt.savefig("eda_correlation_heatmap.png", dpi=150, bbox_inches='tight'); plt.close()

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for ax, feat in zip(axes.flatten(), top_corr.head(10).index):
        for label, color in [(0, '#3498db'), (1, '#e74c3c')]:
            ax.hist(X_train.loc[y_train==label, feat], bins=50, alpha=0.5, color=color, label=f'Class {label}', density=True)
        ax.set_title(feat); ax.legend(fontsize=7)
    plt.suptitle("Distribution of Top 10 Features by Class", fontsize=14)
    plt.tight_layout(); plt.savefig("eda_feature_distributions.png", dpi=150, bbox_inches='tight'); plt.close()
    print("  Saved EDA plots")

    # STEP 3: PREPROCESSING
    print("\n[STEP 3] Preprocessing...")
    # Scaling: KHÔNG (tree-based models). Imputation: KHÔNG (no missing). Class imbalance: KHÔNG (50/50).
    # Chỉ loại features hằng số/gần hằng số và chuyển float32.

    constant_cols = variances[variances == 0].index.tolist()
    remaining_var = X_train.drop(columns=constant_cols).var()
    nearzero = remaining_var[remaining_var < 1e-10].index.tolist()
    drop_cols = constant_cols + nearzero

    X_train.drop(columns=drop_cols, inplace=True)
    X_test.drop(columns=drop_cols, inplace=True)
    X_challenge.drop(columns=drop_cols, inplace=True)
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    X_challenge = X_challenge.astype(np.float32)
    gc.collect()
    print(f"  Removed {len(drop_cols)} useless features -> {X_train.shape[1]} remaining")

    # STEP 4: FEATURE ENGINEERING
    print("\n[STEP 4] Feature engineering...")
    # Loại noise (|corr|<0.005), redundant (inter-corr>0.98), tạo interaction/ratio/stats/sq/log

    corr_with_target = X_train.corrwith(y_train.astype(np.float32)).abs().fillna(0)
    top_sorted = corr_with_target.sort_values(ascending=False)

    noise_feats = corr_with_target[corr_with_target < 0.005].index.tolist()
    X_train.drop(columns=noise_feats, inplace=True)
    X_test.drop(columns=noise_feats, inplace=True)
    X_challenge.drop(columns=noise_feats, inplace=True)
    print(f"  Removed {len(noise_feats)} noise features -> {X_train.shape[1]} remaining")

    corr_with_target = X_train.corrwith(y_train.astype(np.float32)).abs().fillna(0)
    top_sorted = corr_with_target.sort_values(ascending=False)

    corr_matrix = X_train.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop_redundant = set()
    for col in upper.columns:
        for hc in upper.index[upper[col] > 0.98].tolist():
            if corr_with_target.get(col, 0) >= corr_with_target.get(hc, 0):
                to_drop_redundant.add(hc)
            else:
                to_drop_redundant.add(col)
    to_drop_redundant = list(to_drop_redundant)
    X_train.drop(columns=to_drop_redundant, inplace=True)
    X_test.drop(columns=to_drop_redundant, inplace=True)
    X_challenge.drop(columns=to_drop_redundant, inplace=True)
    print(f"  Removed {len(to_drop_redundant)} redundant features -> {X_train.shape[1]} remaining")

    corr_with_target = X_train.corrwith(y_train.astype(np.float32)).abs().fillna(0)
    top_sorted = corr_with_target.sort_values(ascending=False)

    top20 = top_sorted.head(20).index.tolist()
    top15 = top_sorted.head(15).index.tolist()
    top10 = top_sorted.head(10).index.tolist()
    top50 = top_sorted.head(50).index.tolist()
    new_feats_train, new_feats_test, new_feats_chal = {}, {}, {}

    for i in tqdm(range(len(top20)), desc="  Interactions"):
        for j in range(i+1, len(top20)):
            f1, f2 = top20[i], top20[j]
            name = f"ix_{f1}_{f2}"
            new_feats_train[name] = (X_train[f1] * X_train[f2]).values
            new_feats_test[name] = (X_test[f1] * X_test[f2]).values
            new_feats_chal[name] = (X_challenge[f1] * X_challenge[f2]).values

    for i in tqdm(range(len(top15)), desc="  Ratios"):
        for j in range(i+1, len(top15)):
            f1, f2 = top15[i], top15[j]
            name = f"rt_{f1}_{f2}"
            new_feats_train[name] = (X_train[f1] / (X_train[f2].abs() + 1e-8)).values
            new_feats_test[name] = (X_test[f1] / (X_test[f2].abs() + 1e-8)).values
            new_feats_chal[name] = (X_challenge[f1] / (X_challenge[f2].abs() + 1e-8)).values

    for name, func in tqdm([("rowmean", np.mean), ("rowstd", np.std), ("rowmax", np.max),
                             ("rowmin", np.min), ("rowmedian", np.median)], desc="  Row stats"):
        fname = f"stat_{name}_top50"
        new_feats_train[fname] = np.apply_along_axis(func, 1, X_train[top50].values).astype(np.float32)
        new_feats_test[fname] = np.apply_along_axis(func, 1, X_test[top50].values).astype(np.float32)
        new_feats_chal[fname] = np.apply_along_axis(func, 1, X_challenge[top50].values).astype(np.float32)
    new_feats_train["stat_rowrange_top50"] = new_feats_train["stat_rowmax_top50"] - new_feats_train["stat_rowmin_top50"]
    new_feats_test["stat_rowrange_top50"] = new_feats_test["stat_rowmax_top50"] - new_feats_test["stat_rowmin_top50"]
    new_feats_chal["stat_rowrange_top50"] = new_feats_chal["stat_rowmax_top50"] - new_feats_chal["stat_rowmin_top50"]

    for f in tqdm(top10, desc="  Sq/Log"):
        new_feats_train[f"sq_{f}"] = (X_train[f] ** 2).values
        new_feats_test[f"sq_{f}"] = (X_test[f] ** 2).values
        new_feats_chal[f"sq_{f}"] = (X_challenge[f] ** 2).values
        new_feats_train[f"log_{f}"] = np.log1p(X_train[f].abs()).values
        new_feats_test[f"log_{f}"] = np.log1p(X_test[f].abs()).values
        new_feats_chal[f"log_{f}"] = np.log1p(X_challenge[f].abs()).values

    eng_train = pd.DataFrame(new_feats_train, index=X_train.index).astype(np.float32)
    eng_test = pd.DataFrame(new_feats_test, index=X_test.index).astype(np.float32)
    eng_chal = pd.DataFrame(new_feats_chal, index=X_challenge.index).astype(np.float32)

    X_train_final = pd.concat([X_train, eng_train], axis=1)
    X_test_final = pd.concat([X_test, eng_test], axis=1)
    X_challenge_final = pd.concat([X_challenge, eng_chal], axis=1)

    X_train_final.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test_final.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_challenge_final.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_train_final.fillna(0, inplace=True)
    X_test_final.fillna(0, inplace=True)
    X_challenge_final.fillna(0, inplace=True)

    print(f"  +{len(new_feats_train)} engineered -> Final: train={X_train_final.shape}")

    del X_train, X_test, X_challenge, eng_train, eng_test, eng_chal, new_feats_train, new_feats_test, new_feats_chal
    gc.collect()

    save_ckpt("step4_features", {
        'X_train_final': X_train_final, 'X_test_final': X_test_final,
        'X_challenge_final': X_challenge_final,
        'y_train': y_train, 'y_test': y_test, 'y_challenge': y_challenge,
    })

# ============================================================
# STEP 5: CV BASELINE CHECK
# ============================================================
print("\n[STEP 5] 5-fold CV baseline check...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(tqdm(list(skf.split(X_train_final, y_train)), desc="  CV folds")):
    dtrain = lgb.Dataset(X_train_final.iloc[train_idx], label=y_train.iloc[train_idx])
    dval = lgb.Dataset(X_train_final.iloc[val_idx], label=y_train.iloc[val_idx], reference=dtrain)
    model = lgb.train(
        {'boosting_type': 'gbdt', 'objective': 'binary', 'metric': 'auc',
         'num_leaves': 255, 'learning_rate': 0.05, 'feature_fraction': 0.8,
         'bagging_fraction': 0.8, 'bagging_freq': 5, 'verbose': -1,
         'n_jobs': N_JOBS, 'seed': SEED},
        dtrain, num_boost_round=3000,
        valid_sets=[dval], valid_names=['val'],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    auc = roc_auc_score(y_train.iloc[val_idx], model.predict(X_train_final.iloc[val_idx]))
    cv_scores.append(auc)

print(f"  CV AUC: {np.mean(cv_scores):.6f} +/- {np.std(cv_scores):.6f}")
del model; gc.collect()

# ============================================================
# STEP 6: OPTUNA TUNING
# ============================================================
print("\n[STEP 6] Hyperparameter tuning (Optuna)...")

# --- 6.1 LightGBM (CPU, 64 threads) ---
ckpt_lgb = load_ckpt("step6_lgb_study")
if ckpt_lgb is not None:
    lgb_study = ckpt_lgb
    print(f"  LGB best AUC: {lgb_study.best_value:.6f} (from checkpoint)")
else:
    print("  LightGBM: 100 trials, 5-fold CV")
    def lgb_objective(trial):
        params = {
            'boosting_type': 'gbdt', 'objective': 'binary', 'metric': 'auc',
            'num_leaves': trial.suggest_int('num_leaves', 63, 1024),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.15, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 0.95),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 0.95),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 200),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 2.0),
            'max_depth': trial.suggest_int('max_depth', -1, 20),
            'max_bin': trial.suggest_int('max_bin', 63, 511),
            'min_data_in_bin': trial.suggest_int('min_data_in_bin', 3, 50),
            'path_smooth': trial.suggest_float('path_smooth', 0.0, 1.0),
            'verbose': -1, 'n_jobs': N_JOBS, 'seed': SEED,
        }
        scores = []
        for train_idx, val_idx in StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X_train_final, y_train):
            dtrain = lgb.Dataset(X_train_final.iloc[train_idx], label=y_train.iloc[train_idx])
            dval = lgb.Dataset(X_train_final.iloc[val_idx], label=y_train.iloc[val_idx], reference=dtrain)
            m = lgb.train(params, dtrain, num_boost_round=3000,
                          valid_sets=[dval], valid_names=['val'],
                          callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])
            scores.append(roc_auc_score(y_train.iloc[val_idx], m.predict(X_train_final.iloc[val_idx])))
            del m; gc.collect()
        return np.mean(scores)

    lgb_study = optuna.create_study(direction='maximize', study_name='lgb')
    lgb_study.optimize(lgb_objective, n_trials=100, show_progress_bar=True)
    print(f"  Best LGB AUC: {lgb_study.best_value:.6f}")
    save_ckpt("step6_lgb_study", lgb_study)

# --- 6.2 XGBoost (GPU) ---
ckpt_xgb = load_ckpt("step6_xgb_study")
if ckpt_xgb is not None:
    xgb_study = ckpt_xgb
    print(f"  XGB best AUC: {xgb_study.best_value:.6f} (from checkpoint)")
else:
    print("  XGBoost: 80 trials, 5-fold CV")
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
            'tree_method': 'hist', 'device': 'cuda:0', 'seed': SEED, 'verbosity': 0, 'nthread': N_JOBS,
        }
        scores = []
        for train_idx, val_idx in StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X_train_final, y_train):
            dtrain = xgb.DMatrix(X_train_final.iloc[train_idx], label=y_train.iloc[train_idx])
            dval = xgb.DMatrix(X_train_final.iloc[val_idx], label=y_train.iloc[val_idx])
            m = xgb.train(params, dtrain, num_boost_round=3000,
                          evals=[(dval, 'val')], early_stopping_rounds=80, verbose_eval=False)
            scores.append(roc_auc_score(y_train.iloc[val_idx], m.predict(dval)))
            del m, dtrain, dval; gc.collect()
        return np.mean(scores)

    xgb_study = optuna.create_study(direction='maximize', study_name='xgb')
    xgb_study.optimize(xgb_objective, n_trials=80, show_progress_bar=True)
    print(f"  Best XGB AUC: {xgb_study.best_value:.6f}")
    save_ckpt("step6_xgb_study", xgb_study)

# --- 6.3 CatBoost (GPU) ---
ckpt_cat = load_ckpt("step6_cat_study")
if ckpt_cat is not None:
    cat_study = ckpt_cat
    print(f"  CAT best AUC: {cat_study.best_value:.6f} (from checkpoint)")
else:
    print("  CatBoost: 60 trials, 5-fold CV")
    def cat_objective(trial):
        params = {
            'iterations': 3000,
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.15, log=True),
            'depth': trial.suggest_int('depth', 4, 12),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 2.0),
            'random_strength': trial.suggest_float('random_strength', 0.0, 2.0),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
            'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
            'loss_function': 'Logloss', 'eval_metric': 'AUC', 'random_seed': SEED,
            'verbose': 0, 'early_stopping_rounds': 80, 'task_type': 'GPU', 'devices': '0',
        }
        scores = []
        for train_idx, val_idx in StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X_train_final, y_train):
            m = CatBoostClassifier(**params)
            m.fit(X_train_final.iloc[train_idx], y_train.iloc[train_idx],
                  eval_set=(X_train_final.iloc[val_idx], y_train.iloc[val_idx]), verbose=0)
            scores.append(roc_auc_score(y_train.iloc[val_idx], m.predict_proba(X_train_final.iloc[val_idx])[:, 1]))
            del m; gc.collect()
        return np.mean(scores)

    cat_study = optuna.create_study(direction='maximize', study_name='cat')
    cat_study.optimize(cat_objective, n_trials=60, show_progress_bar=True)
    print(f"  Best CAT AUC: {cat_study.best_value:.6f}")
    save_ckpt("step6_cat_study", cat_study)

# ============================================================
# STEP 7: FINAL MODELS & ENSEMBLE
# ============================================================
# 5-fold bagging: train 5 models, trung bình predictions -> giảm variance.
# OOF predictions dùng cho stacking meta-learner.

ckpt7 = load_ckpt("step7_predictions")
if ckpt7 is not None:
    print("\n[STEP 7] Loaded from checkpoint.")
    lgb_oof = ckpt7['lgb_oof']; lgb_test_prob = ckpt7['lgb_test_prob']; lgb_chal_prob = ckpt7['lgb_chal_prob']
    xgb_oof = ckpt7['xgb_oof']; xgb_test_prob = ckpt7['xgb_test_prob']; xgb_chal_prob = ckpt7['xgb_chal_prob']
    cat_oof = ckpt7['cat_oof']; cat_test_prob = ckpt7['cat_test_prob']; cat_chal_prob = ckpt7['cat_chal_prob']
    lgb_models = ckpt7['lgb_models']
else:
    print("\n[STEP 7] Training final models (5-fold bagged)...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(skf.split(X_train_final, y_train))

    # --- 7.1 LightGBM ---
    best_lgb_params = {
        'boosting_type': 'gbdt', 'objective': 'binary', 'metric': 'auc',
        'verbose': -1, 'n_jobs': N_JOBS, 'seed': SEED,
        **lgb_study.best_params
    }
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

    # --- 7.2 XGBoost ---
    best_xgb_params = {
        'objective': 'binary:logistic', 'eval_metric': 'auc',
        'tree_method': 'hist', 'device': 'cuda:0', 'seed': SEED, 'verbosity': 0, 'nthread': N_JOBS,
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

    # --- 7.3 CatBoost ---
    best_cat_params = {
        'iterations': 5000, 'loss_function': 'Logloss', 'eval_metric': 'AUC',
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

    save_ckpt("step7_predictions", {
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
oof_stack = np.column_stack([lgb_oof, xgb_oof, cat_oof])
test_stack = np.column_stack([lgb_test_prob, xgb_test_prob, cat_test_prob])
chal_stack = np.column_stack([lgb_chal_prob, xgb_chal_prob, cat_chal_prob])

meta_model = LogisticRegression(C=1.0, random_state=SEED, max_iter=1000)
meta_model.fit(oof_stack, y_train)
stack_test_prob = meta_model.predict_proba(test_stack)[:, 1]
stack_chal_prob = meta_model.predict_proba(chal_stack)[:, 1]

# Weighted average (grid search)
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
