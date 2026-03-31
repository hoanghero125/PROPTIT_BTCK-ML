import pandas as pd

# Đọc bộ train
X_train = pd.read_csv("train_X.csv")
y_train = pd.read_csv("train_y.csv")

# Đọc bộ test
X_test = pd.read_csv("test_X.csv")
y_test = pd.read_csv("test_y.csv")

# Đọc bộ challenge
X_challenge = pd.read_csv("challenge_X.csv")
y_challenge = pd.read_csv("challenge_y.csv")

import lightgbm as lgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve)
import joblib
import gc

%matplotlib inline
sns.set_style("whitegrid")

print(f"LightGBM version: {lgb.__version__}")

print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)

# Tạo LightGBM Dataset
# free_raw_data=False để giữ lại dữ liệu gốc nếu RAM dư dả, set True nếu thiếu RAM
dtrain = lgb.Dataset(X_train, label=y_train)
dtest = lgb.Dataset(X_test, label=y_test, reference=dtrain)

print("Đã tạo xong LightGBM Datasets.")

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',       # Phân loại nhị phân (0: Benign, 1: Malware)
    'metric': ['auc', 'binary_logloss'], # Đánh giá bằng AUC và Logloss
    'num_leaves': 2048,          # EMBER có nhiều features, cần số lá lớn (thử 1024 hoặc 2048)
    'learning_rate': 0.05,       # Tốc độ học vừa phải
    'feature_fraction': 0.8,     # Chọn ngẫu nhiên 80% features mỗi lần (tránh overfitting)
    'bagging_fraction': 0.8,     # Chọn ngẫu nhiên 80% dữ liệu mỗi lần
    'bagging_freq': 5,
    'verbose': -1,
    'n_jobs': -1                 # Sử dụng tất cả các lõi CPU
}

print("Đã thiết lập tham số.")

callbacks = [
    lgb.log_evaluation(period=50),        # In kết quả sau mỗi 50 vòng
    lgb.early_stopping(stopping_rounds=50) # Dừng nếu không cải thiện sau 50 vòng
]

print("Bắt đầu training...")

bst = lgb.train(
    params,
    dtrain,
    num_boost_round=50,        # Số vòng lặp tối đa
    valid_sets=[dtrain, dtest],  # Đánh giá trên cả tập train và test
    valid_names=['train', 'test'],
    callbacks=callbacks
)

print("Training hoàn tất!")

# Dự đoán trên tập Test
y_pred_prob = bst.predict(X_test, num_iteration=bst.best_iteration)
y_pred_binary = (y_pred_prob >= 0.5).astype(int)

# Tính các metrics
acc = accuracy_score(y_test, y_pred_binary)
auc = roc_auc_score(y_test, y_pred_prob)

print(f"=== KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST ===")
print(f"Accuracy: {acc:.4f}")
print(f"ROC AUC:  {auc:.4f}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_binary, target_names=['Benign', 'Malware']))

# Vẽ Confusion Matrix
cm = confusion_matrix(y_test, y_pred_binary)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malware'], yticklabels=['Benign', 'Malware'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Vẽ top 20 features quan trọng nhất
lgb.plot_importance(bst, max_num_features=20, importance_type='gain', figsize=(10, 8), title='Feature Importance (Gain)')
plt.show()

import sklearn.metrics as metrics

print("=== BẮT ĐẦU ĐÁNH GIÁ TRÊN TẬP CHALLENGE ===")

# 1. Dự đoán xác suất (Probability)
# LightGBM trả về xác suất thuộc lớp 1 (Malware)
y_challenge_prob = bst.predict(X_challenge, num_iteration=bst.best_iteration)

# 2. Chuyển sang nhãn nhị phân (0 hoặc 1) với ngưỡng mặc định 0.5
threshold = 0.5
y_challenge_pred = (y_challenge_prob >= threshold).astype(int)

# 3. Tính toán các chỉ số
acc_challenge = accuracy_score(y_challenge, y_challenge_pred)
roc_auc_challenge = roc_auc_score(y_challenge, y_challenge_prob)

print(f"\nĐộ chính xác (Accuracy): {acc_challenge:.4f}")
print(f"ROC AUC Score:           {roc_auc_challenge:.4f}")
print("\n--- Báo cáo chi tiết (Classification Report) ---")
print(classification_report(y_challenge, y_challenge_pred, target_names=['Benign (0)', 'Malware (1)']))

# 4. Vẽ Confusion Matrix và ROC Curve
fig, ax = plt.subplots(1, 2, figsize=(16, 6))

# --- Biểu đồ 1: Confusion Matrix ---
cm = confusion_matrix(y_challenge, y_challenge_pred)
# Tính phần trăm để dễ hình dung
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[0],
            xticklabels=['Dự đoán Benign', 'Dự đoán Malware'],
            yticklabels=['Thực tế Benign', 'Thực tế Malware'])
ax[0].set_title('Confusion Matrix (Số lượng)')
ax[0].set_ylabel('Nhãn thực tế')
ax[0].set_xlabel('Nhãn dự đoán')

# Chú thích thêm về False Negative (Nguy hiểm nhất trong Malware)
fn = cm[1][0]
fp = cm[0][1]
print(f"\n[QUAN TRỌNG] Phân tích lỗi:")
print(f"- False Negatives (Sót Malware): {fn} mẫu (Malware nhưng bị đoán là An toàn) -> Cực kỳ nguy hiểm.")
print(f"- False Positives (Báo động giả): {fp} mẫu (An toàn nhưng bị đoán là Malware) -> Gây phiền toái.")

# --- Biểu đồ 2: ROC Curve ---
fpr, tpr, _ = metrics.roc_curve(y_challenge, y_challenge_prob)
roc_auc = metrics.auc(fpr, tpr) 

ax[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
ax[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax[1].set_xlim([0.0, 1.0])
ax[1].set_ylim([0.0, 1.05])
ax[1].set_xlabel('False Positive Rate')
ax[1].set_ylabel('True Positive Rate')
ax[1].set_title('Receiver Operating Characteristic (ROC)')
ax[1].legend(loc="lower right")

plt.tight_layout()
plt.show()
