# Binary Classification Challenge — Bài Tập Cuối Khóa ML



## Tổng quan

Bạn được cung cấp một bộ dữ liệu dạng bảng để giải quyết bài toán **phân loại nhị phân (binary classification)**. Dữ liệu được ẩn danh hoàn toàn — không có tên cột, không có ngữ cảnh bài toán. Nhiệm vụ của bạn là xây dựng một pipeline học máy hoàn chỉnh và đạt kết quả vượt qua toàn bộ bảng benchmark đã cho.

Benchmark được tạo ra bằng cách huấn luyện các model với cấu hình mặc định, **không có bất kỳ bước xử lý dữ liệu nào**. Đây là điểm mấu chốt để bạn có thể vượt qua.

---

## Dữ liệu

**Link tải:** [Google Drive](https://drive.google.com/drive/folders/1_NeHZ80RCRh0qon8EDboXSojE0Z8gnLk?usp=sharing)

Dữ liệu được chia thành ba tập độc lập:

| File | Mô tả |
|------|-------|
| `X_train.csv` | Features tập huấn luyện |
| `y_train.csv` | Nhãn tập huấn luyện (0 hoặc 1) |
| `X_test.csv` | Features tập kiểm tra |
| `y_test.csv` | Nhãn tập kiểm tra |
| `X_challenge.csv` | Features tập thử thách |
| `y_challenge.csv` | Nhãn tập thử thách |

**Đặc điểm dữ liệu:**

- Dạng bảng, tất cả features là số (numerical)
- Các cột không có tên — đây là chủ ý, và cũng là một phần của thử thách
- Nhãn nhị phân: `0` hoặc `1`

> Chỉ được sử dụng `X_train` và `y_train` để huấn luyện. Các tập còn lại chỉ dùng để đánh giá.

---

## Nhiệm vụ

1. Thực hiện phân tích và khám phá dữ liệu (EDA)
2. Xây dựng pipeline tiền xử lý và feature engineering
3. Huấn luyện một hoặc nhiều mô hình phân loại nhị phân
4. Đánh giá mô hình trên cả tập `test` và `challenge`
5. Vượt qua **toàn bộ** bảng benchmark test và challenge.

Không có giới hạn về phương pháp, thư viện, hay loại mô hình được sử dụng.

---

## Bảng Benchmark


**Note:** Benchmark được đánh giá trực tiếp trên mô hình, không có bước tiền xử lý data nào.

### Tập Test

| Rank | Model | Accuracy | Precision | Recall | F1 | AUC |
|:----:|-------|:--------:|:---------:|:------:|:--:|:---:|
| 1 | LightGBM | 0.9800 | 0.9758 | 0.9845 | **0.9801** | **0.9981** |
| 2 | XGBoost | 0.9792 | 0.9752 | 0.9835 | **0.9793** | **0.9980** |
| 3 | HistGradBoost | 0.9743 | 0.9684 | 0.9806 | **0.9745** | **0.9973** |
| 4 | BaggingClassifier | 0.9723 | 0.9669 | 0.9781 | **0.9725** | **0.9952** |
| 5 | RandomForest (500) | 0.9676 | 0.9606 | 0.9752 | **0.9678** | **0.9953** |
| 6 | MLP (256-128) | 0.9665 | 0.9695 | 0.9634 | **0.9664** | **0.9924** |
| 7 | ExtraTrees (500) | 0.9641 | 0.9587 | 0.9700 | **0.9643** | **0.9949** |
| 8 | KNN (k=5) | 0.9550 | 0.9489 | 0.9618 | **0.9553** | **0.9800** |
| 9 | DecisionTree (depth=10) | 0.9547 | 0.9521 | 0.9577 | **0.9549** | **0.9799** |
| 10 | LinearSVC | 0.9470 | 0.9387 | 0.9566 | **0.9476** | **0.9866** |
| 11 | DecisionTree (deep) | 0.9435 | 0.9372 | 0.9508 | **0.9439** | **0.9504** |
| 12 | SGD (hinge/SVM) | 0.9420 | 0.9433 | 0.9405 | **0.9419** | **0.9839** |
| 13 | LogisticRegression | 0.9406 | 0.9374 | 0.9443 | **0.9409** | **0.9843** |
| 14 | LogisticReg (L1) | 0.9380 | 0.9365 | 0.9398 | **0.9381** | **0.9834** |
| 15 | SGD (log) | 0.9332 | 0.9240 | 0.9441 | **0.9339** | **0.9817** |
| 16 | AdaBoost | 0.9275 | 0.9257 | 0.9299 | **0.9278** | **0.9826** |
| 17 | BernoulliNB | 0.7021 | 0.7117 | 0.6802 | **0.6956** | **0.7501** |

### Tập Challenge

| Rank | Model | Accuracy | Precision | Recall | F1 |
|:----:|-------|:--------:|:---------:|:------:|:--:|
| 1 | SGD (log) | 0.9416 | 1.0000 | 0.9416 | **0.9700** |
| 2 | ExtraTrees (500) | 0.9389 | 1.0000 | 0.9389 | **0.9684** |
| 3 | LogisticReg (L1) | 0.9311 | 1.0000 | 0.9311 | **0.9643** |
| 4 | LogisticRegression | 0.8990 | 1.0000 | 0.8990 | **0.9468** |
| 5 | AdaBoost | 0.8922 | 1.0000 | 0.8922 | **0.9430** |
| 6 | MLP (256-128) | 0.8956 | 1.0000 | 0.8956 | **0.9450** |
| 7 | SGD (hinge/SVM) | 0.8912 | 1.0000 | 0.8912 | **0.9425** |
| 8 | HistGradBoost | 0.8874 | 1.0000 | 0.8874 | **0.9403** |
| 9 | LightGBM | 0.8787 | 1.0000 | 0.8787 | **0.9354** |
| 10 | XGBoost | 0.8759 | 1.0000 | 0.8759 | **0.9338** |
| 11 | BaggingClassifier | 0.8564 | 1.0000 | 0.8564 | **0.9226** |
| 12 | LinearSVC | 0.8489 | 1.0000 | 0.8489 | **0.9183** |
| 13 | KNN (k=5) | 0.8429 | 1.0000 | 0.8429 | **0.9148** |
| 14 | BernoulliNB | 0.8133 | 1.0000 | 0.8133 | **0.8970** |
| 15 | DecisionTree (depth=10) | 0.8188 | 1.0000 | 0.8188 | **0.9004** |
| 16 | DecisionTree (deep) | 0.6651 | 1.0000 | 0.6651 | **0.7989** |

**Note:** Chỉ số Precision, Recall, F1 được tính ở trên nhãn 1.

---

## Hướng tiếp cận gợi ý

Vì benchmark không thực hiện xử lý dữ liệu, dư địa cải thiện chủ yếu đến từ:

- **EDA:** Phân tích phân phối, tương quan, phát hiện outlier và missing values
- **Feature engineering:** Tạo features mới, chọn lọc features có tương quan cao với nhãn
- **Tiền xử lý:** Scaling, imputation, xử lý mất cân bằng nhãn (class imbalance)
- **Tối ưu hyperparameter:** Cross-validation, grid search hoặc Bayesian optimization
- **Ensemble:** Stacking, blending, voting classifier

---

## Yêu cầu nộp bài

Mỗi bạn nộp đủ ba thành phần sau:

---

### 1. Code

Nộp một file Jupyter Notebook (`.ipynb`) hoặc script Python (`.py`) có cấu trúc rõ ràng, theo đúng thứ tự pipeline dưới đây:

| Bước | Nội dung bắt buộc |
|------|-------------------|
| 1. Load dữ liệu | Đọc toàn bộ 6 file, kiểm tra shape, dtype, missing values |
| 2. EDA | Phân phối nhãn, phân phối từng feature, ma trận tương quan, phát hiện outlier |
| 3. Tiền xử lý | Scaling, imputation, xử lý class imbalance (nếu có) — ghi rõ lý do chọn từng phương pháp |
| 4. Feature Engineering | Tạo hoặc loại bỏ features, ghi rõ căn cứ quyết định |
| 5. Huấn luyện | Tối thiểu một model chính, có cross-validation |
| 6. Tối ưu | Hyperparameter tuning với grid search, random search hoặc Bayesian optimization |
| 7. Đánh giá | In đầy đủ Accuracy, Precision, Recall, F1, AUC trên cả `test` và `challenge` |

**Yêu cầu kỹ thuật:**

- Đặt seed cố định (`random_state=42`) để đảm bảo kết quả có thể tái hiện
- Không được hard-code kết quả — chạy lại notebook phải cho ra cùng output
- Comment giải thích các đoạn code quan trọng, không comment hiển nhiên
- Không để lại cell lỗi hoặc output rác trong notebook khi nộp

---

### 2. Slide trình bày

Chuẩn bị slide (PowerPoint, Google Slides, hoặc PDF) để trình bày trước nhóm, tối thiểu **12 slide**, theo cấu trúc sau:

| Slide | Nội dung |
|-------|----------|
| 1 | Tiêu đề, tên thành viên |
| 2 | Tổng quan bài toán và hướng tiếp cận ban đầu |
| 3–4 | Kết quả EDA: các phát hiện quan trọng từ dữ liệu |
| 5–6 | Pipeline tiền xử lý và feature engineering: các bước đã làm và lý do |
| 7–8 | Mô hình đã thử nghiệm: so sánh kết quả, lý do chọn model cuối |
| 9 | Kết quả cuối trên tập test và challenge, so sánh với benchmark |
| 10 | Phân tích: tại sao pipeline của nhóm vượt được benchmark? |
| 11 | Những gì đã thử nhưng không hiệu quả và rút ra bài học gì |
| 12 | Hướng cải thiện nếu có thêm thời gian |

Slide cần có biểu đồ, bảng số liệu cụ thể — không chỉ là văn bản mô tả.

---

### 3. Trình bày pipeline (thuyết trình trực tiếp)

Mỗi bạn có **10–15 phút** trình bày và **5 phút** hỏi đáp. 

**Cấu trúc trình bày gợi ý:**

```
[2 phút]  Bài toán và dữ liệu — nhóm hiểu dữ liệu như thế nào?
[4 phút]  Pipeline — từng bước làm gì, tại sao?
[3 phút]  Kết quả — số liệu cụ thể, so sánh với benchmark
[3 phút]  Bài học — điều gì hiệu quả, điều gì không, nếu làm lại sẽ thay đổi gì?
[3 phút]  Demo live (nếu có) hoặc walk-through notebook
```

**Tiêu chí đánh giá phần trình bày:**

| Tiêu chí | Mô tả |
|----------|-------|
| Hiểu bài toán | Nắm được ý tưởng không? |
| Lý luận pipeline | Mỗi quyết định kỹ thuật có được giải thích rõ ràng không? |
| Kết quả | Có vượt benchmark không? Kết quả có tái hiện được không? |
| Phân tích thất bại | Bạn có thử nhiều hướng và rút ra được bài học không? |
| Trả lời câu hỏi | Thành viên có nắm được code và lý thuyết đằng sau không? |

---

## Tham khảo nhanh

```python
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score

# Load dữ liệu
X_train     = pd.read_csv("X_train.csv")
y_train     = pd.read_csv("y_train.csv").squeeze()
X_test      = pd.read_csv("X_test.csv")
y_test      = pd.read_csv("y_test.csv").squeeze()
X_challenge = pd.read_csv("X_challenge.csv")
y_challenge = pd.read_csv("y_challenge.csv").squeeze()

# Đánh giá mô hình
def evaluate(model, X, y, label=""):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    print(f"--- {label} ---")
    print(classification_report(y, y_pred, digits=4))
    print(f"AUC: {roc_auc_score(y, y_prob):.4f}\n")

evaluate(model, X_test, y_test, label="Test")
evaluate(model, X_challenge, y_challenge, label="Challenge")
```
