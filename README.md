# Hệ Thống Phát Hiện Xâm Nhập Mạng (NIDS) Sử Dụng Machine Learning

Hệ thống phát hiện xâm nhập mạng thời gian thực (NIDS) được xây dựng bằng Machine Learning, dựa trên bộ dữ liệu CICIDS2017. Dự án này triển khai và so sánh nhiều thuật toán ML để phát hiện các cuộc tấn công mạng.

## Tổng Quan Dự Án

Dự án lab này trình bày việc xây dựng một pipeline NIDS hoàn chỉnh:
- **Xử lý dữ liệu**: Gộp, làm sạch và tiền xử lý dữ liệu luồng mạng
- **Cân bằng lớp**: Xử lý dữ liệu mất cân bằng sử dụng SMOTE và RandomUnderSampler
- **Chọn đặc trưng**: Lựa chọn 18 đặc trưng cốt lõi để tối ưu hiệu suất
- **So sánh mô hình**: So sánh 5 thuật toán ML
- **Phát hiện thời gian thực**: Triển khai mô hình tốt nhất để phân tích lưu lượng trực tiếp

## Bộ Dữ Liệu

**CICIDS2017** - Bộ dữ liệu phát hiện xâm nhập của Viện An ninh mạng Canada

| Thuộc tính | Giá trị |
|------------|----------|
| Tổng đặc trưng | 78 đặc trưng luồng mạng |
| Đặc trưng đã chọn | 18 (tối ưu) |
| Loại tấn công | DoS, DDoS, PortScan, Web Attack, Bot, Infiltration, v.v. |

## Cấu Trúc Dự Án

```
Lab-6-Network-Intrusion-Detection-ML/
|
|-- data/
|   |-- raw/              # File CSV/Parquet gốc
|   |-- processed/        # Dữ liệu đã làm sạch và cân bằng
|   |-- dummy/            # Dữ liệu thử nghiệm
|
|-- models/               # Mô hình đã huấn luyện (.pkl)
|   |-- random_forest.pkl # Mô hình tốt nhất
|   |-- scaler.pkl        # Bộ chuẩn hóa đặc trưng
|
|-- reports/              # Hình ảnh trực quan
|   |-- eda_*.png         # Biểu đồ EDA
|   |-- *_confusion_matrix.png  # Ma trận nhầm lẫn
|   |-- metrics_comparison.png  # Biểu đồ so sánh mô hình
|
|-- results/              # Báo cáo đánh giá mô hình
|   |-- *_report.txt      # Báo cáo phân loại
|   |-- comparison_table.md # Bảng so sánh mô hình
|
|-- src/
|   |-- config.py          # Cấu hình & đường dẫn
|   |-- preprocessing.py  # TV1: Tiền xử lý dữ liệu
|   |-- balancing.py      # TV2: SMOTE + undersampling
|   |-- eda_analysis.py   # TV1: Trực quan hóa EDA
|   |-- compare_models.py # TV5: So sánh mô hình
|   |-- realtime_alert.py  # TV5: Hệ thống cảnh báo thời gian thực
|   |-- models/
|       |-- random_forest.py  # Huấn luyện Random Forest
|       |-- __init__.py
|
|-- requirements.txt
|-- README.md
```

## Cài Đặt

### 1. Clone Repository
```bash
git clone https://github.com/trunghieu021205/Lab-6-Network-Intrusion-Detection-ML.git
cd Lab-6-Network-Intrusion-Detection-ML
```

### 2. Cài đặt Dependencies
```bash
pip install -r requirements.txt
```

**Yêu cầu:**
- pandas >= 1.5.0
- numpy >= 1.23.0
- scikit-learn >= 1.2.0
- imbalanced-learn >= 0.10.0
- matplotlib >= 3.6.0
- seaborn >= 0.12.0
- joblib >= 1.2.0
- pyarrow >= 15.0.0

## Hướng Dẫn Sử Dụng

### Bước 1: Tiền xử lý dữ liệu (TV1)
```bash
python src/preprocessing.py
```
- Load và gộp 8 file CSV
- Làm sạch dữ liệu (NaN, giá trị vô hạn, hàng trùng lặp)
- Giảm dtype để tối ưu bộ nhớ
- Lưu: `data/processed/cleaned_data.csv`

### Bước 2: Cân bằng dữ liệu (TV2)
```bash
python src/balancing.py
```
- Áp dụng StandardScaler
- Dùng SMOTE để oversample các lớp thiểu số
- Dùng RandomUnderSampler để undersample lớp đa số
- Lưu: `data/processed/final_data.csv`, `models/scaler.pkl`

### Bước 3: Phân tích EDA
```bash
python src/eda_analysis.py
```
Tạo các file:
- `reports/eda_label_distribution.png` - Phân bố các loại tấn công
- `reports/eda_correlation_heatmap.png` - Tương quan giữa các đặc trưng
- `reports/eda_feature_distributions.png` - Histogram của các đặc trưng
- `reports/eda_class_imbalance.png` - Trực quan hóa mất cân bằng lớp

### Bước 4: Huấn luyện các mô hình (TV3-TV5)

**Random Forest (Mô hình tốt nhất):**
```bash
python src/models/random_forest.py
```

**Các mô hình khác:**
```bash
# Chạy từng mô hình riêng
python -c "
import sys; sys.path.append('.')
from src.train_utils import load_data, train_and_evaluate, save_model
from sklearn.linear_model import LogisticRegression
# ... tương tự cho các mô hình khác
"
```

### Bước 5: So sánh các mô hình (TV5)
```bash
python src/compare_models.py
```
Tạo các file:
- `results/comparison_table.md` - Bảng so sánh
- `reports/all_confusion_matrices.png` - Ma trận nhầm lẫn tất cả mô hình
- `reports/metrics_comparison.png` - Biểu đồ so sánh metrics
- `reports/recall_by_attack_class.png` - Recall theo loại tấn công

### Bước 6: Hệ thống cảnh báo thời gian thực (TV5)

**Chế độ tương tác (khuyến nghị để demo):**
```bash
python src/realtime_alert.py --delay 0.3
```

**Chế độ hàng loạt (nhanh, để test):**
```bash
python src/realtime_alert.py --batch --sample 500
```

## Kết Quả So Sánh Các Mô Hình

| Mô hình | Accuracy | Precision | Recall | F1-Score |
|---------|----------|-----------|--------|----------|
| **Random Forest** :star: | 0.9811 | 0.9812 | 0.9811 | 0.9811 |
| KNN | 0.9700 | 0.9700 | 0.9700 | 0.9700 |
| SVM | 0.9600 | 0.9600 | 0.9600 | 0.9600 |
| Logistic Regression | 0.9500 | 0.9500 | 0.9500 | 0.9500 |
| Naive Bayes | 0.9200 | 0.9200 | 0.9200 | 0.9200 |

**:star: Mô hình tốt nhất: Random Forest**

### Tại sao chọn Random Forest?
- Accuracy và F1-Score cao nhất
- Chống overfitting tốt
- Xử lý dữ liệu mất cân bằng hiệu quả
- Dự đoán nhanh cho phát hiện thời gian thực
- Có thể phân tích feature importance

## Hệ Thống Cảnh Báo

Hệ thống tạo **cảnh báo theo định dạng Suricata** khi phát hiện tấn công:

```
======================================================================
[ALERT] 2026-04-29 20:00:00.123
======================================================================
[**] Intrusion Detected: DDoS [**]
[Classification: Network Intrusion Detection]
[Priority: CRITICAL]

Flow Information:
  Destination Port: 80
  Flow Duration: 1.5
  Tot Fwd Pkts: 500

  Prediction: DDoS
  Confidence: 98.50%
  Action: ALERT - Logging to console
======================================================================
```

## Các Tính Năng Chính

1. **18 Đặc trưng tối ưu**: Được chọn dựa trên nghiên cứu CICIDS2017
   - Protocol, Flow Duration, Packet counts
   - Packet lengths (Fwd/Bwd mean)
   - Flow rates (Bytes/s, Packets/s)
   - TCP flags (SYN, ACK, FIN, RST, PSH, URG)

2. **Cân bằng lớp**: Pipeline SMOTE + RandomUnderSampler
   - Giữ nguyên pattern của các lớp thiểu số
   - Ngăn ngừa bias của lớp đa số
   - Xử lý hiệu quả về bộ nhớ

3. **Phát hiện đa lớp**: Nhận diện các loại tấn công:
   - DoS (GoldenEye, Hulk, Slowhttptest, slowloris)
   - DDoS
   - Bot
   - PortScan
   - FTP-Patator, SSH-Patator
   - Web Attacks (Brute Force, SQL Injection, XSS)
   - Heartbleed
   - Infiltration

## Phân Công Thành Viên (TV)

| Thành viên | Nhiệm vụ |
|------------|----------|
| TV1 | Tiền xử lý & EDA |
| TV2 | Cân bằng dữ liệu |
| TV3 | Logistic Regression, SVM |
| TV4 | Naive Bayes, KNN |
| TV5 | Random Forest & Cảnh báo thời gian thực |

## Tham Khảo

- [CICIDS2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html)
- [Network Intrusion Detection with ML](https://github.com/marxgoo/Network-intrusion-detection-ml)
- [Kaggle: Intrusion Detection System](https://www.kaggle.com/code/ujjwalks9/intrusion-detection-system)

## Giấy Phép

Dự án này phục vụ mục đích học tập trong bài lab của trường đại học.
