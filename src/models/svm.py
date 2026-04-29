"""
TV2: Support Vector Machine (SVM) Model
Phát hiện xâm nhập mạng bằng thuật toán SVM
"""

import os
import sys
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Force utf-8 encoding for prints
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Thêm thư mục gốc vào sys.path để import được các module khác
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.config import DATA_DUMMY_DIR, SELECTED_FEATURES, RANDOM_STATE, SVM_SAMPLE_FRACTION
from src.train_utils import load_data, train_and_evaluate, save_model


def run_svm(filepath):
    """
    Train và đánh giá mô hình SVM.

    Lưu ý: SVM tốn nhiều thời gian với dữ liệu lớn, nên mặc định
    chỉ sử dụng SVM_SAMPLE_FRACTION (20%) của tập train.

    Args:
        filepath (str): Đường dẫn đến file CSV chứa dữ liệu.

    Returns:
        pipeline: SVM pipeline (StandardScaler + SVC) đã được train.
    """
    print("=" * 50)
    print("TV2: SUPPORT VECTOR MACHINE (SVM) MODEL")
    print("=" * 50)

    # 1. Load dữ liệu và chia train/test
    print("\n[1/4] Loading data...")
    X_train, X_test, y_train, y_test = load_data(filepath)

    # 2. Lấy mẫu nhỏ hơn để tránh treo máy (SVM có độ phức tạp O(n^2~3))
    sample_size = int(len(X_train) * SVM_SAMPLE_FRACTION)
    print(f"\n[2/4] Sampling {SVM_SAMPLE_FRACTION*100:.0f}% of training data "
          f"({sample_size}/{len(X_train)} samples) để tăng tốc SVM...")

    X_train_sample = X_train.sample(n=sample_size, random_state=RANDOM_STATE)
    y_train_sample = y_train.loc[X_train_sample.index]

    # 3. Xây dựng Pipeline: StandardScaler + SVC
    # SVM rất nhạy với scale của features, nên PHẢI dùng StandardScaler
    print("\n[3/4] Training SVM pipeline (StandardScaler + SVC)...")
    print("      Kernel: RBF | C: 1.0 | Gamma: scale")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(
            kernel='rbf',    # Radial Basis Function kernel - phù hợp với dữ liệu phi tuyến
            C=1.0,           # Tham số regularization
            gamma='scale',   # Tự động điều chỉnh gamma theo số features
            random_state=RANDOM_STATE,
            probability=False  # Tắt để tăng tốc
        ))
    ])

    # 4. Train và đánh giá
    print("\n[4/4] Evaluating...")
    pipeline, y_pred, report = train_and_evaluate(
        pipeline, X_train_sample, X_test, y_train_sample, y_test, model_name="TV4_SVM"
    )

    # 5. Lưu model
    save_model(pipeline, "TV4_SVM")

    print("\n" + "=" * 50)
    print("Classification Report:\n")
    print(report)
    print("=" * 50)
    print("SVM hoàn thành! Kết quả đã lưu: results/TV4_SVM.txt | reports/TV4_SVM_confusion_matrix.png")

    return pipeline


# ============================================================
# CHẠY TRỰC TIẾP VỚI DUMMY DATA (để test cú pháp)
# ============================================================
if __name__ == "__main__":
    dummy_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "data", "dummy", "dummy_data.csv"
    )

    if not os.path.exists(dummy_path):
        print(f"[INFO] Dummy data không tìm thấy tại: {dummy_path}")
        print("[INFO] Đang tạo dummy data bằng create_dummy_data.py...")
        from src.create_dummy_data import create_dummy_data
        create_dummy_data()

    print(f"\n[INFO] Sử dụng file: {dummy_path}")
    run_svm(dummy_path)
