"""
TV4: K-Nearest Neighbors (KNN) Model
Phát hiện xâm nhập mạng bằng thuật toán KNN
"""

import os
import sys
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# Force utf-8 encoding for prints
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Thêm thư mục gốc vào sys.path để import được các module khác
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.config import DATA_DUMMY_DIR, SELECTED_FEATURES, RANDOM_STATE
from src.train_utils import load_data, train_and_evaluate, save_model


def run_knn(filepath):
    """
    Train và đánh giá mô hình KNN.

    Args:
        filepath (str): Đường dẫn đến file CSV chứa dữ liệu.

    Returns:
        model: KNN model đã được train.
    """
    print("=" * 50)
    print("TV4: K-NEAREST NEIGHBORS (KNN) MODEL")
    print("=" * 50)

    # 1. Load dữ liệu và chia train/test
    print("\n[1/3] Loading data...")
    X_train, X_test, y_train, y_test = load_data(filepath)

    # 2. Khởi tạo model KNN
    # n_neighbors=5: sử dụng 5 điểm lân cận gần nhất
    # metric='minkowski': khoảng cách Minkowski (mặc định, tương đương Euclidean với p=2)
    # n_jobs=-1: sử dụng tất cả CPU cores để tăng tốc
    print("\n[2/3] Training KNN model (n_neighbors=5)...")
    model = KNeighborsClassifier(
        n_neighbors=5,
        metric='minkowski',
        p=2,
        n_jobs=-1
    )

    # 3. Train và đánh giá
    print("\n[3/3] Evaluating...")
    model, y_pred, report = train_and_evaluate(
        model, X_train, X_test, y_train, y_test, model_name="TV4_knn"
    )

    # 4. Lưu model
    save_model(model, "TV4_knn")

    print("\n" + "=" * 50)
    print("Classification Report:\n")
    print(report)
    print("=" * 50)
    print("KNN hoàn thành! Kết quả đã lưu: results/TV4_knn.txt | reports/TV4_knn_confusion_matrix.png")

    return model


# ============================================================
# CHẠY TRỰC TIẾP - ƯU TIÊN final_data.csv, fallback dummy_data.csv
# ============================================================
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    # Ưu tiên 1: final_data.csv từ TV2 (dữ liệu thật đã cân bằng)
    final_path = os.path.join(BASE_DIR, "data", "processed", "final_data.csv")

    # Ưu tiên 2: dummy_data.csv (để test)
    dummy_path = os.path.join(BASE_DIR, "data", "dummy", "dummy_data.csv")

    if os.path.exists(final_path):
        print(f"[INFO] Tìm thấy dữ liệu thật: {final_path}")
        print("[INFO] Sử dụng final_data.csv (dữ liệu thật từ TV2)")
        data_path = final_path
    elif os.path.exists(dummy_path):
        print("[WARNING] Không tìm thấy final_data.csv!")
        print(f"[WARNING] Đặt file vào: {final_path}")
        print(f"[INFO] Fallback: dùng dummy_data.csv để test")
        data_path = dummy_path
    else:
        print("[INFO] Không có dữ liệu, đang tạo dummy data...")
        from src.create_dummy_data import create_dummy_data
        create_dummy_data()
        data_path = dummy_path

    print(f"\n[INFO] Chạy KNN với file: {data_path}")
    run_knn(data_path)
