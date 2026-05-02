"""
TV5: Random Forest Model
Phát hiện xâm nhập mạng bằng thuật toán Random Forest (Best Model)

Phân công: TV5 (Realtime Lead) chịu trách nhiệm model này.
Random Forest là mô hình chính được sử dụng cho hệ thống realtime_alert.py
"""

import os
import sys
from sklearn.ensemble import RandomForestClassifier

# Force utf-8 encoding for Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Thêm thư mục gốc vào sys.path để import được các module khác
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.config import DATA_PROCESSED_DIR, DATA_DUMMY_DIR, RANDOM_STATE
from src.train_utils import load_data, train_and_evaluate, save_model


def run_random_forest(filepath):
    """
    Train và đánh giá mô hình Random Forest.

    Args:
        filepath (str): Đường dẫn đến file CSV chứa dữ liệu.

    Returns:
        model: Random Forest model đã được train.
    """
    print("=" * 50)
    print("TV5: RANDOM FOREST MODEL (Best Model)")
    print("=" * 50)

    # 1. Load dữ liệu và chia train/test
    print("\n[1/3] Loading data...")
    X_train, X_test, y_train, y_test = load_data(filepath)

    # 2. Khởi tạo model Random Forest với tham số tối ưu
    # n_estimators=200: số cây quyết định
    # max_depth=20: giới hạn độ sâu để tránh overfitting
    # min_samples_split=5, min_samples_leaf=2: điều kiện dừng
    # n_jobs=-1: sử dụng tất cả CPU cores
    print("\n[2/3] Training Random Forest model...")
    print("      n_estimators: 200 | max_depth: 20 | n_jobs: -1")
    model = RandomForestClassifier(
        n_estimators=200,
        #max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )

    # 3. Train và đánh giá
    print("\n[3/3] Evaluating...")
    model, y_pred, report = train_and_evaluate(
        model, X_train, X_test, y_train, y_test, model_name="RandomForest"
    )

    # 4. Lưu model (dùng cho realtime_alert.py)
    save_model(model, "random_forest")

    print("\n" + "=" * 50)
    print("Classification Report:\n")
    print(report)
    print("=" * 50)
    print("Random Forest hoan thanh!")
    print("Ket qua da luu: results/RandomForest_report.txt | reports/RandomForest_confusion_matrix.png")
    print("Model da luu: models/random_forest.pkl (dung cho realtime_alert.py)")

    return model


# ============================================================
# CHẠY TRỰC TIẾP - ƯU TIÊN final_data.csv, fallback dummy_data.csv
# ============================================================
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    # Ưu tiên 1: final_data.csv từ TV2 (dữ liệu thật đã cân bằng)
    final_path = os.path.join(BASE_DIR, "data", "processed", "final_data.csv")

    # Ưu tiên 2: cleaned_data.csv từ TV1 (chưa cân bằng nhưng vẫn dùng được)
    cleaned_path = os.path.join(BASE_DIR, "data", "processed", "cleaned_data.csv")

    # Ưu tiên 3: dummy_data.csv (để test)
    dummy_path = os.path.join(BASE_DIR, "data", "dummy", "dummy_data.csv")

    if os.path.exists(final_path):
        print(f"[INFO] Tim thay du lieu that (balanced): {final_path}")
        data_path = final_path
    elif os.path.exists(cleaned_path):
        print(f"[WARNING] Khong co final_data.csv, dung cleaned_data.csv")
        print(f"[INFO] Goi y: Chay balancing.py truoc de co ket qua tot hon")
        data_path = cleaned_path
    elif os.path.exists(dummy_path):
        print("[WARNING] Khong tim thay du lieu that!")
        print(f"[INFO] Fallback: dung dummy_data.csv de test")
        data_path = dummy_path
    else:
        print("[INFO] Khong co du lieu, dang tao dummy data...")
        from src.create_dummy_data import create_dummy_data
        create_dummy_data()
        data_path = dummy_path

    print(f"\n[INFO] Chay Random Forest voi file: {data_path}")
    run_random_forest(data_path)
