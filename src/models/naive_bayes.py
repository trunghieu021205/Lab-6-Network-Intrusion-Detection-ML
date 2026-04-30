"""
TV3: Naive Bayes Model
Phát hiện xâm nhập mạng bằng thuật toán Gaussian Naive Bayes

Phân công: TV3 (EDA Lead) chịu trách nhiệm model này.
"""

import os
import sys
from sklearn.naive_bayes import GaussianNB

# Force utf-8 encoding for Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Thêm thư mục gốc vào sys.path để import được các module khác
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.config import DATA_PROCESSED_DIR, DATA_DUMMY_DIR, RANDOM_STATE
from src.train_utils import load_data, train_and_evaluate, save_model


def run_naive_bayes(filepath):
    """
    Train và đánh giá mô hình Gaussian Naive Bayes.

    Args:
        filepath (str): Đường dẫn đến file CSV chứa dữ liệu.

    Returns:
        model: Naive Bayes model đã được train.
    """
    print("=" * 50)
    print("TV3: GAUSSIAN NAIVE BAYES MODEL")
    print("=" * 50)

    # 1. Load dữ liệu và chia train/test
    print("\n[1/3] Loading data...")
    X_train, X_test, y_train, y_test = load_data(filepath)

    # 2. Khởi tạo model Gaussian Naive Bayes
    # GaussianNB: giả định mỗi feature tuân theo phân phối Gaussian (chuẩn)
    # Phù hợp với dữ liệu liên tục, đơn giản và nhanh
    print("\n[2/3] Training Naive Bayes model...")
    print("      Variant: GaussianNB (assumes Gaussian feature distributions)")
    model = GaussianNB()

    # 3. Train và đánh giá
    print("\n[3/3] Evaluating...")
    model, y_pred, report = train_and_evaluate(
        model, X_train, X_test, y_train, y_test, model_name="NaiveBayes"
    )

    # 4. Lưu model
    save_model(model, "naive_bayes")

    print("\n" + "=" * 50)
    print("Classification Report:\n")
    print(report)
    print("=" * 50)
    print("Naive Bayes hoan thanh!")
    print("Ket qua da luu: results/NaiveBayes_report.txt | reports/NaiveBayes_confusion_matrix.png")

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

    print(f"\n[INFO] Chay Naive Bayes voi file: {data_path}")
    run_naive_bayes(data_path)