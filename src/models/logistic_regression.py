"""
TV1: Logistic Regression Model
Phát hiện xâm nhập mạng bằng thuật toán Logistic Regression

Phân công: TV1 (Data Lead) chịu trách nhiệm model này.
"""

import os
import sys
from sklearn.linear_model import LogisticRegression

# Force utf-8 encoding for prints
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Thêm thư mục gốc vào sys.path để import được các module khác
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.config import DATA_PROCESSED_DIR, DATA_DUMMY_DIR, RANDOM_STATE
from src.train_utils import load_data, train_and_evaluate, save_model


def run_logistic_regression(filepath):
    """
    Train và đánh giá mô hình Logistic Regression.

    Args:
        filepath (str): Đường dẫn đến file CSV chứa dữ liệu.

    Returns:
        model: Logistic Regression model đã được train.
    """
    print("=" * 50)
    print("TV1: LOGISTIC REGRESSION MODEL")
    print("=" * 50)

    # 1. Load dữ liệu và chia train/test
    print("\n[1/3] Loading data...")
    X_train, X_test, y_train, y_test = load_data(filepath)

    # 2. Khởi tạo model Logistic Regression
    # max_iter=1000: tăng số vòng lặp để đảm bảo hội tụ với dữ liệu lớn
    # solver='lbfgs': thuật toán tối ưu phù hợp với multiclass
    # multi_class='auto': tự động chọn one-vs-rest hoặc multinomial
    # C=1.0: tham số regularization (L2 mặc định)
    print("\n[2/3] Training Logistic Regression model...")
    print("      Solver: lbfgs | Max iter: 1000 | C: 1.0 | Multi-class: auto")
    model = LogisticRegression(
        solver='lbfgs',
        max_iter=1000,
        C=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1       # Sử dụng tất cả CPU cores
    )

    # 3. Train và đánh giá
    print("\n[3/3] Evaluating...")
    model, y_pred, report = train_and_evaluate(
        model, X_train, X_test, y_train, y_test, model_name="LogisticRegression"
    )

    # 4. Lưu model
    save_model(model, "logistic_regression")

    print("\n" + "=" * 50)
    print("Classification Report:\n")
    print(report)
    print("=" * 50)
    print("Logistic Regression hoàn thành!")
    print("Kết quả đã lưu: results/LogisticRegression_report.txt | reports/LogisticRegression_confusion_matrix.png")

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
        print(f"[INFO] Tìm thấy dữ liệu thật (balanced): {final_path}")
        data_path = final_path
    elif os.path.exists(cleaned_path):
        print(f"[WARNING] Không có final_data.csv, dùng cleaned_data.csv")
        print(f"[INFO] Gợi ý: Chạy balancing.py trước để có kết quả tốt hơn")
        data_path = cleaned_path
    elif os.path.exists(dummy_path):
        print("[WARNING] Không tìm thấy dữ liệu thật!")
        print(f"[INFO] Fallback: dùng dummy_data.csv để test")
        data_path = dummy_path
    else:
        print("[INFO] Không có dữ liệu, đang tạo dummy data...")
        from src.create_dummy_data import create_dummy_data
        create_dummy_data()
        data_path = dummy_path

    print(f"\n[INFO] Chạy Logistic Regression với file: {data_path}")
    run_logistic_regression(data_path)
