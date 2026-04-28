import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import joblib
import os
import sys

# Force utf-8 encoding for prints
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.config import DATA_PROCESSED_DIR, MODELS_DIR, SELECTED_FEATURES, RANDOM_STATE

def load_cleaned_data():
    """Load dữ liệu đã được TV1 làm sạch"""
    filepath = os.path.join(DATA_PROCESSED_DIR, 'cleaned_data.csv')
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Không tìm thấy {filepath}. Hãy chạy preprocessing.py trước!")
    df = pd.read_csv(filepath)
    print(f"Loaded cleaned data: {df.shape}")
    return df

def apply_balancing(df):
    """Áp dụng StandardScaler + SMOTE + RandomUnderSampler"""
    X = df[SELECTED_FEATURES]
    y = df['Label']
    
    print("\nPhân bố class gốc:")
    print(y.value_counts())
    
    # 1. StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=SELECTED_FEATURES)
    
    # Lưu scaler
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
    print("Đã lưu scaler.pkl")
    
    # 2. Balancing
    class_counts = y.value_counts()
    max_count = class_counts.max()
    min_count = class_counts.min()
    
    # Tính toán k_neighbors tự động (tránh lỗi với class quá nhỏ)
    k_neighbors = min(5, min_count - 1) if min_count > 1 else 1

    # NGƯỠNG ĐỘNG (Học từ tài liệu tham khảo)
    # Thay vì dùng SMOTE ép mọi class lên 2.2 triệu mẫu (gây Overfit và cháy RAM),
    # ta sẽ tập trung vào Undersampling (giảm mẫu đa số) xuống mức an toàn.
    UPPER_LIMIT = min(max_count, 100000)
    LOWER_LIMIT = min(max_count, 20000)
    
    under_strategy = {c: min(count, UPPER_LIMIT) for c, count in class_counts.items()}
    over_strategy = {c: max(under_strategy[c], LOWER_LIMIT) for c in class_counts.index}
    
    print("  Đang chạy RandomUnderSampler (Giảm bớt đa số)...")
    undersampler = RandomUnderSampler(sampling_strategy=under_strategy, random_state=RANDOM_STATE)
    X_under, y_under = undersampler.fit_resample(X_scaled, y)
    
    print("  Đang chạy SMOTE (Tăng cường thiểu số)...")
    smote = SMOTE(sampling_strategy=over_strategy, random_state=RANDOM_STATE, k_neighbors=k_neighbors)
    X_final, y_final = smote.fit_resample(X_under, y_under)
    
    print("\nPhân bố class sau khi cân bằng:")
    print(pd.Series(y_final).value_counts())
    
    # Ghép lại thành DataFrame
    df_balanced = pd.DataFrame(X_final, columns=SELECTED_FEATURES)
    df_balanced['Label'] = y_final
    
    return df_balanced

def main():
    print("=" * 50)
    print("TV2: CÂN BẰNG DỮ LIỆU (BALANCING)")
    print("=" * 50)
    
    df_clean = load_cleaned_data()
    df_balanced = apply_balancing(df_clean)
    
    output_path = os.path.join(DATA_PROCESSED_DIR, 'final_data.csv')
    df_balanced.to_csv(output_path, index=False)
    
    print(f"\n Đã lưu dữ liệu cân bằng tại: {output_path}")
    print(f"Kích thước cuối cùng: {df_balanced.shape}")

if __name__ == "__main__":
    main()
