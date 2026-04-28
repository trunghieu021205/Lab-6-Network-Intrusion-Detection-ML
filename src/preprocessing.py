import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import sys

# Force utf-8 encoding for prints
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.config import DATA_RAW_DIR, DATA_PROCESSED_DIR, SELECTED_FEATURES

def load_and_merge_data():
    """Đọc các file CSV và gộp lại, chỉ lấy đúng các cột cần thiết để tiết kiệm RAM"""
    if not os.path.exists(DATA_RAW_DIR):
        raise FileNotFoundError(f"Thư mục {DATA_RAW_DIR} không tồn tại.")
        
    all_files = [f for f in os.listdir(DATA_RAW_DIR) if f.endswith('.csv')]
    if not all_files:
        raise FileNotFoundError(f"Không tìm thấy file CSV nào trong {DATA_RAW_DIR}.")
        
    df_list = []
    
    # 18 features + Label
    columns_to_load = SELECTED_FEATURES + ['Label']
    
    for file in all_files:
        filepath = os.path.join(DATA_RAW_DIR, file)
        try:
            # Đọc 1 dòng đầu để lấy tên cột thực tế trong file
            temp_df = pd.read_csv(filepath, nrows=1)
            # Lọc bớt khoảng trắng ở tên cột
            actual_cols = {col: col.strip() for col in temp_df.columns}
            
            # Chỉ chọn những cột thuộc danh sách columns_to_load
            usecols = [col for col, stripped in actual_cols.items() if stripped in columns_to_load]
            
            df = pd.read_csv(filepath, usecols=usecols, low_memory=False)
            
            # Đổi lại tên cột cho chuẩn (xóa khoảng trắng thừa)
            df.rename(columns=actual_cols, inplace=True)
            
            df_list.append(df)
            print(f"Loaded {file}: {df.shape}")
        except Exception as e:
            print(f"Lỗi khi đọc file {file}: {e}")
            
    df_merged = pd.concat(df_list, ignore_index=True)
    print(f"Merged shape: {df_merged.shape}")
    return df_merged

def clean_data(df):
    """Xử lý missing, infinite, zero-variance, duplicates"""
    print("\n--- Bắt đầu làm sạch dữ liệu ---")
    # Thay thế infinite bằng NaN
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        
        # Điền missing values bằng median
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"  Điền NaN ở cột {col} bằng {median_val}")
            
    # Drop zero-variance features (chỉ trong danh sách features, bỏ qua Label)
    features_only = [c for c in df.columns if c != 'Label']
    zero_var_cols = [col for col in features_only if df[col].nunique() <= 1]
    if zero_var_cols:
        df = df.drop(columns=zero_var_cols)
        print(f"  Dropped zero-variance columns: {zero_var_cols}")
        
    # Drop duplicates
    before = len(df)
    df = df.drop_duplicates()
    print(f"  Dropped {before - len(df)} duplicate rows")
    
    # Downcast dtypes để tiết kiệm RAM
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='unsigned')
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
        
    return df

def encode_labels(df):
    """Label Encoding cho cột Label"""
    le = LabelEncoder()
    df['Label'] = le.fit_transform(df['Label'])
    return df, le

def main():
    print("=" * 50)
    print("TV1: TIỀN XỬ LÝ DỮ LIỆU (PREPROCESSING)")
    print("=" * 50)
    
    # 1. Load data
    df_raw = load_and_merge_data()
    
    # 2. Clean data
    df_clean = clean_data(df_raw)
    
    # 3. Encode labels
    df_clean, label_encoder = encode_labels(df_clean)
    
    # 4. Save
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    output_path = os.path.join(DATA_PROCESSED_DIR, 'cleaned_data.csv')
    df_clean.to_csv(output_path, index=False)
    
    print(f"\n✅ Đã lưu dữ liệu làm sạch tại: {output_path}")
    print(f"Kích thước cuối cùng: {df_clean.shape}")
    print(f"Các lớp (classes): {label_encoder.classes_}")

if __name__ == "__main__":
    main()
