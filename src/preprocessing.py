import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.config import DATA_RAW_DIR, DATA_PROCESSED_DIR, SELECTED_FEATURES


def load_and_merge_data():
    if not os.path.exists(DATA_RAW_DIR):
        raise FileNotFoundError(f"Thư mục {DATA_RAW_DIR} không tồn tại.")

    all_files = [
        f for f in os.listdir(DATA_RAW_DIR)
        if f.endswith('.csv') or f.endswith('.parquet')
    ]

    if not all_files:
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu trong {DATA_RAW_DIR}.")

    df_list = []
    columns_to_load = SELECTED_FEATURES + ['Label']
    
    RENAME_DICT = {
        'Total Fwd Packets': 'Tot Fwd Pkts',
        'Total Backward Packets': 'Tot Bwd Pkts',
        'Total Length of Fwd Packets': 'TotLen Fwd Pkts',
        'Total Length of Bwd Packets': 'TotLen Bwd Pkts',
        'Fwd Packet Length Mean': 'Fwd Pkt Len Mean',
        'Bwd Packet Length Mean': 'Bwd Pkt Len Mean',
        'Flow Bytes/s': 'Flow Byt/s',
        'Flow Packets/s': 'Flow Pkts/s',
        'Packet Length Mean': 'Pkt Len Mean',
        'Packet Length Std': 'Pkt Len Std',
        'SYN Flag Count': 'SYN Flag Cnt',
        'ACK Flag Count': 'ACK Flag Cnt',
        'FIN Flag Count': 'FIN Flag Cnt',
        'RST Flag Count': 'RST Flag Cnt',
        'PSH Flag Count': 'PSH Flag Cnt',
        'URG Flag Count': 'URG Flag Cnt'
    }

    for file in all_files:
        filepath = os.path.join(DATA_RAW_DIR, file)

        try:
            print(f"Reading: {file}")

            # =========================
            # CSV
            # =========================
            if file.endswith(".csv"):
                temp_df = pd.read_csv(filepath, nrows=1)
                
                # Map từ tên gốc -> tên viết tắt (nếu có trong RENAME_DICT)
                actual_cols = {}
                for col in temp_df.columns:
                    stripped = col.strip()
                    mapped_name = RENAME_DICT.get(stripped, stripped)
                    actual_cols[col] = mapped_name

                #  FIX 1: check Label
                if 'Label' not in actual_cols.values():
                    print(f"Bỏ qua {file}: không có cột Label")
                    continue

                usecols = [
                    col for col, mapped in actual_cols.items()
                    if mapped in columns_to_load
                ]

                df = pd.read_csv(filepath, usecols=usecols, low_memory=False)
                df.rename(columns=actual_cols, inplace=True)

            # =========================
            # PARQUET
            # =========================
            elif file.endswith(".parquet"):
                try:
                    df = pd.read_parquet(filepath, engine="pyarrow")
                except ImportError:
                    #  FIX 2: không nuốt lỗi
                    raise ImportError("Thiếu thư viện pyarrow. Chạy: pip install pyarrow")

                df.columns = df.columns.str.strip()
                df.rename(columns=RENAME_DICT, inplace=True)

                #  FIX 1: check Label
                if 'Label' not in df.columns:
                    print(f" Bỏ qua {file}: không có cột Label")
                    continue

                df = df[[c for c in df.columns if c in columns_to_load]]

            else:
                continue

            df_list.append(df)
            print(f" Loaded {file}: {df.shape}")

        except ImportError as ie:
            #  FIX 2: dừng luôn nếu thiếu dependency
            raise ie
        except Exception as e:
            print(f"Lỗi file {file}: {e}")

    if not df_list:
        raise ValueError(" Không load được file nào hợp lệ.")

    df_merged = pd.concat(df_list, ignore_index=True)
    print(f"Merged shape: {df_merged.shape}")

    return df_merged


def clean_data(df):
    print("\n--- Bắt đầu làm sạch dữ liệu ---")

    # Loại bỏ các hàng bị lỗi lặp lại header (ví dụ Label == 'Label')
    if 'Label' in df.columns:
        before_drop = len(df)
        df = df[df['Label'] != 'Label']
        if before_drop - len(df) > 0:
            print(f"  Drop header rows: {before_drop - len(df)} rows")
        # Chuẩn hóa nhãn (xóa khoảng trắng thừa)
        df['Label'] = df['Label'].astype(str).str.strip()

    # Ép kiểu dữ liệu sang số cho tất cả các cột đặc trưng (sẽ chuyển 'Infinity' thành NaN)
    features_only = [c for c in df.columns if c != 'Label']
    for col in features_only:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # xử lý NaN / inf
    for col in features_only:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"  Fill NaN {col} = {median_val}")

    # drop zero variance
    features_only = [c for c in df.columns if c != 'Label']
    zero_var_cols = [c for c in features_only if df[c].nunique() <= 1]

    if zero_var_cols:
        df = df.drop(columns=zero_var_cols)
        print(f"  Drop zero-var: {zero_var_cols}")

    # drop duplicates
    before = len(df)
    df = df.drop_duplicates()
    print(f"  Drop duplicates: {before - len(df)} rows")

    #  FIX 3: downcast an toàn
    for col in df.select_dtypes(include=['int64']).columns:
        if (df[col] < 0).any():
            df[col] = pd.to_numeric(df[col], downcast='integer')
        else:
            df[col] = pd.to_numeric(df[col], downcast='unsigned')

    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')

    return df


def encode_labels(df):
    #  optional: check thêm để chắc chắn
    if 'Label' not in df.columns:
        raise ValueError(" Không tìm thấy cột Label để encode.")

    if df['Label'].isna().any():
        raise ValueError(" Label chứa NaN → dữ liệu không hợp lệ.")

    le = LabelEncoder()
    df['Label'] = le.fit_transform(df['Label'])
    return df, le


def main():
    print("=" * 50)
    print("TV1: PREPROCESSING")
    print("=" * 50)

    df_raw = load_and_merge_data()

    df_clean = clean_data(df_raw)

    df_clean, label_encoder = encode_labels(df_clean)

    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    output_path = os.path.join(DATA_PROCESSED_DIR, 'cleaned_data.csv')

    df_clean.to_csv(output_path, index=False)

    print(f"\n Saved: {output_path}")
    print(f"Shape: {df_clean.shape}")
    print(f"Classes: {label_encoder.classes_}")


if __name__ == "__main__":
    main()