# src/config.py
import os

# Đường dẫn
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(BASE_DIR, 'data/raw')
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, 'data/processed')
DATA_DUMMY_DIR = os.path.join(BASE_DIR, 'data/dummy')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')

# 18 features được chọn (theo lab gốc)
SELECTED_FEATURES = [
    'Protocol', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
    'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Mean',
    'Bwd Pkt Len Mean', 'Flow Byt/s', 'Flow Pkts/s', 'Pkt Len Mean',
    'Pkt Len Std', 'SYN Flag Cnt', 'ACK Flag Cnt', 'FIN Flag Cnt',
    'RST Flag Cnt', 'PSH Flag Cnt', 'URG Flag Cnt'
]

# Random seed để kết quả reproducible
RANDOM_STATE = 42

# Tỉ lệ train/test
TEST_SIZE = 0.2

# Cho SVM (sample fraction để tránh treo máy)
SVM_SAMPLE_FRACTION = 0.2  # Chỉ dùng 20% dữ liệu cho SVM