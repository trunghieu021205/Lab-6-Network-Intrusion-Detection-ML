import pandas as pd
import numpy as np
import os
import sys

# Force utf-8 encoding for prints
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from sklearn.datasets import make_classification

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.config import SELECTED_FEATURES, DATA_DUMMY_DIR, DATA_RAW_DIR

def create_dummy_data(
    n_samples=5000,
    n_features=18,
    n_informative=12,
    n_redundant=3,
    n_classes=2,
    weights=[0.85, 0.15],
    random_state=42
):
    print("=" * 50)
    print("TV1: GENERATE DUMMY DATA FOR TESTING")
    print("=" * 50)

    # 1. Generate synthetic dataset
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=n_classes,
        weights=weights,
        random_state=random_state
    )

    # 2. Build DataFrame using exact feature names from config
    df = pd.DataFrame(X, columns=SELECTED_FEATURES)
    
    # Map label numbers to strings to simulate raw dataset
    labels = np.array(["BENIGN", "DDoS"])
    df["Label"] = labels[y]

    # 3. Save standard dummy data (for TV3, TV4 testing if they don't have final_data)
    os.makedirs(DATA_DUMMY_DIR, exist_ok=True)
    dummy_path = os.path.join(DATA_DUMMY_DIR, "dummy_data.csv")
    df.to_csv(dummy_path, index=False)
    print(f"Created standard dummy data: {dummy_path}")

    # 4. Save fake RAW data (to test preprocessing.py)
    # We will split it into 2 files and inject some dirty data (NaN, Inf)
    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    
    df1 = df.iloc[:2500].copy()
    df2 = df.iloc[2500:].copy()

    # Inject dirty data into df1
    df1.loc[10:20, 'Flow Duration'] = np.nan
    df1.loc[50:60, 'Tot Fwd Pkts'] = np.inf

    # Inject dirty data into df2
    df2.loc[100:110, 'Flow Byt/s'] = -np.inf
    
    raw1_path = os.path.join(DATA_RAW_DIR, "fake_raw_part1.csv")
    raw2_path = os.path.join(DATA_RAW_DIR, "fake_raw_part2.csv")
    
    df1.to_csv(raw1_path, index=False)
    df2.to_csv(raw2_path, index=False)
    print(f"Created fake raw data 1 (with NaN, Inf): {raw1_path}")
    print(f"Created fake raw data 2 (with -Inf): {raw2_path}")
    
    print("\nYou can now run 'python src/preprocessing.py'!")

if __name__ == "__main__":
    create_dummy_data()