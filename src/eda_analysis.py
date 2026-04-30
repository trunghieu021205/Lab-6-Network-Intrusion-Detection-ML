import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Force utf-8 encoding for Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.config import SELECTED_FEATURES, REPORTS_DIR

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")

def load_real_data():
    """Load dữ liệu đã được xử lý. Ưu tiên cleaned_data.csv, fallback dummy."""
    # Ưu tiên 1: cleaned_data.csv (từ TV1 preprocessing)
    filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed', 'cleaned_data.csv')
    # Ưu tiên 2: dummy_data.csv (để test khi chưa có data thật)
    dummy_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'dummy', 'dummy_data.csv')

    if os.path.exists(filepath):
        print(f"[INFO] Loading real data: {filepath}")
        df = pd.read_csv(filepath)
        print(f"[INFO] Loaded {df.shape[0]:,} rows, {df.shape[1]} columns")
        return df
    elif os.path.exists(dummy_path):
        print(f"[WARNING] cleaned_data.csv not found! Using dummy data: {dummy_path}")
        df = pd.read_csv(dummy_path)
        print(f"[INFO] Loaded {df.shape[0]:,} rows (dummy)")
        return df
    else:
        print("[ERROR] Không tìm thấy dữ liệu. Hãy chạy preprocessing.py hoặc create_dummy_data.py trước!")
        return None
def plot_attack_distribution(df):
    if 'Label' not in df.columns:
        raise ValueError("Thieu cot 'Label' trong du lieu dau vao.")
    
    attack_counts = df['Label'].value_counts(dropna=False)
    if attack_counts.empty:
        print("Khong co du lieu ve phan bo Label.")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # bieu do cot
    axes[0].bar(attack_counts.index, attack_counts.values, color='#5B9BD5', edgecolor='black')
    axes[0].set_title('Phan bo cac loai traffic (so luong)')
    axes[0].set_xlabel('Loai')
    axes[0].set_ylabel('So luong')
    axes[0].tick_params(axis='x', rotation=45)
    
    # bieu do tron
    axes[1].pie(attack_counts.values, labels=attack_counts.index, autopct='%1.1f%%', startangle=90)
    axes[1].set_title('Phan bo cac loai traffic (ty le %)')
    plt.tight_layout()
    eda_dir = os.path.join(REPORTS_DIR, 'eda_plots')
    os.makedirs(eda_dir, exist_ok=True)
    plt.savefig(os.path.join(eda_dir, 'attack_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Da luu: attack_distribution.png")

def plot_correlation_heatmap(df):
    corr = df[SELECTED_FEATURES].corr()
    plt.figure(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0, square=True, linewidths=0.5)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    eda_dir = os.path.join(REPORTS_DIR, 'eda_plots')
    os.makedirs(eda_dir, exist_ok=True)
    plt.savefig(os.path.join(eda_dir, 'correlation_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Da luu: correlation_heatmap.png")

def main():
    print("TV3: EDA tren du lieu that")
    df = load_real_data()
    if df is not None:
        plot_attack_distribution(df)
        plot_correlation_heatmap(df)
        print("EDA hoan tat.")

if __name__ == "__main__":
    main()
