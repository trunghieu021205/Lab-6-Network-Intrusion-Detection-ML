import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.config import DATA_DUMMY_DIR, REPORTS_DIR

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")

def load_data():
    filepath = os.path.join(DATA_DUMMY_DIR, 'dummy_data.csv')
    if not os.path.exists(filepath):
        print("Khong tim thay dummy_data.csv. Cho TV1 tao du lieu.")
        return None
    df = pd.read_csv(filepath)
    print("Da load du lieu. So dong:", df.shape[0], " So cot:", df.shape[1])
    return df

def plot_attack_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    attack_counts = df['Label'].value_counts()
    
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
    out_dir = os.path.join(REPORTS_DIR, 'eda_plots')
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, 'attack_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Da luu bieu do: attack_distribution.png")

def main():
    print("TV3: Bat dau phan tich EDA voi dummy data")
    df = load_data()
    if df is not None:
        plot_attack_distribution(df)
        print("Hoan thanh EDA co ban.")
        print("Kich thuoc du lieu:", df.shape)

if __name__ == "__main__":
    main()