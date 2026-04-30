import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.config import SELECTED_FEATURES

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")

def load_real_data():
    filepath = 'data/processed/cleaned_data.csv'
    if not os.path.exists(filepath):
        print("Khong tim thay cleaned_data.csv. Hay dat file vao data/processed/")
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
    os.makedirs('reports/eda_plots', exist_ok=True)
    plt.savefig('reports/eda_plots/attack_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Da luu: attack_distribution.png")

def plot_correlation_heatmap(df):
    corr = df[SELECTED_FEATURES].corr()
    plt.figure(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0, square=True, linewidths=0.5)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('reports/eda_plots/correlation_heatmap.png', dpi=150, bbox_inches='tight')
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
