"""
TV3: Exploratory Data Analysis (EDA)
Phân tích dữ liệu và trực quan hóa các đặc trưng quan trọng.

Phân công: TV3 (EDA Lead) chịu trách nhiệm module này.
"""

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

# Set aesthetic style
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

def plot_label_distribution(df):
    """Trực quan hóa phân bố nhãn (Bar + Pie)."""
    if 'Label' not in df.columns:
        print("[SKIP] No Label column")
        return
    
    attack_counts = df['Label'].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Pie chart
    ax1 = axes[0]
    # Group small classes as "Other" for cleaner pie chart
    threshold = 0.02
    small_classes = attack_counts[attack_counts / attack_counts.sum() < threshold]
    large_classes = attack_counts[attack_counts / attack_counts.sum() >= threshold]
    
    plot_data = large_classes.copy()
    if len(small_classes) > 0:
        plot_data['Other'] = small_classes.sum()
    
    ax1.pie(plot_data, labels=plot_data.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Set3"))
    ax1.set_title('Phân bố Traffic (Tỉ lệ %)', fontweight='bold')
    
    # Bar chart (Log scale)
    ax2 = axes[1]
    attack_counts.plot(kind='bar', ax=ax2, color='steelblue', edgecolor='black')
    ax2.set_yscale('log')
    ax2.set_title('Số lượng Traffic (Log Scale)', fontweight='bold')
    ax2.set_ylabel('Count (log)')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    save_path = os.path.join(REPORTS_DIR, 'eda_plots', 'eda_label_distribution.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {save_path}")

def plot_correlation_heatmap(df):
    """Vẽ Feature Correlation Heatmap với annotation."""
    if not all(col in df.columns for col in SELECTED_FEATURES):
        print("[SKIP] Missing features for correlation heatmap")
        return

    # Sample if too large for performance
    df_sample = df[SELECTED_FEATURES]
    if len(df_sample) > 10000:
        df_sample = df_sample.sample(n=10000, random_state=42)
        
    corr = df_sample.corr()
    plt.figure(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, annot_kws={'size': 8})
    
    plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(REPORTS_DIR, 'eda_plots', 'eda_correlation_heatmap.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {save_path}")

def plot_feature_distributions(df, top_n=6):
    """Phân phối các đặc trưng hàng đầu (Benign vs Attack)."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != 'Label' and c in SELECTED_FEATURES]
    
    if len(numeric_cols) < 1:
        return
    
    # Lấy top_n features có độ biến thiên cao nhất
    top_features = df[numeric_cols].var().sort_values(ascending=False).head(top_n).index.tolist()
    
    n_cols = 3
    n_rows = (len(top_features) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()
    
    for i, feature in enumerate(top_features):
        ax = axes[i]
        # Giả định Benign là lớp phổ biến nhất hoặc nhãn cụ thể
        sns.histplot(data=df.sample(n=min(len(df), 5000)), x=feature, hue='Label', 
                     kde=True, ax=ax, element="step", common_norm=False)
        ax.set_title(f'Distribution: {feature}', fontsize=11, fontweight='bold')
        ax.get_legend().remove() if ax.get_legend() else None
    
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.suptitle('Phân phối Đặc trưng (Top Features)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_path = os.path.join(REPORTS_DIR, 'eda_plots', 'eda_feature_distributions.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {save_path}")

def plot_imbalance_analysis(df):
    """Phân tích mất cân bằng dữ liệu."""
    if 'Label' not in df.columns: return
    
    label_counts = df['Label'].value_counts()
    max_count = label_counts.max()
    min_count = label_counts.min()
    ratio = max_count / min_count if min_count > 0 else float('inf')
    
    plt.figure(figsize=(10, 6))
    ax = label_counts.plot(kind='barh', color='salmon', edgecolor='black')
    plt.title(f'Data Imbalance Analysis (Ratio: {ratio:.1f}:1)', fontweight='bold')
    plt.xlabel('Count')
    
    # Add labels
    for i, v in enumerate(label_counts):
        ax.text(v + 3, i, f'{v:,}', va='center')
        
    plt.tight_layout()
    save_path = os.path.join(REPORTS_DIR, 'eda_plots', 'eda_class_imbalance.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {save_path}")

def main():
    print("\n" + "=" * 50)
    print("TV3: EXPLORATORY DATA ANALYSIS (EDA) - UPGRADED")
    print("=" * 50)
    
    df = load_real_data()
    if df is not None:
        os.makedirs(os.path.join(REPORTS_DIR, 'eda_plots'), exist_ok=True)
        
        print("\n[1/4] Plotting label distribution...")
        plot_label_distribution(df)
        
        print("[2/4] Plotting correlation heatmap...")
        plot_correlation_heatmap(df)
        
        print("[3/4] Plotting feature distributions...")
        plot_feature_distributions(df)
        
        print("[4/4] Plotting imbalance analysis...")
        plot_imbalance_analysis(df)
        
        print("\n" + "=" * 50)
        print("EDA Hoàn tất! Kết quả lưu tại: reports/eda_plots/")
        print("=" * 50)

if __name__ == "__main__":
    main()
