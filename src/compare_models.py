# src/compare_models.py
"""
Compare all 5 ML models and generate visualization.
TV5: Model Comparison with Visualization
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import numpy as np

# Force utf-8 encoding for Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import RESULTS_DIR, REPORTS_DIR

# =====================================================
# Chuẩn hóa: tên model → tên file report
# =====================================================
MODEL_FILES = {
    'Naive Bayes':         'NaiveBayes_report.txt',
    'Logistic Regression': 'LogisticRegression_report.txt',
    'KNN':                 'KNN_report.txt',
    'SVM':                 'SVM_report.txt',
    'Random Forest':       'RandomForest_report.txt',
}

# Tên model trong report → tên model hiển thị (để load PNG)
MODEL_PNG_NAMES = {
    'Naive Bayes':         'NaiveBayes',
    'Logistic Regression': 'LogisticRegression',
    'KNN':                 'KNN',
    'SVM':                 'SVM',
    'Random Forest':       'RandomForest',
}


# =====================================================
# 1. Parse metrics từ file report
# =====================================================
def parse_results():
    """Đọc metrics từ các file _report.txt và tạo DataFrame so sánh."""
    results = []

    for model_name, filename in MODEL_FILES.items():
        filepath = os.path.join(RESULTS_DIR, filename)
        if not os.path.exists(filepath):
            print(f"[SKIP] {filename} not found — chưa train model này.")
            continue

        accuracy = precision = recall = f1 = None
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('Accuracy:'):
                    accuracy = float(line.split(':')[1].strip())
                elif line.startswith('Precision (weighted):'):
                    precision = float(line.split(':')[1].strip())
                elif line.startswith('Recall (weighted):'):
                    recall = float(line.split(':')[1].strip())
                elif line.startswith('F1-Score (weighted):'):
                    f1 = float(line.split(':')[1].strip())

        if None in (accuracy, precision, recall, f1):
            print(f"[WARN] {filename}: thiếu metrics, bỏ qua.")
            continue

        results.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
        })

    return pd.DataFrame(results)


# =====================================================
# 2. Bảng so sánh Markdown
# =====================================================
def create_comparison_table():
    """Tạo bảng so sánh và lưu markdown."""
    df = parse_results()

    if df.empty:
        print("[WARN] Không có kết quả nào! Hãy train models trước.")
        return None

    df = df.sort_values('F1-Score', ascending=False).reset_index(drop=True)
    best_model = df.iloc[0]['Model']

    md = "# Model Comparison Results\n\n"
    md += f"**Best Model: {best_model}** (highest F1-Score)\n\n"
    md += "| Model | Accuracy | Precision | Recall | F1-Score |\n"
    md += "|-------|----------|-----------|--------|----------|\n"
    for _, row in df.iterrows():
        star = " :star:" if row['Model'] == best_model else ""
        md += (f"| {row['Model']}{star} | "
               f"{row['Accuracy']:.4f} | {row['Precision']:.4f} | "
               f"{row['Recall']:.4f} | {row['F1-Score']:.4f} |\n")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, 'comparison_table.md')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(md)

    print("\n" + "=" * 70)
    print("MODEL COMPARISON TABLE")
    print("=" * 70)
    print(md)
    print(f"[Saved] {out_path}")
    return df


# =====================================================
# 3. Biểu đồ so sánh metrics
# =====================================================
def plot_metrics_comparison(df):
    """Bar chart so sánh 4 metrics của 5 mô hình."""
    if df is None or df.empty:
        return

    os.makedirs(REPORTS_DIR, exist_ok=True)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(df))
    width = 0.18

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, metric in enumerate(metrics):
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, df[metric], width, label=metric, alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=7, rotation=45)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Model'], rotation=15, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = os.path.join(REPORTS_DIR, 'metrics_comparison.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {out}")


# =====================================================
# 4. Ghép tất cả confusion matrix PNG
# =====================================================
def plot_all_confusion_matrices():
    """
    Load các file *_confusion_matrix.png đã được train_utils.py tạo sẵn,
    ghép thành 1 figure tổng hợp.
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    axes = axes.flatten()

    for idx, (model_name, png_key) in enumerate(MODEL_PNG_NAMES.items()):
        ax = axes[idx]
        png_path = os.path.join(REPORTS_DIR, f"{png_key}_confusion_matrix.png")

        if os.path.exists(png_path):
            img = mpimg.imread(png_path)
            ax.imshow(img)
            ax.set_title(model_name, fontsize=12, fontweight='bold', pad=6)
            ax.axis('off')
            print(f"[OK] Loaded: {png_path}")
        else:
            ax.text(0.5, 0.5,
                    f'{model_name}\nConfusion Matrix PNG\nnot found\n\nRun model first',
                    ha='center', va='center', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax.set_title(model_name, fontsize=12, fontweight='bold')
            ax.axis('off')
            print(f"[SKIP] {model_name}: {png_key}_confusion_matrix.png not found")

    # Ẩn ô thứ 6 (dư)
    axes[5].axis('off')

    plt.suptitle('Confusion Matrices — All Models', fontsize=16, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(REPORTS_DIR, 'all_confusion_matrices.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {out}")


# =====================================================
# 5. Recall theo từng lớp tấn công
# =====================================================
def plot_recall_by_attack_class():
    """Đọc classification report từ file txt, vẽ recall theo từng class."""
    os.makedirs(REPORTS_DIR, exist_ok=True)

    recall_data = {}

    for model_name, filename in MODEL_FILES.items():
        filepath = os.path.join(RESULTS_DIR, filename)
        if not os.path.exists(filepath):
            continue

        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        in_report = False
        class_recalls = {}

        for line in content.split('\n'):
            if 'classification report' in line.lower():
                in_report = True
                continue
            if in_report:
                parts = line.strip().split()
                # Dòng dữ liệu class: [class_id] [precision] [recall] [f1] [support]
                if len(parts) == 5:
                    try:
                        class_id = parts[0]
                        recall_val = float(parts[2])
                        # Bỏ qua dòng tổng hợp
                        if class_id not in ('accuracy', 'macro', 'weighted', 'micro'):
                            class_recalls[class_id] = recall_val
                    except ValueError:
                        continue

        if class_recalls:
            recall_data[model_name] = class_recalls

    if not recall_data:
        print("[SKIP] No per-class recall data found in report files")
        return

    all_classes = sorted(next(iter(recall_data.values())).keys(),
                         key=lambda x: int(x) if x.isdigit() else x)
    x = np.arange(len(all_classes))
    width = 0.15
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (model_name, recalls) in enumerate(recall_data.items()):
        values = [recalls.get(cls, 0) for cls in all_classes]
        offset = (i - len(recall_data) / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=model_name,
               alpha=0.85, color=colors[i % len(colors)])

    ax.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='90% threshold')
    ax.set_xlabel('Attack Class', fontsize=12)
    ax.set_ylabel('Recall', fontsize=12)
    ax.set_title('Recall by Attack Class\n(High Recall = Fewer Missed Attacks)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(all_classes, rotation=45, ha='right', fontsize=9)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = os.path.join(REPORTS_DIR, 'recall_by_attack_class.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {out}")


# =====================================================
# MAIN
# =====================================================
def main():
    print("\n" + "=" * 50)
    print("TV5: MODEL COMPARISON")
    print("=" * 50 + "\n")

    # 1. Bảng so sánh
    df = create_comparison_table()

    # 2. Biểu đồ
    print("\nGenerating plots...")
    print("-" * 30)

    plot_metrics_comparison(df)
    plot_all_confusion_matrices()
    plot_recall_by_attack_class()

    print("\n" + "=" * 50)
    print("All visualizations complete!")
    print(f"  - results/comparison_table.md")
    print(f"  - reports/metrics_comparison.png")
    print(f"  - reports/all_confusion_matrices.png")
    print(f"  - reports/recall_by_attack_class.png")
    print("=" * 50)


if __name__ == "__main__":
    main()
