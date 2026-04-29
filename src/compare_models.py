# src/compare_models.py
"""
Compare all 5 ML models and generate confusion matrices.
TV5: Model Comparison with Visualization
"""
import os
import sys
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Force utf-8 encoding for Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.config import RESULTS_DIR, REPORTS_DIR

# Model name mapping
MODEL_FILES = {
    'Logistic Regression': 'LogisticRegression_report.txt',
    'Naive Bayes': 'NaiveBayes_report.txt',
    'KNN': 'KNN_report.txt',
    'SVM': 'SVM_report.txt',
    'Random Forest': 'RandomForest_report.txt'
}

# Attack class names (from CICIDS2017)
ATTACK_LABELS = [
    'BENIGN', 'Bot', 'DDoS', 'DoS GoldenEye', 'DoS Hulk',
    'DoS Slowhttptest', 'DoS slowloris', 'FTP-Patator', 'Heartbleed',
    'Infiltration', 'PortScan', 'SSH-Patator', 'Web Attack - Brute Force',
    'Web Attack - Sql Injection', 'Web Attack - XSS'
]


def parse_results():
    """Parse all model result files and create comparison DataFrame."""
    results = []
    
    for model_name, filename in MODEL_FILES.items():
        filepath = os.path.join(RESULTS_DIR, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
                accuracy = precision = recall = f1 = None
                
                for line in content.split('\n'):
                    if 'Accuracy:' in line:
                        accuracy = float(line.split(':')[1].strip())
                    elif 'Precision (weighted):' in line:
                        precision = float(line.split(':')[1].strip())
                    elif 'Recall (weighted):' in line:
                        recall = float(line.split(':')[1].strip())
                    elif 'F1-Score (weighted):' in line:
                        f1 = float(line.split(':')[1].strip())
                
                results.append({
                    'Model': model_name,
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1-Score': f1
                })
    
    return pd.DataFrame(results)


def plot_all_confusion_matrices():
    """Generate confusion matrix for each model using prediction data from reports."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    colors = ['Blues', 'Greens', 'Reds', 'Purples', 'Oranges']
    
    for idx, (model_name, filename) in enumerate(MODEL_FILES.items()):
        filepath = os.path.join(RESULTS_DIR, filename)
        if not os.path.exists(filepath):
            continue
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract confusion matrix from report
        cm = None
        lines = content.split('\n')
        in_cm = False
        cm_rows = []
        
        for line in lines:
            if 'confusion matrix' in line.lower():
                in_cm = True
                continue
            if in_cm:
                if line.strip() and not any(x in line.lower() for x in ['precision', 'recall', 'f1', 'accuracy', 'support']):
                    parts = line.split()
                    if parts and parts[0].isdigit():
                        cm_rows.append([int(x) for x in parts if x.isdigit()])
                else:
                    if cm_rows:
                        break
        
        ax = axes[idx]
        
        if cm_rows and len(cm_rows) > 1:
            cm = np.array(cm_rows)
            # Truncate labels if needed
            n_labels = cm.shape[0]
            labels = ATTACK_LABELS[:n_labels] if n_labels <= len(ATTACK_LABELS) else [f'C{i}' for i in range(n_labels)]
            
            sns.heatmap(cm, annot=True, fmt='d', cmap=colors[idx % len(colors)],
                       xticklabels=labels, yticklabels=labels, ax=ax)
        else:
            ax.text(0.5, 0.5, f'{model_name}\nConfusion Matrix\nNot Available',
                   ha='center', va='center', fontsize=12)
        
        ax.set_title(f'{model_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)
    
    # Hide the 6th subplot
    axes[5].axis('off')
    
    plt.suptitle('Confusion Matrices - All Models', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(REPORTS_DIR, 'all_confusion_matrices.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {output_path}")


def plot_confusion_matrix_single(model_name, filename):
    """Generate individual confusion matrix plot for a model."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    filepath = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(filepath):
        print(f"[SKIP] {filename} not found")
        return
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract confusion matrix
    lines = content.split('\n')
    in_cm = False
    cm_rows = []
    
    for line in lines:
        if 'confusion matrix' in line.lower():
            in_cm = True
            continue
        if in_cm:
            if line.strip() and not any(x in line.lower() for x in ['precision', 'recall', 'f1', 'accuracy', 'support']):
                parts = line.split()
                if parts and parts[0].isdigit():
                    cm_rows.append([int(x) for x in parts if x.isdigit()])
            else:
                if cm_rows:
                    break
    
    if not cm_rows or len(cm_rows) <= 1:
        print(f"[SKIP] {model_name}: No confusion matrix data found")
        return
    
    cm = np.array(cm_rows)
    n_labels = cm.shape[0]
    labels = ATTACK_LABELS[:n_labels] if n_labels <= len(ATTACK_LABELS) else [f'C{i}' for i in range(n_labels)]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=labels, yticklabels=labels,
               annot_kws={'size': 10})
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('Actual Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    output_path = os.path.join(REPORTS_DIR, f'{model_name}_confusion_matrix.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {output_path}")


def plot_metrics_comparison(df):
    """Create bar chart comparing metrics across models."""
    if df.empty:
        print("[SKIP] No results to compare")
        return
    
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(df))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, df[metric], width, label=metric, alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8, rotation=45)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Model'], rotation=15, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(REPORTS_DIR, 'metrics_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {output_path}")


def plot_recall_by_attack_class():
    """Plot recall for attack classes - important for cybersecurity (FN is dangerous)."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Read each model's report to get per-class recall
    recall_data = {}
    
    for model_name, filename in MODEL_FILES.items():
        filepath = os.path.join(RESULTS_DIR, filename)
        if not os.path.exists(filepath):
            continue
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse per-class metrics from classification report
        in_report = False
        class_recalls = {}
        
        for line in content.split('\n'):
            if 'classification report' in line.lower():
                in_report = True
                continue
            if in_report:
                parts = line.strip().split()
                if len(parts) >= 4:
                    try:
                        class_name = ' '.join(parts[:-3])
                        recall = float(parts[-3])
                        if class_name not in ['weighted', 'macro', 'micro', 'accuracy']:
                            class_recalls[class_name] = recall
                    except (ValueError, IndexError):
                        continue
        
        if class_recalls:
            recall_data[model_name] = class_recalls
    
    if not recall_data:
        print("[SKIP] No per-class recall data found")
        return
    
    # Create grouped bar chart
    all_classes = sorted(list(next(iter(recall_data.values())).keys()))
    
    x = np.arange(len(all_classes))
    width = 0.15
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (model_name, recalls) in enumerate(recall_data.items()):
        values = [recalls.get(cls, 0) for cls in all_classes]
        offset = (i - 2) * width
        ax.bar(x + offset, values, width, label=model_name, alpha=0.85, color=colors[i % len(colors)])
    
    ax.set_xlabel('Attack Class', fontsize=12)
    ax.set_ylabel('Recall', fontsize=12)
    ax.set_title('Recall by Attack Class (Important: High Recall = Fewer Missed Attacks)', 
                fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(all_classes, rotation=45, ha='right', fontsize=9)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.15)
    ax.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='90% threshold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(REPORTS_DIR, 'recall_by_attack_class.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {output_path}")


def create_comparison_table():
    """Create and display comparison table."""
    df = parse_results()
    
    if df.empty:
        print("[WARN] No results found! Train models first.")
        return None
    
    # Sort by F1-Score
    df = df.sort_values('F1-Score', ascending=False)
    best_model = df.iloc[0]['Model']
    
    # Create markdown table
    md_table = "# Model Comparison Results\n\n"
    md_table += f"**Best Model: {best_model}** (highest F1-Score)\n\n"
    md_table += "| Model | Accuracy | Precision | Recall | F1-Score |\n"
    md_table += "|-------|----------|-----------|--------|----------|\n"
    
    for _, row in df.iterrows():
        star = " :star:" if row['Model'] == best_model else ""
        md_table += (f"| {row['Model']}{star} | "
                    f"{row['Accuracy']:.4f} | {row['Precision']:.4f} | "
                    f"{row['Recall']:.4f} | {row['F1-Score']:.4f} |\n")
    
    output_path = os.path.join(RESULTS_DIR, 'comparison_table.md')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_table)
    
    print("\n" + "=" * 70)
    print("MODEL COMPARISON TABLE")
    print("=" * 70)
    print(md_table)
    print(f"\n[Saved] {output_path}")
    
    return df


def main():
    print("\n" + "=" * 50)
    print("TV5: MODEL COMPARISON")
    print("=" * 50 + "\n")
    
    # 1. Create comparison table
    df = create_comparison_table()
    
    # 2. Generate all plots
    print("\nGenerating plots...")
    print("-" * 30)
    
    plot_all_confusion_matrices()
    plot_metrics_comparison(df)
    plot_recall_by_attack_class()
    
    # 3. Individual confusion matrices
    print("\nIndividual confusion matrices:")
    for model_name, filename in MODEL_FILES.items():
        plot_confusion_matrix_single(model_name, filename)
    
    print("\n" + "=" * 50)
    print("All visualizations complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
