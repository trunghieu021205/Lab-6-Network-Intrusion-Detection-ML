# src/eda_analysis.py
"""
Exploratory Data Analysis (EDA) for Network Intrusion Detection.
TV1: EDA - Distribution plots and Correlation heatmap.
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
from src.config import DATA_PROCESSED_DIR, DATA_RAW_DIR, SELECTED_FEATURES, REPORTS_DIR


def load_data_for_eda():
    """Load cleaned data for EDA."""
    filepath = os.path.join(DATA_PROCESSED_DIR, 'cleaned_data.csv')
    
    if not os.path.exists(filepath):
        # Fallback to raw data if cleaned doesn't exist
        print("[WARN] cleaned_data.csv not found, trying raw data...")
        raw_files = [f for f in os.listdir(DATA_RAW_DIR) if f.endswith('.csv')]
        if raw_files:
            filepath = os.path.join(DATA_RAW_DIR, raw_files[0])
            df = pd.read_csv(filepath)
            print(f"[INFO] Loaded raw data: {df.shape}")
            return df
        raise FileNotFoundError(
            f"Data not found at {filepath}. "
            "Please run preprocessing.py first."
        )
    
    df = pd.read_csv(filepath)
    print(f"[INFO] Loaded data: {df.shape}")
    return df


def plot_label_distribution(df, save_path):
    """Plot distribution of attack types (Label column)."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    if 'Label' not in df.columns:
        print("[SKIP] No Label column found")
        return
    
    label_counts = df['Label'].value_counts()
    
    # Handle numeric labels (if already encoded)
    if label_counts.index.dtype in ['int64', 'float64'] or all(isinstance(x, (int, float)) for x in label_counts.index[:5]):
        # Try to map back if possible, otherwise use numeric
        label_map = {}
        for val in label_counts.index:
            label_map[val] = f"Class_{val}"
    else:
        label_map = None
    
    plt.figure(figsize=(14, 8))
    
    # Create bar plot
    colors = plt.cm.Set3(np.linspace(0, 1, len(label_counts)))
    
    ax = label_counts.plot(kind='bar', color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Attack Type', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Attack Types (CICIDS2017)', fontsize=14, fontweight='bold')
    
    # Add count labels on bars
    for i, (idx, val) in enumerate(label_counts.items()):
        ax.text(i, val + label_counts.max() * 0.01, f'{val:,}', 
               ha='center', va='bottom', fontsize=8, rotation=45)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {save_path}")
    
    # Also print summary
    print("\n" + "=" * 50)
    print("LABEL DISTRIBUTION SUMMARY")
    print("=" * 50)
    print(f"Total samples: {label_counts.sum():,}")
    print(f"Number of classes: {len(label_counts)}")
    print("\nTop 5 most common:")
    print(label_counts.head())
    print("\nClass percentages:")
    print((label_counts / label_counts.sum() * 100).round(2))


def plot_correlation_heatmap(df, save_path):
    """Plot correlation heatmap for selected features."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    # Select only numeric features
    feature_cols = [c for c in SELECTED_FEATURES if c in df.columns]
    if not feature_cols:
        feature_cols = [c for c in df.columns if c != 'Label' and df[c].dtype in ['int64', 'float64', 'int32', 'float32']]
    
    if len(feature_cols) < 2:
        print("[SKIP] Not enough numeric features for correlation")
        return
    
    # Sample if too large (for performance)
    if len(df) > 5000:
        df_sample = df[feature_cols].sample(n=5000, random_state=42)
    else:
        df_sample = df[feature_cols]
    
    # Calculate correlation matrix
    corr_matrix = df_sample.corr()
    
    plt.figure(figsize=(16, 14))
    
    # Create heatmap with annotations
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Upper triangle mask
    
    ax = sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        square=True,
        linewidths=0.5,
        annot_kws={'size': 7},
        cbar_kws={'shrink': 0.8, 'label': 'Correlation Coefficient'},
        vmin=-1, vmax=1
    )
    
    ax.set_title('Feature Correlation Heatmap\n(Top: Highly Correlated, Bottom: Uncorrelated)',
                fontsize=14, fontweight='bold', pad=20)
    
    # Rotate labels
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {save_path}")
    
    # Print top correlated pairs
    print("\n" + "=" * 50)
    print("TOP 10 HIGHLY CORRELATED FEATURE PAIRS")
    print("=" * 50)
    
    # Get upper triangle indices
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_pairs.append({
                'Feature 1': corr_matrix.columns[i],
                'Feature 2': corr_matrix.columns[j],
                'Correlation': corr_matrix.iloc[i, j]
            })
    
    corr_df = pd.DataFrame(corr_pairs)
    corr_df['Abs_Corr'] = corr_df['Correlation'].abs()
    corr_df = corr_df.sort_values('Abs_Corr', ascending=False)
    
    print("\nTop positive correlations:")
    print(corr_df[corr_df['Correlation'] > 0.5][['Feature 1', 'Feature 2', 'Correlation']].head(5).to_string(index=False))
    
    print("\nTop negative correlations:")
    print(corr_df[corr_df['Correlation'] < -0.3][['Feature 1', 'Feature 2', 'Correlation']].head(5).to_string(index=False))


def plot_feature_distributions(df, save_path, top_n=6):
    """Plot distribution of top features."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != 'Label']
    
    if len(numeric_cols) < 1:
        print("[SKIP] No numeric features found")
        return
    
    # Select top features by variance
    variances = df[numeric_cols].var().sort_values(ascending=False)
    top_features = variances.head(top_n).index.tolist()
    
    n_cols = 3
    n_rows = (len(top_features) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if len(top_features) > 1 else [axes]
    
    for i, feature in enumerate(top_features):
        ax = axes[i]
        
        # Check if label exists for coloring
        if 'Label' in df.columns:
            # Plot by label (Benign vs Attack)
            benign = df[df['Label'] == df['Label'].iloc[0]][feature]
            attack = df[df['Label'] != df['Label'].iloc[0]][feature]
            
            ax.hist(benign, bins=50, alpha=0.5, label='Benign', color='green', density=True)
            ax.hist(attack, bins=50, alpha=0.5, label='Attack', color='red', density=True)
            ax.legend()
        else:
            ax.hist(df[feature], bins=50, color='steelblue', alpha=0.7, density=True)
        
        ax.set_xlabel(feature, fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'Distribution: {feature}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(len(top_features), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Feature Distributions (Benign vs Attack)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {save_path}")


def plot_imbalance_analysis(df, save_path):
    """Visualize class imbalance before balancing."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    if 'Label' not in df.columns:
        print("[SKIP] No Label column")
        return
    
    label_counts = df['Label'].value_counts()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Pie chart
    ax1 = axes[0]
    colors = plt.cm.Set2(np.linspace(0, 1, len(label_counts)))
    
    # Group small classes as "Other"
    threshold = 0.02  # 2%
    small_classes = label_counts[label_counts / label_counts.sum() < threshold]
    large_classes = label_counts[label_counts / label_counts.sum() >= threshold]
    
    if len(small_classes) > 0:
        other_sum = small_classes.sum()
        plot_data = large_classes.copy()
        if other_sum > 0:
            plot_data['Other'] = other_sum
    else:
        plot_data = label_counts
    
    wedges, texts, autotexts = ax1.pie(
        plot_data, 
        autopct=lambda p: f'{p:.1f}%' if p > 3 else '',
        startangle=90,
        colors=plt.cm.Set3(np.linspace(0, 1, len(plot_data)))
    )
    ax1.set_title('Class Distribution (Pie Chart)', fontsize=12, fontweight='bold')
    ax1.legend(plot_data.index, title='Classes', loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    
    # Bar chart with log scale
    ax2 = axes[1]
    label_counts.plot(kind='bar', ax=ax2, color='steelblue', edgecolor='black', linewidth=0.5)
    ax2.set_yscale('log')
    ax2.set_xlabel('Attack Type', fontsize=10)
    ax2.set_ylabel('Count (log scale)', fontsize=10)
    ax2.set_title('Class Distribution (Log Scale - Shows Imbalance)', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add imbalance ratio annotation
    max_count = label_counts.max()
    min_count = label_counts.min()
    ratio = max_count / min_count if min_count > 0 else float('inf')
    
    ax2.annotate(
        f'Imbalance Ratio: {ratio:.1f}:1',
        xy=(0.95, 0.95), xycoords='axes fraction',
        fontsize=10, ha='right', va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {save_path}")


def main():
    print("\n" + "=" * 50)
    print("TV1: EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 50 + "\n")
    
    # Load data
    df = load_data_for_eda()
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    # Generate all EDA plots
    print("\nGenerating EDA visualizations...")
    print("-" * 40)
    
    # 1. Label distribution
    plot_label_distribution(
        df, 
        os.path.join(REPORTS_DIR, 'eda_label_distribution.png')
    )
    
    # 2. Correlation heatmap
    plot_correlation_heatmap(
        df,
        os.path.join(REPORTS_DIR, 'eda_correlation_heatmap.png')
    )
    
    # 3. Feature distributions
    plot_feature_distributions(
        df,
        os.path.join(REPORTS_DIR, 'eda_feature_distributions.png')
    )
    
    # 4. Class imbalance analysis
    plot_imbalance_analysis(
        df,
        os.path.join(REPORTS_DIR, 'eda_class_imbalance.png')
    )
    
    print("\n" + "=" * 50)
    print("EDA Complete! Check /reports/ folder for plots.")
    print("=" * 50)
    print("\nGenerated files:")
    print("  - eda_label_distribution.png")
    print("  - eda_correlation_heatmap.png")
    print("  - eda_feature_distributions.png")
    print("  - eda_class_imbalance.png")


if __name__ == "__main__":
    main()
