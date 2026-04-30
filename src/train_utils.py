"""
train_utils.py - Shared utilities for all model training scripts.
Provides: load_data, train_and_evaluate, plot_confusion_matrix, save_model
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_score, recall_score, f1_score)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import sys

# Force utf-8 encoding for Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (SELECTED_FEATURES, RESULTS_DIR, REPORTS_DIR,
                        MODELS_DIR, TEST_SIZE, RANDOM_STATE)


def load_data(filepath, use_selected_features=True):
    """Load và chia train/test từ file CSV."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"[ERROR] File không tồn tại: {filepath}")

    df = pd.read_csv(filepath)

    if use_selected_features:
        missing = [f for f in SELECTED_FEATURES if f not in df.columns]
        if missing:
            raise ValueError(f"[ERROR] Thiếu features: {missing}")
        X = df[SELECTED_FEATURES]
    else:
        X = df.drop('Label', axis=1)

    if 'Label' not in df.columns:
        raise ValueError("[ERROR] Thiếu cột 'Label' trong dữ liệu!")

    y = df['Label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def train_and_evaluate(model, X_train, X_test, y_train, y_test, model_name):
    """Train model, tính metrics, lưu report và confusion matrix PNG."""
    # --- Train ---
    model.fit(X_train, y_train)
    print(f"[OK] {model_name} trained successfully!")

    # --- Predict ---
    y_pred = model.predict(X_test)

    # --- Metrics ---
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall    = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1        = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    report    = classification_report(y_test, y_pred, zero_division=0)
    cm        = confusion_matrix(y_test, y_pred)

    # --- Save report TXT ---
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(RESULTS_DIR, f"{model_name}_report.txt")
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write(f"=== {model_name} ===\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision (weighted): {precision:.4f}\n")
        f.write(f"Recall (weighted): {recall:.4f}\n")
        f.write(f"F1-Score (weighted): {f1:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm) + "\n")
        f.write("\nClassification Report:\n")
        f.write(report)
    print(f"[OK] Report saved: {results_path}")

    # --- Save confusion matrix PNG ---
    _plot_confusion_matrix(y_test, y_pred, model_name)

    return model, y_pred, report


def _plot_confusion_matrix(y_test, y_pred, model_name):
    """Vẽ và lưu confusion matrix PNG vào reports/."""
    cm = confusion_matrix(y_test, y_pred)
    labels = sorted(list(set(y_test) | set(y_pred)))

    # Nếu label là số nguyên, giữ nguyên; nếu là string thì dùng thẳng
    fig, ax = plt.subplots(figsize=(max(8, len(labels)), max(6, len(labels) - 2)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax,
                annot_kws={"size": 8})
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('Actual Label', fontsize=11)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    os.makedirs(REPORTS_DIR, exist_ok=True)
    png_path = os.path.join(REPORTS_DIR, f"{model_name}_confusion_matrix.png")
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Confusion matrix saved: {png_path}")


def save_model(model, model_name):
    """Lưu model thành file .pkl."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    filepath = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    joblib.dump(model, filepath)
    print(f"[OK] Model saved: {filepath}")
