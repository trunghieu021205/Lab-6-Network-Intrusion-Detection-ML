# src/train_utils.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.config import *

def load_data(filepath, use_selected_features=True):
    """Load dữ liệu đã xử lý và chia train/test"""
    df = pd.read_csv(filepath)
    
    # Tách features và label
    if use_selected_features:
        X = df[SELECTED_FEATURES]
    else:
        X = df.drop('Label', axis=1)
    
    y = df['Label']
    
    # Chia train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def train_and_evaluate(model, X_train, X_test, y_train, y_test, model_name):
    """Train model và trả về kết quả đánh giá"""
    # Train
    model.fit(X_train, y_train)
    print(f"✅ {model_name} trained successfully!")
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Classification report
    report = classification_report(y_test, y_pred)
    
    # Save results
    results_path = os.path.join(RESULTS_DIR, f"{model_name}_report.txt")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(results_path, 'w') as f:
        f.write(f"=== {model_name} ===\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision (weighted): {precision:.4f}\n")
        f.write(f"Recall (weighted): {recall:.4f}\n")
        f.write(f"F1-Score (weighted): {f1:.4f}\n")
        f.write("\n" + report)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, model_name)
    
    return model, y_pred, report

def plot_confusion_matrix(y_test, y_pred, model_name):
    """Vẽ và lưu confusion matrix"""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Save
    os.makedirs(REPORTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(REPORTS_DIR, f"{model_name}_confusion_matrix.png"))
    plt.close()
    print(f"✅ Confusion matrix saved for {model_name}")

def save_model(model, model_name):
    """Lưu model thành file .pkl"""
    os.makedirs(MODELS_DIR, exist_ok=True)
    filepath = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    joblib.dump(model, filepath)
    print(f"✅ Model saved to {filepath}")
