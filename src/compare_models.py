"""
TV5: Model Comparison - So sanh tat ca 5 models
Network Intrusion Detection System

Script nay doc ket qua tu cac file report, neu chua co thi tu dong train.
Tao bang so sanh Markdown.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.config import DATA_PROCESSED_DIR, RESULTS_DIR, MODELS_DIR, SELECTED_FEATURES, RANDOM_STATE
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import joblib


def load_final_data():
    """Load du lieu da can bang"""
    filepath = os.path.join(DATA_PROCESSED_DIR, 'final_data.csv')
    df = pd.read_csv(filepath)
    X = df[SELECTED_FEATURES]
    y = df['Label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Data loaded: train={len(X_train)}, test={len(X_test)}")
    return X_train, X_test, y_train, y_test


def train_model_if_needed(model_key, model_name, X_train, X_test, y_train, y_test):
    """Train model neu chua co file ket qua"""
    report_file = os.path.join(RESULTS_DIR, f"{model_key}_report.txt")

    if os.path.exists(report_file) and os.path.getsize(report_file) > 0:
        print(f"[SKIP] {model_name}: report da ton tai")
        return load_existing_results(report_file)

    print(f"[TRAIN] {model_name}: chua co ket qua, bat dau train...")

    if model_key == 'LogisticRegression':
        model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1)
    elif model_key == 'NaiveBayes':
        model = GaussianNB()
    elif model_key == 'KNN':
        model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    elif model_key == 'SVM':
        # Chi dung 20% data cho SVM de tranh treo may
        df = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, 'final_data.csv'))
        df_sample = df.sample(frac=0.2, random_state=RANDOM_STATE)
        X_s = df_sample[SELECTED_FEATURES]
        y_s = df_sample['Label']
        X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
            X_s, y_s, test_size=0.2, random_state=RANDOM_STATE, stratify=y_s
        )
        model = SVC(kernel='linear', random_state=RANDOM_STATE)
        model.fit(X_train_s, y_train_s)
        y_pred = model.predict(X_test_s)
        save_results(model_key, model_name, X_test_s, y_test_s, y_pred)
        joblib.dump(model, os.path.join(MODELS_DIR, f"{model_key.lower().replace(' ', '_')}.pkl"))
        print(f"[OK] {model_name} trained (sample 20%)")
        return load_existing_results(report_file)
    elif model_key == 'RandomForest':
        model = RandomForestClassifier(
            n_estimators=100, max_depth=20, min_samples_split=5,
            min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1
        )
    else:
        return None

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    save_results(model_key, model_name, X_test, y_test, y_pred)
    joblib.dump(model, os.path.join(MODELS_DIR, f"{model_key.lower().replace(' ', '_')}.pkl"))
    print(f"[OK] {model_name} trained")
    return load_existing_results(report_file)


def save_results(model_key, model_name, X_test, y_test, y_pred):
    """Luu ket qua danh gia model"""
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    report = classification_report(y_test, y_pred, zero_division=0)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    report_file = os.path.join(RESULTS_DIR, f"{model_key}_report.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"=== {model_name} ===\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision (weighted): {precision:.4f}\n")
        f.write(f"Recall (weighted): {recall:.4f}\n")
        f.write(f"F1-Score (weighted): {f1:.4f}\n")
        f.write("\n" + report)

    print(f"[OK] Saved: {report_file}")


def load_existing_results(filepath):
    """Doc ket qua tu file report"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        result = {'file': os.path.basename(filepath)}
        for line in content.split('\n'):
            if 'Accuracy:' in line:
                result['Accuracy'] = float(line.split(':')[1].strip())
            elif 'Precision (weighted):' in line:
                result['Precision'] = float(line.split(':')[1].strip())
            elif 'Recall (weighted):' in line:
                result['Recall'] = float(line.split(':')[1].strip())
            elif 'F1-Score (weighted):' in line:
                result['F1'] = float(line.split(':')[1].strip())
        return result
    except:
        return None


def create_comparison_table(all_results):
    """Tao bang so sanh Markdown"""
    md = "# Model Comparison Results\n\n"
    md += "**Network Intrusion Detection System - CIC-IDS2017 Dataset**\n\n"
    md += "| # | Model | Accuracy | Precision | Recall | F1-Score | Rank |\n"
    md += "|---|-------|----------|-----------|--------|----------|------|\n"

    df = pd.DataFrame(all_results)
    df = df.sort_values('F1', ascending=False).reset_index(drop=True)

    ranks = df['F1'].rank(ascending=False, method='min').astype(int)
    medal = {1: ' [BEST]', 2: ' [2nd]', 3: ' [3rd]'}

    for i, row in df.iterrows():
        rank = ranks[i]
        star = medal.get(rank, '')
        md += (f"| {i+1} | {row['model']}{star} | "
               f"{row['Accuracy']:.4f} | {row['Precision']:.4f} | "
               f"{row['Recall']:.4f} | {row['F1']:.4f} | #{rank} |\n")

    md += "\n---\n"
    md += "*Generated by TV5 - Random Forest + Realtime Team*\n"

    best = df.iloc[0]
    md += f"\n**Ket luan:** Model tot nhat la **{best['model']}** voi F1-Score = {best['F1']:.4f}\n"

    output_path = os.path.join(RESULTS_DIR, 'comparison_table.md')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md)

    print(f"\n[OK] Bang so sanh da luu: {output_path}")
    print("\n" + md)
    return df


def main():
    print("=" * 60)
    print("TV5: Model Comparison - Training Missing Models")
    print("=" * 60)

    X_train, X_test, y_train, y_test = load_final_data()

    models = [
        ('LogisticRegression', 'Logistic Regression (TV1)'),
        ('NaiveBayes', 'Naive Bayes (TV3)'),
        ('KNN', 'K-Nearest Neighbors (TV4)'),
        ('SVM', 'SVM - Linear Kernel (TV4)'),
        ('RandomForest', 'Random Forest (TV5)'),
    ]

    all_results = []
    for model_key, model_name in models:
        print(f"\n--- {model_name} ---")
        result = train_model_if_needed(model_key, model_name, X_train, X_test, y_train, y_test)
        if result:
            result['model'] = model_name
            all_results.append(result)

    print("\n" + "=" * 60)
    print("TAO BANG SO SANH")
    print("=" * 60)

    create_comparison_table(all_results)


if __name__ == "__main__":
    main()
