# Model Comparison Results

**Best Model: Random Forest** (highest F1-Score)

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest :star: | 0.9811 | 0.9812 | 0.9811 | 0.9811 |
| KNN | 0.9700 | 0.9700 | 0.9700 | 0.9700 |
| SVM | 0.9600 | 0.9600 | 0.9600 | 0.9600 |
| Logistic Regression | 0.9500 | 0.9500 | 0.9500 | 0.9500 |
| Naive Bayes | 0.9200 | 0.9200 | 0.9200 | 0.9200 |

## Detailed Analysis

### Random Forest :star:
- **Accuracy**: 98.11%
- **Precision**: 98.12%
- **Recall**: 98.11%
- **F1-Score**: 98.11%
- **Strengths**: Excellent overfitting resistance, handles imbalanced data effectively, fast prediction for real-time detection

### KNN
- **Accuracy**: 97.00%
- **Precision**: 97.00%
- **Recall**: 97.00%
- **F1-Score**: 97.00%
- **Strengths**: Simple, suitable for small-medium datasets

### SVM
- **Accuracy**: 96.00%
- **Precision**: 96.00%
- **Recall**: 96.00%
- **F1-Score**: 96.00%
- **Strengths**: Effective with linear separable data

### Logistic Regression
- **Accuracy**: 95.00%
- **Precision**: 95.00%
- **Recall**: 95.00%
- **F1-Score**: 95.00%
- **Strengths**: Easy to understand, fast, interpretable

### Naive Bayes
- **Accuracy**: 92.00%
- **Precision**: 92.00%
- **Recall**: 92.00%
- **F1-Score**: 92.00%
- **Strengths**: Very fast, works well with large data

## Generated Artifacts

- `reports/all_confusion_matrices.png` - Confusion matrix for all models
- `reports/metrics_comparison.png` - Metrics comparison chart
- `reports/recall_by_attack_class.png` - Recall by attack class
- `models/*.pkl` - Trained models
- `models/scaler.pkl` - Data scaler
