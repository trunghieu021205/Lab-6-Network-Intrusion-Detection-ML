import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.train_utils import load_data, train_and_evaluate, save_model
from src.config import DATA_PROCESSED_DIR
from sklearn.naive_bayes import GaussianNB

def main():
    print("TV3: Training Naive Bayes")
    data_path = os.path.join(DATA_PROCESSED_DIR, 'final_data.csv')
    if not os.path.exists(data_path):
        data_path = os.path.join(DATA_PROCESSED_DIR, 'cleaned_data.csv')
        print("final_data.csv chua co, dung cleaned_data.csv tam")
    
    X_train, X_test, y_train, y_test = load_data(data_path)
    model = GaussianNB()
    model_trained, y_pred, report = train_and_evaluate(
        model, X_train, X_test, y_train, y_test, "NaiveBayes"
    )
    save_model(model_trained, "naive_bayes")
    print(report)

if __name__ == "__main__":
    main()