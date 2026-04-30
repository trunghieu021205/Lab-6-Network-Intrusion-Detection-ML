# src/models/random_forest.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.train_utils import load_data, train_and_evaluate, save_model
from src.config import DATA_PROCESSED_DIR
from sklearn.ensemble import RandomForestClassifier

def main():
    print("\n" + "=" * 50)
    print("TV5: Training Random Forest (Best Model)")
    print("=" * 50)
    
    # Load data
    data_path = os.path.join(DATA_PROCESSED_DIR, 'final_data.csv')
    X_train, X_test, y_train, y_test = load_data(data_path)
    
    # Create model with optimized parameters
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    # Train and evaluate
    model_trained, y_pred, report = train_and_evaluate(
        model, X_train, X_test, y_train, y_test, "RandomForest"
    )
    
    # Save model
    save_model(model_trained, "random_forest")
    
    print("\n" + report)
    print("\n✅ TV5: Random Forest completed!")

if __name__ == "__main__":
    main()
