import pandas as pd
from sklearn.datasets import make_classification

def create_dummy_data(
    n_samples=5000,
    n_features=18,
    n_informative=12,
    n_redundant=3,
    n_classes=2,
    weights=[0.85, 0.15],
    random_state=42,
    output_path="data/dummy/dummy_data.csv"
):
    # 1. Generate synthetic dataset
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=n_classes,
        weights=weights,
        random_state=random_state
    )

    # 2. Create feature names: f0 -> f17
    feature_cols = [f"f{i}" for i in range(n_features)]

    # 3. Build DataFrame
    df = pd.DataFrame(X, columns=feature_cols)
    df["label"] = y

    # 4. Save to CSV
    df.to_csv(output_path, index=False)

    print("✅ Dummy data created successfully!")
    print(f"Shape: {df.shape}")
    print(f"Saved to: {output_path}")
    print(df.head())

if __name__ == "__main__":
    create_dummy_data()