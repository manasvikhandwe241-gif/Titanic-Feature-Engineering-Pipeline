
import pandas as pd
import joblib

from preprocessing import build_preprocessor
from feature_selection import select_features


# 1. Load dataset
df = pd.read_csv("data/dataset.csv")

print("Original Dataset Shape:", df.shape)

# 2. Remove duplicate records
df = df.drop_duplicates()

# 3. Separate target variable
y = df["Survived"]

# 4. Perform feature selection
selected_features = select_features(df)

# 5. Keep only selected features
X = df[selected_features]

# 6. Build preprocessing pipeline
preprocessor = build_preprocessor()

# 7. Fit and transform the data
X_transformed = preprocessor.fit_transform(X)

# 8. Get transformed feature names
feature_names = preprocessor.get_feature_names_out()

# 9. Convert transformed data into DataFrame
transformed_df = pd.DataFrame(
    X_transformed,
    columns=feature_names
)

# 10. Display results
print("\n--- PIPELINE RESULTS ---")
print("Original Shape:", df.shape)
print("Selected Features:", selected_features)
print("Transformed Shape:", transformed_df.shape)

print("\n--- FIRST FIVE TRANSFORMED ROWS ---")
print(transformed_df.head())

# 11. Save transformed dataset
transformed_df.to_csv(
    "output/transformed_data.csv",
    index=False
)

# 12. Save fitted preprocessing pipeline
joblib.dump(
    preprocessor,
    "models/feature_pipeline.pkl"
)

print("\nPreprocessing Pipeline Saved Successfully!")
print("Transformed Dataset Saved Successfully!")
