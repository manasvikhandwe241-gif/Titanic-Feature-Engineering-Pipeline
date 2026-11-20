
import pandas as pd
import joblib

# 1. Load the saved preprocessing pipeline
preprocessor = joblib.load("models/feature_pipeline.pkl")

print("Saved Pipeline Loaded Successfully!")

# 2. Create new sample customer/passenger data
new_data = pd.DataFrame({
    "Age": [25, None, 40],
    "Fare": [15.50, 32.00, 80.00],
    "Sex": ["male", "female", "female"],
    "Pclass": [3, 2, 1],
    "Embarked": ["S", "C", "Q"]
})

print("\n--- NEW RAW DATA ---")
print(new_data)

# 3. Transform new data using the loaded pipeline
new_transformed = preprocessor.transform(new_data)

# 4. Get feature names
feature_names = preprocessor.get_feature_names_out()

# 5. Display transformed data
result = pd.DataFrame(
    new_transformed,
    columns=feature_names
)

print("\n--- TRANSFORMED NEW DATA ---")
print(result)

print("\nNew Data Shape:", result.shape)
print("\nInference Transformation Successful!")
