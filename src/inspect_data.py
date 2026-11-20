
import pandas as pd

# Load the Titanic dataset
df = pd.read_csv("data/dataset.csv")

# Display first five rows
print("\n--- FIRST FIVE ROWS ---")
print(df.head())

# Dataset dimensions
print("\n--- DATASET SHAPE ---")
print(df.shape)

# Column information
print("\n--- DATASET INFORMATION ---")
df.info()

# Missing values
print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

# Duplicate records
print("\n--- DUPLICATE RECORDS ---")
print(df.duplicated().sum())

# Data types
print("\n--- DATA TYPES ---")
print(df.dtypes)
