
import pandas as pd
from feature_selection import select_features

df = pd.read_csv("data/dataset.csv")

# Remove duplicate records
df = df.drop_duplicates()

selected_features = select_features(df)

print("\nTotal selected features:", len(selected_features))
