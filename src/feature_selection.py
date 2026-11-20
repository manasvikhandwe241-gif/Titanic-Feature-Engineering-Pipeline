
import pandas as pd


def select_features(df):
    """
    Select relevant features for Titanic survival analysis
    using correlation analysis and domain relevance.
    """

    # Numerical feature correlation with target
    numeric_features = ["Age", "Fare", "Pclass"]

    correlations = (
        df[numeric_features + ["Survived"]]
        .corr(numeric_only=True)["Survived"]
        .drop("Survived")
        .sort_values(key=abs, ascending=False)
    )

    print("\n--- CORRELATION WITH SURVIVAL ---")
    print(correlations)

    # Retain relevant numerical and categorical features
    selected_features = [
        "Age",
        "Fare",
        "Sex",
        "Pclass",
        "Embarked"
    ]

    print("\n--- SELECTED FEATURES ---")
    for feature in selected_features:
        print(feature)

    return selected_features
