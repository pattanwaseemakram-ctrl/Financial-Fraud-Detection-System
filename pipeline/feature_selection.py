import pandas as pd
from sklearn.ensemble import RandomForestClassifier
try:
    df = pd.read_csv("Dataset/scaled_transactions.csv")

    print(df.head())

    print("\nDataset Information:")
    print(df.info())

    print("\nDataset Columns:")
    print(df.columns.tolist())


    # Separate features and target

    X = df.drop("Suspicious Activity Flag", axis=1)

    y = df["Suspicious Activity Flag"]

    print("\nFeatures (X):")
    print(X.columns.tolist())

    print("\nTarget (y):")
    print(y.value_counts())




    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)

    print("\nTraining Target Distribution:")
    print(y_train.value_counts())

    print("\nTesting Target Distribution:")
    print(y_test.value_counts())




    # Feature Selection using Random Forest model

    rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
    )

    # Train Random Forest only on training data 
    rf.fit(X_train, y_train)

    # Get feature importance
    feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": rf.feature_importances_
    })

    # Sort by importance
    feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
   )

    print("\nFeature Importance:")
    print(feature_importance)








    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 8))

    plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
    )

    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Random Forest Feature Importance")

    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.show()





    threshold = 0.01

    selected_features = feature_importance[
    feature_importance["Importance"] >= threshold
    ]["Feature"].tolist()

    print("\nSelected Features:")
    print(selected_features)

    print("\nNumber of Selected Features:")
    print(len(selected_features))





    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    print("Selected X_train shape:", X_train_selected.shape)
    print("Selected X_test shape:", X_test_selected.shape)










    # Save selected training and testing datasets

    X_train_selected.to_csv(
    "Dataset/X_train_selected.csv",
    index=False
  )

    X_test_selected.to_csv(
    "Dataset/X_test_selected.csv",
    index=False
    )

    y_train.to_csv(
    "Dataset/y_train.csv",
    index=False
   )

    y_test.to_csv(
    "Dataset/y_test.csv",
    index=False
)

    print("\nSelected datasets saved successfully.")






    X_train.to_csv("Dataset/X_train.csv", index=False)
    X_test.to_csv("Dataset/X_test.csv", index=False)

except FileNotFoundError:
    print("Error: Required input dataset was not found.")

except ValueError as e:
    print("Feature selection data error:", e)

except Exception as e:
    print("Unexpected error during feature selection:", e)