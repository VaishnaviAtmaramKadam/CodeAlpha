import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
accuracy_score,
precision_score,
recall_score,
f1_score,
roc_auc_score,
classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# 1. Load Dataset

print("Loading Credit Scoring dataset...")

credit = fetch_openml(
name="credit-g",
version=1,
as_frame=True
)

X = credit.data.copy()
y = credit.target.astype(str)

# Good Credit = 1, Bad Credit = 0

y = (y == "good").astype(int)

print("Dataset loaded successfully!")
print("Dataset shape:", X.shape)

# 2. Identify Numerical and Categorical Columns

numeric_features = X.select_dtypes(
include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
exclude=["number"]
).columns.tolist()

# 3. Preprocess Numerical Data

numeric_transformer = Pipeline(steps=[
("imputer", SimpleImputer(strategy="median")),
("scaler", StandardScaler())
])

# 4. Preprocess Categorical Data

categorical_transformer = Pipeline(steps=[
("imputer", SimpleImputer(strategy="most_frequent")),
("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# 5. Combine Preprocessing

preprocessor = ColumnTransformer(transformers=[
("num", numeric_transformer, numeric_features),
("cat", categorical_transformer, categorical_features)
])

# 6. Split Dataset into Training and Testing Sets

X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.2,
random_state=42,
stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# 7. Define Machine Learning Models

models = {
"Logistic Regression": LogisticRegression(max_iter=2000),


"Decision Tree": DecisionTreeClassifier(
    max_depth=5,
    random_state=42
),

"Random Forest": RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)


}

# 8. Train and Evaluate All Models

results = []

print("Models found:", list(models.keys()))
print("\nTraining and evaluating models...\n")

for name, model in models.items():
    print("Currently training:", name)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

# Make predictions
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

# Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)

# Store results
    results.append({
        "Model": name,
        "Accuracy": round(accuracy, 3),
        "Precision": round(precision, 3),
        "Recall": round(recall, 3),
        "F1 Score": round(f1, 3),
        "ROC-AUC": round(roc_auc, 3)
    })

# Display individual model results
    print("=" * 50)
    print("Model:", name)
    print("=" * 50)

    print(classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Bad Credit", "Good Credit"],
        zero_division=0
    ))


# 9. Compare All Models

results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))

# 10. Save Results to CSV

results_df.to_csv("model_results.csv", index=False)

print("\nResults saved to model_results.csv")
print("All three models trained successfully!")
print("Project completed successfully!")
