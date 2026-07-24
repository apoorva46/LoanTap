import pandas as pd
import joblib

from preprocess import preprocess_train

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


# Load Dataset

df = pd.read_csv("data/LoanTap.csv")

print(f"Dataset Shape : {df.shape}")


# Preprocess Dataset

X, y = preprocess_train(df)

print(f"Feature Matrix Shape : {X.shape}")


# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y,
)


print(f"Train Shape : {X_train.shape}")
print(f"Test Shape  : {X_test.shape}")


# Feature Scaling

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# Model Training

model = LogisticRegression(
    max_iter=1000,
    random_state=42,
)

model.fit(
    X_train_scaled,
    y_train,
)


# Predictions

y_pred = model.predict(X_test_scaled)

y_prob = model.predict_proba(X_test_scaled)[:, 1]


# Evaluation

print("\nConfusion Matrix\n")

print(
    confusion_matrix(
        y_test,
        y_pred,
    )
)


print("\nClassification Report\n")

print(
    classification_report(
        y_test,
        y_pred,
    )
)


auc = roc_auc_score(
    y_test,
    y_prob,
)

print(f"\nROC-AUC Score : {auc:.4f}")


# Feature Importance

feature_importance = pd.DataFrame(
    {
        "Feature": X.columns,
        "Coefficient": model.coef_[0],
    }
)

feature_importance = feature_importance.sort_values(
    by="Coefficient",
    ascending=False,
)

print("\nTop 15 Important Features\n")

print(
    feature_importance.head(15)
)


# Save Artifacts

joblib.dump(
    model,
    "models/loan_default_model.pkl",
)

joblib.dump(
    scaler,
    "models/scaler.pkl",
)

joblib.dump(
    X.columns.tolist(),
    "models/feature_columns.pkl",
)

print("\nModel Saved Successfully")