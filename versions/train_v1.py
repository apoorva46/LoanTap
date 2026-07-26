import pandas as pd
import joblib

from preprocess import preprocess_train

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# ==========================
# Load Dataset
# ==========================

df = pd.read_csv("data/LoanTap.csv")

print(f"Dataset Shape : {df.shape}")

# ==========================
# Preprocess Dataset
# ==========================

X, y = preprocess_train(df)

print(f"Feature Matrix Shape : {X.shape}")

# ==========================
# Train Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y,
)

print(f"Train Shape : {X_train.shape}")
print(f"Test Shape  : {X_test.shape}")

# ==========================
# Feature Scaling
# (We'll improve this later)
# ==========================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================
# Models
# ==========================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42,
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42,
    ),

    "XGBoost": XGBClassifier(
        random_state=42,
        eval_metric="logloss",
    ),

    "LightGBM": LGBMClassifier(
        random_state=42,
    ),

    "CatBoost": CatBoostClassifier(
        verbose=0,
        random_state=42,
    ),
}

# ==========================
# Train & Evaluate
# ==========================

results = []

best_auc = 0
best_model = None
best_model_name = ""

for name, model in models.items():

    print("=" * 70)
    print(f"Training : {name}")

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(y_test, y_pred)

    recall = recall_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred)

    auc = roc_auc_score(y_test, y_prob)

    print("\nConfusion Matrix\n")

    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report\n")

    print(classification_report(y_test, y_pred))

    print(f"ROC-AUC : {auc:.4f}")

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": auc,
    })

    if auc > best_auc:

        best_auc = auc
        best_model = model
        best_model_name = name

# ==========================
# Comparison Table
# ==========================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="ROC-AUC",
    ascending=False,
)

print("\n")
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(results_df)

# ==========================
# Save Best Model
# ==========================

print("\nBest Model :", best_model_name)
print(f"Best ROC-AUC : {best_auc:.4f}")

joblib.dump(
    best_model,
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

print("\nBest Model Saved Successfully!")