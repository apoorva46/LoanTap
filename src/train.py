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
# Models
# ==========================

def get_models():
    """
    Returns all machine learning models used for training.
    """

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            random_state=42,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42,
        ),
        "XGBoost": XGBClassifier(
            random_state=42,
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

def evaluate_model(
    model,
    model_name,
    X_train,
    X_test,
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test,
):
    """
    Train and evaluate a machine learning model.
    """

    print("=" * 70)
    print(f"Training : {model_name}")

    # Scale only Logistic Regression
    if model_name == "Logistic Regression":
        train_data = X_train_scaled
        test_data = X_test_scaled
    else:
        train_data = X_train
        test_data = X_test
    print(type(train_data))
    print(type(y_train))


    # Train model
    model.fit(train_data, y_train)

    # Predictions
    y_pred = model.predict(test_data)
    y_prob = model.predict_proba(test_data)[:, 1]

    # Metrics
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

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": auc,
    }

    return model, metrics, auc

# ==========================
# Save Best Model
# ==========================

def save_best_model(model, scaler, feature_columns):
    """
    Save trained model and preprocessing artifacts.
    """

    joblib.dump(
        model,
        "models/loan_default_model.pkl",
    )

    joblib.dump(
        scaler,
        "models/scaler.pkl",
    )

    joblib.dump(
        feature_columns,
        "models/feature_columns.pkl",
    )

    print("\n" + "=" * 70)
    print("Best Model Saved Successfully!")
    print("=" * 70)

# ==========================
# Main Function
# ==========================

def main():

    # ==========================
    # Load Dataset
    # ==========================

    df = pd.read_csv("data/LoanTap.csv")

    print(f"Dataset Shape : {df.shape}")

    # ==========================
    # Preprocess Dataset
    # ==========================

    X, y = preprocess_train(df)

    print(X.select_dtypes(include=["datetime64[ns]"]).columns.tolist())

    print(f"Feature Matrix Shape : {X.shape}")

    print("\nColumn Names:\n")
    print(X.columns.tolist())

    

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

    print("\nX_train dtypes:")
    print(X_train.dtypes.value_counts())

    print("\nObject columns:")
    print(X_train.select_dtypes(include=["object"]).columns.tolist())

    print("\nDatetime columns:")
    print(X_train.select_dtypes(include=["datetime64[ns]"]).columns.tolist())

    # ==========================
    # Feature Scaling
    # ==========================

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ==========================
    # Models
    # ==========================

    models = get_models()

    results = []

    best_auc = 0
    best_model = None
    best_model_name = ""

    # ==========================
    # Train & Evaluate
    # ==========================

    for name, model in models.items():

        trained_model, metrics, auc = evaluate_model(
            model=model,
            model_name=name,
            X_train=X_train,
            X_test=X_test,
            X_train_scaled=X_train_scaled,
            X_test_scaled=X_test_scaled,
            y_train=y_train,
            y_test=y_test,
        )

        results.append(metrics)

        if auc > best_auc:
            best_auc = auc
            best_model = trained_model
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

    print(f"\nBest Model : {best_model_name}")
    print(f"Best ROC-AUC : {best_auc:.4f}")

    save_best_model(
        best_model,
        scaler,
        X.columns.tolist(),
    )

if __name__ == "__main__":
    main()
