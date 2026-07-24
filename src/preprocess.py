import pandas as pd


# Features to use for the model
FEATURE_COLUMNS = [
    "loan_amnt",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "purpose",
    "dti",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "mort_acc",
    "pub_rec_bankruptcies",
    "application_type",
]


def preprocess_train(df):
    """
    Preprocess training data.
    """

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Keep only required loan statuses
    df = df[df["loan_status"].isin(["Fully Paid", "Charged Off"])].copy()

    # Target Encoding
    df["target"] = df["loan_status"].map(
        {
            "Fully Paid": 1,
            "Charged Off": 0,
        }
    )

    # Keep only required columns
    df = df[FEATURE_COLUMNS + ["target"]]

    # Fill missing numeric values
    numeric_cols = df.select_dtypes(include=["number"]).columns

    df[numeric_cols] = df[numeric_cols].fillna(
        df[numeric_cols].median()
    )

    # Fill missing categorical values
    categorical_cols = df.select_dtypes(include="object").columns

    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # One-Hot Encoding
    X = pd.get_dummies(
        df.drop(columns="target"),
        drop_first=True,
    )

    y = df["target"]

    return X, y


def preprocess_predict(df):
    """
    Preprocess prediction data.
    """

    numeric_cols = df.select_dtypes(include=["number"]).columns

    df[numeric_cols] = df[numeric_cols].fillna(
        df[numeric_cols].median()
    )

    categorical_cols = df.select_dtypes(include="object").columns

    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    df = pd.get_dummies(
        df,
        drop_first=True,
    )

    return df