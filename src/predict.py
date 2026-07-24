import pandas as pd
import joblib

from .preprocess import preprocess_predict


# Load Saved Artifacts

model = joblib.load("models/loan_default_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")


def predict_loan(input_df):
    """
    Predict loan repayment status.
    """

    # Preprocess Input

    input_df = preprocess_predict(input_df)

    # Match Training Features

    input_df = input_df.reindex(
        columns=feature_columns,
        fill_value=0,
    )

    # Scale Features

    input_scaled = scaler.transform(input_df)

    # Prediction

    prediction = model.predict(input_scaled)[0]

    probability = model.predict_proba(
        input_scaled
    )[0][1]

    result = (
        "Fully Paid"
        if prediction == 1
        else "Charged Off"
    )

    return result, probability



if __name__ == "__main__":

    sample = pd.DataFrame(
        {
            "loan_amnt": [10000],
            "term": ["36 months"],
            "int_rate": [11.5],
            "installment": [330],
            "grade": ["B"],
            "sub_grade": ["B2"],
            "emp_length": ["10+ years"],
            "home_ownership": ["MORTGAGE"],
            "annual_inc": [75000],
            "verification_status": ["Verified"],
            "purpose": ["debt_consolidation"],
            "dti": [18.5],
            "open_acc": [12],
            "pub_rec": [0],
            "revol_bal": [12000],
            "revol_util": [45.3],
            "total_acc": [24],
            "mort_acc": [1],
            "pub_rec_bankruptcies": [0],
            "application_type": ["Individual"],
        }
    )

    result, probability = predict_loan(sample)

    print(result)
    print(probability)