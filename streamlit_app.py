import streamlit as st
import pandas as pd

from src.predict import predict_loan

st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="🏦",
    layout="wide",
)

st.title("🏦 Loan Default Prediction System")

st.markdown(
    """
Predict whether a loan applicant is likely to **Fully Pay** the loan
or become **Charged Off** using a Logistic Regression model.
"""
)

st.divider()

st.subheader("Applicant Information")

left, right = st.columns(2)

with left:

    loan_amnt = st.number_input(
        "Loan Amount ($)",
        min_value=500,
        value=10000,
        step=500,
    )

    term = st.selectbox(
        "Loan Term",
        [
            "36 months",
            "60 months",
        ],
    )

    int_rate = st.number_input(
        "Interest Rate (%)",
        value=12.5,
    )

    installment = st.number_input(
        "Monthly Installment",
        value=350.0,
    )

    grade = st.selectbox(
        "Grade",
        [
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
        ],
    )

    sub_grade = st.selectbox(
        "Sub Grade",
        [
            "A1","A2","A3","A4","A5",
            "B1","B2","B3","B4","B5",
            "C1","C2","C3","C4","C5",
            "D1","D2","D3","D4","D5",
            "E1","E2","E3","E4","E5",
            "F1","F2","F3","F4","F5",
            "G1","G2","G3","G4","G5",
        ],
    )

    emp_length = st.selectbox(
        "Employment Length",
        [
            "10+ years",
            "9 years",
            "8 years",
            "7 years",
            "6 years",
            "5 years",
            "4 years",
            "3 years",
            "2 years",
            "1 year",
            "< 1 year",
        ],
    )

    home_ownership = st.selectbox(
        "Home Ownership",
        [
            "RENT",
            "MORTGAGE",
            "OWN",
            "OTHER",
            "NONE",
            "ANY",
        ],
    )

    annual_inc = st.number_input(
        "Annual Income ($)",
        value=60000,
    )

    verification_status = st.selectbox(
        "Verification Status",
        [
            "Not Verified",
            "Source Verified",
            "Verified",
        ],
    )

with right:

    purpose = st.selectbox(
        "Loan Purpose",
        [
            "vacation",
            "debt_consolidation",
            "credit_card",
            "home_improvement",
            "small_business",
            "major_purchase",
            "other",
            "medical",
            "wedding",
            "car",
            "moving",
            "house",
            "educational",
            "renewable_energy",
        ],
    )

    application_type = st.selectbox(
        "Application Type",
        [
            "INDIVIDUAL",
            "JOINT",
            "DIRECT_PAY",
        ],
    )

    dti = st.number_input(
        "Debt-To-Income Ratio",
        value=15.0,
    )

    open_acc = st.number_input(
        "Open Accounts",
        value=10,
    )

    pub_rec = st.number_input(
        "Public Records",
        value=0,
    )

    revol_bal = st.number_input(
        "Revolving Balance",
        value=15000,
    )

    revol_util = st.number_input(
        "Revolving Utilization (%)",
        value=45.0,
    )

    total_acc = st.number_input(
        "Total Accounts",
        value=20,
    )

    mort_acc = st.number_input(
        "Mortgage Accounts",
        value=1,
    )

    pub_rec_bankruptcies = st.number_input(
        "Public Record Bankruptcies",
        value=0,
    )

st.divider()

col1, col2, col3 = st.columns([2, 1, 2])

with col2:

    predict = st.button(
        "🔍 Predict Loan",
        use_container_width=True,
    )


if predict:

    input_df = pd.DataFrame(
        {
            "loan_amnt": [loan_amnt],
            "term": [term],
            "int_rate": [int_rate],
            "installment": [installment],
            "grade": [grade],
            "sub_grade": [sub_grade],
            "emp_length": [emp_length],
            "home_ownership": [home_ownership],
            "annual_inc": [annual_inc],
            "verification_status": [verification_status],
            "purpose": [purpose],
            "dti": [dti],
            "open_acc": [open_acc],
            "pub_rec": [pub_rec],
            "revol_bal": [revol_bal],
            "revol_util": [revol_util],
            "total_acc": [total_acc],
            "mort_acc": [mort_acc],
            "pub_rec_bankruptcies": [pub_rec_bankruptcies],
            "application_type": [application_type],
        }
    )

    prediction, probability = predict_loan(input_df)

    st.divider()

    st.subheader("Prediction Result")

    if prediction == "Fully Paid":

        st.success(
            "✅ This applicant is likely to Fully Pay the loan."
        )

    else:

        st.error(
            "❌ This applicant is likely to be Charged Off."
        )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Repayment Probability",
            f"{probability * 100:.2f}%",
        )

    with col2:

        st.metric(
            "Default Probability",
            f"{(1 - probability) * 100:.2f}%",
        )

    st.divider()

    with st.expander("Applicant Details", expanded=False):

        st.dataframe(
            input_df,
            use_container_width=True,
        )    