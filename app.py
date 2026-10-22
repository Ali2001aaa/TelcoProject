"""Streamlit app: predict churn risk for one (fictional) telecom customer."""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import sklearn

import churn_utils  # noqa: F401  (needed so joblib can find add_features inside the saved pipeline)

MODEL_PATH = Path(__file__).parent / "models" / "final_model.joblib"

st.set_page_config(page_title="Telco Churn Risk", page_icon="📞")


@st.cache_resource
def load_artifact():
    """Load the saved pipeline, threshold and input schema."""
    return joblib.load(MODEL_PATH)


def build_input_row(values, input_columns):
    """Return a one-row DataFrame with the same columns (and order) as the training data."""
    return pd.DataFrame([values])[input_columns]


def validate_inputs(values, categories):
    """Return a list of error messages (empty list = inputs are valid)."""
    errors = []
    if values["MonthlyCharges"] <= 0:
        errors.append("Monthly charges must be greater than 0.")
    if values["TotalCharges"] < 0:
        errors.append("Total charges cannot be negative.")
    if values["tenure"] == 0 and values["TotalCharges"] > 0:
        errors.append("A customer with 0 months of tenure should have total charges of 0.")
    for column, allowed in categories.items():
        if values[column] not in allowed:
            errors.append(f"Unexpected value for {column}: {values[column]}")
    return errors


st.title("📞 Telco Customer Churn Risk")
st.write(
    "Enter customer details to estimate the risk of leaving. "
    "**Educational demo** trained on a fictional IBM sample dataset, not for real customer decisions."
)

try:
    artifact = load_artifact()
except FileNotFoundError:
    st.error("Model file not found. Run notebooks/model_development.ipynb first to create models/final_model.joblib.")
    st.stop()

if artifact.get("sklearn_version") != sklearn.__version__:
    st.warning(
        f"Model was trained with scikit-learn {artifact.get('sklearn_version')} but this app runs "
        f"{sklearn.__version__}. Pin the same version in requirements.txt."
    )

cats = artifact["categories"]
values = {}

st.subheader("Customer profile")
c1, c2 = st.columns(2)
with c1:
    values["gender"] = st.selectbox("Gender", cats["gender"])
    values["SeniorCitizen"] = 1 if st.selectbox("Senior citizen", ["No", "Yes"]) == "Yes" else 0
with c2:
    values["Partner"] = st.selectbox("Has partner", cats["Partner"])
    values["Dependents"] = st.selectbox("Has dependents", cats["Dependents"])

st.subheader("Services")
values["PhoneService"] = st.selectbox("Phone service", cats["PhoneService"], index=cats["PhoneService"].index("Yes"))
if values["PhoneService"] == "No":
    values["MultipleLines"] = "No phone service"
else:
    values["MultipleLines"] = st.selectbox("Multiple lines", ["No", "Yes"])

values["InternetService"] = st.selectbox("Internet service", cats["InternetService"])
addons = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
if values["InternetService"] == "No":
    for addon in addons:
        values[addon] = "No internet service"
    st.info("No internet service selected, so internet add-ons are set to 'No internet service'.")
else:
    cols = st.columns(3)
    for i, addon in enumerate(addons):
        with cols[i % 3]:
            values[addon] = st.selectbox(addon, ["No", "Yes"])

st.subheader("Contract and billing")
b1, b2 = st.columns(2)
with b1:
    values["Contract"] = st.selectbox("Contract", cats["Contract"])
    values["PaperlessBilling"] = st.selectbox("Paperless billing", cats["PaperlessBilling"])
    values["PaymentMethod"] = st.selectbox("Payment method", cats["PaymentMethod"])
with b2:
    values["tenure"] = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12, step=1)
    values["MonthlyCharges"] = st.number_input("Monthly charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=0.05)
    estimate_total = st.checkbox("Estimate total charges as tenure x monthly charges", value=True)
    if estimate_total:
        values["TotalCharges"] = round(values["tenure"] * values["MonthlyCharges"], 2)
        st.caption(f"Total charges used: ${values['TotalCharges']:,.2f}")
    else:
        values["TotalCharges"] = st.number_input("Total charges ($)", min_value=0.0, value=0.0, step=1.0)

if st.button("Predict churn risk", type="primary"):
    problems = validate_inputs(values, cats)
    if problems:
        for message in problems:
            st.error(message)
    else:
        try:
            row = build_input_row(values, artifact["input_columns"])
            probability = float(artifact["model"].predict_proba(row)[0, 1])
            threshold = float(artifact["threshold"])
        except Exception as error:  # show a friendly message instead of a stack trace
            st.error(f"Could not make a prediction: {error}")
        else:
            label = "Churn probability" if artifact.get("probabilities_calibrated") else "Churn risk score"
            st.metric(label, f"{probability:.1%}")
            st.progress(min(max(probability, 0.0), 1.0))
            if probability >= threshold:
                st.error(f"⚠️ **Likely to churn** (score is at or above the decision threshold of {threshold:.2f}).")
            else:
                st.success(f"✅ **Likely to stay** (score is below the decision threshold of {threshold:.2f}).")
            if not artifact.get("probabilities_calibrated"):
                st.caption("This score ranks customers by risk; it is not a calibrated probability.")
