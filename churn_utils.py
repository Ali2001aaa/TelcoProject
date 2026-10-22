"""Helper code shared by the notebook and the Streamlit app.

The feature-engineering function lives in this module (not in the notebook)
so that the saved joblib pipeline can be loaded again by app.py.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

SEED = 42

# Columns that describe whether the customer has a given service
SERVICE_COLUMNS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]

ORIGINAL_NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
ORIGINAL_CATEGORICAL = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
ENGINEERED_NUMERIC = ORIGINAL_NUMERIC + ["avg_monthly_spend", "n_services", "autopay"]
ENGINEERED_CATEGORICAL = ORIGINAL_CATEGORICAL + ["tenure_band"]

# name -> (numeric columns, categorical columns)
FEATURE_SETS = {
    "original": (ORIGINAL_NUMERIC, ORIGINAL_CATEGORICAL),
    "original_no_total": ([c for c in ORIGINAL_NUMERIC if c != "TotalCharges"], ORIGINAL_CATEGORICAL),
    "engineered": (ENGINEERED_NUMERIC, ENGINEERED_CATEGORICAL),
}


def add_features(df):
    """Clean TotalCharges and add engineered columns. Uses only values from the same row,
    so nothing is learned from other customers (no leakage).

    Parameters: df - raw customer data with the original columns.
    Returns: a copy of df with TotalCharges as float plus the new columns.
    """
    df = df.copy()

    # TotalCharges is text in the raw file; blanks become NaN.
    total = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # Customers with tenure 0 have not been billed yet, so a blank means 0.
    total = total.where(~((df["tenure"] == 0) & total.isna()), 0.0)
    df["TotalCharges"] = total

    # Average amount paid per month of tenure (falls back to MonthlyCharges for tenure 0)
    safe_tenure = df["tenure"].replace(0, np.nan)
    df["avg_monthly_spend"] = (df["TotalCharges"] / safe_tenure).fillna(df["MonthlyCharges"])

    # Number of services the customer has
    df["n_services"] = (df[SERVICE_COLUMNS] == "Yes").sum(axis=1)

    # Tenure groups (in months)
    df["tenure_band"] = pd.cut(
        df["tenure"], bins=[-1, 12, 24, 48, np.inf], labels=["0-12", "13-24", "25-48", "49+"]
    ).astype(str)

    # 1 if the customer pays automatically (bank transfer or credit card)
    df["autopay"] = df["PaymentMethod"].str.contains("automatic").astype(int)
    return df


def build_pipeline(model, numeric_cols, categorical_cols, scale=False, select=False):
    """Build: add_features -> ColumnTransformer -> (optional feature selection) -> model.

    Parameters:
        model: an unfitted scikit-learn classifier.
        numeric_cols / categorical_cols: columns to use after add_features().
        scale: standardise numeric columns (needed for Logistic Regression).
        select: add L1-based feature selection (fitted inside each training fold).
    Returns: an unfitted Pipeline that accepts raw customer data.
    """
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))

    preprocessor = ColumnTransformer([
        ("num", Pipeline(numeric_steps), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
    ])

    steps = [("features", FunctionTransformer(add_features)), ("prep", preprocessor)]
    if select:
        # L1 penalty pushes weak features to exactly zero weight; those are dropped
        l1_model = LinearSVC(
            penalty="l1", dual=False, C=0.05, class_weight="balanced", max_iter=5000, random_state=SEED
        )
        steps.append(("select", SelectFromModel(l1_model, threshold=1e-5)))
    steps.append(("model", model))
    return Pipeline(steps)
