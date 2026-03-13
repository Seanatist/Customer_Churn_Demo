import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


@st.cache_resource
def load_artifacts():
    """
    Load the trained churn model and encoders that were created
    by running CustomerChurn.py.
    """
    base_dir = Path(__file__).resolve().parent

    with open(base_dir / "customer_churn_model.pkl", "rb") as f:
        model_data = pickle.load(f)

    with open(base_dir / "encoders.pkl", "rb") as f:
        encoders = pickle.load(f)

    model = model_data["model"]
    feature_names = model_data["features_names"]

    return model, encoders, feature_names


def build_input_ui(encoders):
    """
    Build Streamlit widgets for all input features, using the
    encoder classes to populate categorical choices.
    Returns a dict matching the feature names used in training.
    """
    st.sidebar.header("Customer profile")

    # Use encoder classes as choices for categorical columns
    # This keeps UI options consistent with training data.
    def cat_input(col_name, label=None):
        encoder = encoders[col_name]
        options = list(encoder.classes_)
        label = label or col_name
        return st.sidebar.selectbox(label, options)

    # Categorical inputs
    gender = cat_input("gender", "Gender")
    senior_citizen = cat_input("SeniorCitizen", "Senior Citizen")
    partner = cat_input("Partner", "Partner")
    dependents = cat_input("Dependents", "Dependents")
    phone_service = cat_input("PhoneService", "Phone service")
    multiple_lines = cat_input("MultipleLines", "Multiple lines")
    internet_service = cat_input("InternetService", "Internet service")
    online_security = cat_input("OnlineSecurity", "Online security")
    online_backup = cat_input("OnlineBackup", "Online backup")
    device_protection = cat_input("DeviceProtection", "Device protection")
    tech_support = cat_input("TechSupport", "Tech support")
    streaming_tv = cat_input("StreamingTV", "Streaming TV")
    streaming_movies = cat_input("StreamingMovies", "Streaming movies")
    contract = cat_input("Contract", "Contract type")
    paperless_billing = cat_input("PaperlessBilling", "Paperless billing")
    payment_method = cat_input("PaymentMethod", "Payment method")

    # Numeric inputs – keep sensible bounds and defaults
    st.sidebar.header("Billing & tenure")
    tenure = st.sidebar.number_input("Tenure (months)", min_value=0, max_value=120, value=12, step=1)
    monthly_charges = st.sidebar.number_input(
        "Monthly charges",
        min_value=0.0,
        max_value=300.0,
        value=70.0,
        step=1.0,
    )
    total_charges = st.sidebar.number_input(
        "Total charges",
        min_value=0.0,
        max_value=10000.0,
        value=70.0 * max(tenure, 1),
        step=10.0,
    )

    input_data = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    return input_data


def preprocess_input(input_data, encoders, feature_names):
    """
    Apply the same label encoders used in training, handle unseen labels,
    and ensure columns are ordered to match the model's expectations.
    """
    input_df = pd.DataFrame([input_data])

    # Apply encoders to categorical columns
    for col, encoder in encoders.items():
        if col in input_df.columns:
            input_df[col] = input_df[col].astype(str)

            # Handle unseen labels by extending classes_ on the fly
            classes = list(encoder.classes_)
            new_values = [x for x in input_df[col] if x not in classes]
            if new_values:
                encoder.classes_ = np.append(encoder.classes_, new_values)

            input_df[col] = encoder.transform(input_df[col])

    # Ensure all expected feature columns are present
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns to match training
    input_df = input_df[feature_names]

    return input_df


def main():
    st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
    st.title("Customer Churn Prediction")
    st.markdown(
        "Interactively explore how a trained model predicts whether a telecom customer is "
        "likely to **churn** (leave) based on their profile and billing information."
    )

    try:
        model, encoders, feature_names = load_artifacts()
    except FileNotFoundError:
        st.error(
            "Model files not found. Please run `CustomerChurn.py` once in this folder "
            "so that `encoders.pkl` and `customer_churn_model.pkl` are created."
        )
        return

    input_data = build_input_ui(encoders)

    if st.button("Predict churn"):
        X = preprocess_input(input_data, encoders, feature_names)
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0][1]  # probability of churn class

        if pred == 1:
            st.error(f"⚠️ This customer is **likely to churn**. (probability: {proba:.2%})")
        else:
            st.success(f"✅ This customer is **unlikely to churn**. (probability: {proba:.2%})")

        with st.expander("Show model-ready features"):
            st.write(pd.DataFrame(X, index=["input"]))


if __name__ == "__main__":
    main()

