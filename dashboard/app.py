import streamlit as st
import pandas as pd
import requests
import os
from datetime import datetime

# -------------------------------------------------
# Configuration
# -------------------------------------------------
API_URL = "http://127.0.0.1:8000/predict"
DATA_DIR = "../data"
PREDICTIONS_FILE = os.path.join(DATA_DIR, "predictions.csv")

os.makedirs(DATA_DIR, exist_ok=True)

# -------------------------------------------------
# Page setup
# -------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide"
)

# -------------------------------------------------
# Session state
# -------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "input"

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
with st.sidebar:
    # ---------------------------------
    # Logo
    # ---------------------------------
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.image("assets/logo.png", width=60)



    # ---------------------------------
    # Product name & tagline
    # ---------------------------------
    st.markdown("## Customer Churn System")
    st.markdown(
        "<span style='color:#6b7280;'>Predict • Analyze • Decide</span>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    # ---------------------------------
    # Model selection
    # ---------------------------------
    model_map = {
        "Random Forest (Recommended)": "rf",
        "Logistic Regression": "logreg",
        "KNN": "knn"
    }

    selected_label = st.selectbox(
        "Prediction Model",
        list(model_map.keys()),
        index=0
    )

    selected_model = model_map[selected_label]

    st.markdown("---")

    # ---------------------------------
    # System description
    # ---------------------------------
    st.markdown(
        "<p style='color:#6b7280; font-size:0.85rem;'>"
        "This system predicts customer churn using multiple machine learning "
        "models served via an API."
        "</p>",
        unsafe_allow_html=True
    )

    # ---------------------------------
    # Social links (bottom)
    # ---------------------------------
    st.markdown("#### Connect")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.link_button(
        "LinkedIn Profile",
        "https://www.linkedin.com/in/akshay-cu/"
    )
    with col1:
        st.link_button(
        "GitHub Profile",
        "https://github.com/AkshayCu-Codes"
    )

    






# =================================================
# PAGE 1 — INPUT
# =================================================
if st.session_state.page == "input":

    st.title("Customer Churn Prediction")
    st.caption("Enter customer details to generate a churn prediction")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            tenure = st.number_input("Tenure (months)", 0, 120, 12)

        with col2:
            phone = st.selectbox("Phone Service", ["No", "Yes"])
            multiline = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

        with col3:
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

        paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        monthly = st.number_input("Monthly Charges (€)", 0.0, 200.0, 70.0)
        total = st.number_input("Total Charges (€)", 0.0, 10000.0, 900.0)

        submitted = st.form_submit_button("Run Prediction")

    if submitted:
        payload = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiline,
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total
        }

        response = requests.post(API_URL, params={"model": selected_model}, json=payload)

        if response.status_code == 200:
            result = response.json()

            record = payload.copy()
            record.update({
                "model": selected_model,
                "prediction": result["prediction"],
                "probability": result["churn_probability"],
                "timestamp": datetime.now().isoformat()
            })

            st.session_state.last_prediction = record

            df = pd.DataFrame([record])
            if os.path.exists(PREDICTIONS_FILE):
                df.to_csv(PREDICTIONS_FILE, mode="a", header=False, index=False)
            else:
                df.to_csv(PREDICTIONS_FILE, index=False)

            st.session_state.page = "results"
            st.rerun()
        else:
            st.error("Prediction service is unavailable.")

# =================================================
# PAGE 2 — RESULTS & ANALYTICS
# =================================================
if st.session_state.page == "results":

    r = st.session_state.last_prediction

    st.title("Prediction Results")

    if r["prediction"] == 1:
        st.error(f"High churn risk detected ({r['probability']*100:.1f}% probability)")
    else:
        st.success(f"Low churn risk detected ({r['probability']*100:.1f}% probability)")

    st.subheader("Prediction Analytics")

    hist = pd.read_csv(PREDICTIONS_FILE)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("Churn Outcome Distribution")
        st.bar_chart(hist["prediction"].value_counts())

    with col2:
        st.markdown("Model Usage Distribution")
        st.bar_chart(hist["model"].value_counts())

    with col3:
        st.markdown("Tenure Trend")
        st.line_chart(hist["tenure"])

    st.subheader("Prediction History")
    st.dataframe(hist, use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        if st.button("New Prediction"):
            st.session_state.page = "input"
            st.rerun()
    with c2:
        if st.button("Clear All Predictions"):
            os.remove(PREDICTIONS_FILE)
            st.session_state.page = "input"
            st.rerun()