from fastapi import FastAPI, Query
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# -------------------------------------------------
# App initialization
# -------------------------------------------------
app = FastAPI(
    title="Multi-Model Customer Churn Prediction API",
    description="Predict customer churn using KNN, Logistic Regression, or Random Forest",
    version="1.0"
)

# -------------------------------------------------
# Load trained models
# -------------------------------------------------
MODEL_DIR = "../model"

MODELS = {
    "knn": joblib.load(os.path.join(MODEL_DIR, "churn_knn_model.pkl")),
    "logreg": joblib.load(os.path.join(MODEL_DIR, "churn_logreg_model.pkl")),
    "rf": joblib.load(os.path.join(MODEL_DIR, "churn_rf_model.pkl")),
}

DEFAULT_MODEL = "rf"

# -------------------------------------------------
# Input schema (RAW VALUES — NO MANUAL ENCODING)
# -------------------------------------------------
class CustomerInput(BaseModel):
    gender: str
    SeniorCitizen: str
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

# -------------------------------------------------
# Prediction endpoint
# -------------------------------------------------
@app.post("/predict")
def predict_churn(
    customer: CustomerInput,
    model: str = Query(
        DEFAULT_MODEL,
        description="Model to use: knn | logreg | rf (default: rf)"
    )
):
    if model not in MODELS:
        return {
            "error": f"Invalid model '{model}'. Choose from knn, logreg, rf."
        }

    selected_model = MODELS[model]

    # Convert input to DataFrame (IMPORTANT)
    input_df = pd.DataFrame([customer.dict()])

    # Predict churn
    prediction = selected_model.predict(input_df)[0]

    # Probability (if available)
    churn_probability = None
    if hasattr(selected_model, "predict_proba"):
        churn_probability = float(
            selected_model.predict_proba(input_df)[0][1]
        )

    return {
        "model_used": model,
        "prediction": int(prediction),
        "message": "Likely to Leave" if prediction == 1 else "Likely to Stay",
        "churn_probability": round(churn_probability, 4) if churn_probability is not None else None
    }

# -------------------------------------------------
# Health check
# -------------------------------------------------
@app.get("/")
def health_check():
    return {
        "status": "API is running",
        "available_models": list(MODELS.keys()),
        "default_model": DEFAULT_MODEL
    }
