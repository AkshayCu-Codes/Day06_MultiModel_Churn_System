# 📊 Day 06 — Multi-Model Customer Churn Intelligence System

This project represents **Day 06** of the AIML project series and brings together everything built in earlier days into a **production-style, multi-model machine learning system** with a backend API and a professional dashboard.

The system allows business users to predict customer churn, compare multiple ML models, and analyze prediction trends — all through a clean UI.

---

## 🎯 Project Objectives

- Serve **multiple ML models** through a single API
- Default to the **best-performing model** while allowing user override
- Maintain a **consistent training–serving contract**
- Log every prediction for analytics
- Provide a **professional, recruiter-ready dashboard**

---

## 🧠 Models Used

All models are trained on the **Telco Customer Churn** dataset and saved as pipelines.

| Model | Origin Day | Notes |
|------|----------|------|
| KNN | Day 03 | Baseline distance-based model |
| Logistic Regression | Day 04 | Linear probabilistic model |
| Random Forest | Day 05 | Best-performing ensemble model (default) |

---

## 🏗️ System Architecture

```
Streamlit Dashboard
    |
    |  (HTTP Request)
    v
FastAPI Backend
    |
    |  (Dynamic model selection)
    v
ML Model Pipelines (.pkl)
    |
    v
Prediction + Probability
    |
    v
predictions.csv → Analytics
```

---

## 📁 Project Structure

```
Day06_MultiModel_Churn_System/
├── api/
│   └── app.py                # FastAPI backend
│
├── dashboard/
│   ├── app.py                # Streamlit dashboard
│   └── assets/
│       ├── logo.png
│       ├── linkedin.png
│       └── github.png
│
├── model/
│   ├── churn_knn_model.pkl
│   ├── churn_logreg_model.pkl
│   └── churn_rf_model.pkl
│
├── data/
│   ├── predictions.csv       # Generated at runtime
│   └── leaderboard.csv       # Optional (future extension)
│
└── README.md
```

---

## 🚀 How to Run the System

### 1️⃣ Start the API (Terminal 1)

```bash
cd api
uvicorn app:app --reload
```

API will run at:
```
http://127.0.0.1:8000
```

Swagger Docs:
```
http://127.0.0.1:8000/docs
```

---

### 2️⃣ Start the Dashboard (Terminal 2)

```bash
cd dashboard
streamlit run app.py
```

Dashboard URL:
```
http://localhost:8501
```

---

## 🖥️ Dashboard Features

- Model selector (RF default)
- Human-friendly inputs (no 0/1 confusion)
- Two-step UX:
  - Input page
  - Results & analytics page
- Churn probability display
- Prediction history table
- Analytics charts:
  - Churn distribution
  - Model usage
  - Tenure trend
- Clear-all prediction logs
- Sidebar branding + profile links

---

## 🔌 API Usage

### Endpoint

```
POST /predict?model=rf|knn|logreg
```

### Example Request Body

```json
{
  "gender": "Male",
  "SeniorCitizen": "No",
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 70.5,
  "TotalCharges": 845.3
}
```

### Example Response

```json
{
  "model_used": "rf",
  "prediction": 1,
  "message": "Likely to Leave",
  "churn_probability": 0.87
}
```

---

## 🧩 Key Engineering Highlights

- Dynamic model selection via query parameter
- Training–serving consistency enforced
- Pipeline-based preprocessing
- API-first architecture
- Real-time analytics from live predictions
- Clear separation of concerns

---

## 📌 Future Enhancements

- Model leaderboard page
- ROC / AUC comparison view
- Export analytics (CSV / PDF)
- Authentication layer
- Docker & cloud deployment

---

