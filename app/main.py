# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(
    title="Bank Customer Churn Prediction API",
    version="1.0"
)

# Load trained pipeline
model = joblib.load("model/churn_pipeline.pkl")

class Customer(BaseModel):
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

@app.get("/")
def home():
    return {"message": "Churn Prediction API is running"}

@app.post("/predict")
def predict_churn(customer: Customer):

    data = pd.DataFrame([customer.dict()])

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    return {
        "churn_prediction": int(prediction),
        "churn_probability": round(float(probability), 4)
    }
