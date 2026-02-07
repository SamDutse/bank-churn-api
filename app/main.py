from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Literal
import joblib
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


app = FastAPI(
    title="Bank Customer Churn Prediction API",
    description="Predict whether a bank customer will churn",
    version="1.0"
)

@app.on_event("startup")
def startup_event():
    logger.info("Churn Prediction API started successfully")


# Load trained pipeline
model = joblib.load("model/churn_pipeline.pkl")


class Customer(BaseModel):
    CreditScore: int = Field(..., ge=300, le=900)
    Geography: Literal["France", "Germany", "Spain"]
    Gender: Literal["Male", "Female"]
    Age: int = Field(..., ge=18, le=100)
    Tenure: int = Field(..., ge=0, le=10)
    Balance: float = Field(..., ge=0)
    NumOfProducts: int = Field(..., ge=1, le=4)
    HasCrCard: int = Field(..., ge=0, le=1)
    IsActiveMember: int = Field(..., ge=0, le=1)
    EstimatedSalary: float = Field(..., ge=0)

    @validator("Balance")
    def check_balance(cls, v):
        if v < 0:
            raise ValueError("Balance cannot be negative")
        return v

@app.get("/")
def home():
    return {"message": "Churn Prediction API is running"}



@app.post("/predict")
def predict(customer: Customer):
    try:
        logger.info(f"Received prediction request: {customer.dict()}")

        df = pd.DataFrame([customer.dict()])
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        logger.info(
            f"Prediction result: churn={prediction}, probability={probability}"
        )

        return {
            "prediction": int(prediction),
            "churn_probability": round(float(probability), 4)
        }

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")

        raise HTTPException(
            status_code=500,
            detail="Internal server error during prediction"
        )
