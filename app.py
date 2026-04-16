from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="Banking Churn Prediction API")

# Load the saved Model and Scaler
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')

class CustomerData(BaseModel):
    CreditScore: int
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float
    Geography_Germany: int
    Geography_Spain: int
    Gender_Male: int

@app.get("/")
def home():
    return {"message": "Banking Churn API is Running"}

@app.post("/predict")
def predict(data: CustomerData):

    input_dict = data.dict()
    df_input = pd.DataFrame([input_dict])
    
    scaled_data = scaler.transform(df_input)
    
    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)[0][1]
    
    return {
        "churn_prediction": int(prediction),
        "churn_probability": round(float(probability), 2),
        "status": "At Risk" if prediction == 1 else "Safe"
    }
