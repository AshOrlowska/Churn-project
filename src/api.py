from fastapi import FastAPI
from pydantic import BaseModel
import joblib, pandas as pd

pipe = joblib.load("models/churn_pipeline.pkl")
app = FastAPI()

class Customer(BaseModel):
    tenure: int
    MonthlyCharges: float
    Contract: str

@app.post("/predict")
def predict_churn(customer: Customer):
    data = pd.DataFrame([customer.dict()])
    proba = pipe.predict_proba(data)[:,1][0]
    return {"churn_probability": round(float(proba), 4)}
