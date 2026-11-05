from contextlib import asynccontextmanager
import pandas as pd
from fastapi import FastAPI, HTTPException
import joblib
import os
from pathlib import Path
from .schemas import CustomerInput, ChurnResponse

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

ml_assets = {}



@asynccontextmanager
async def lifespan(app: FastAPI):

    ml_assets['scaler'] = joblib.load(MODELS_DIR / 'scaler.pkl')
    ml_assets['oh_encoder'] = joblib.load(MODELS_DIR / 'oh_encoder.pkl')
    ml_assets['model'] = joblib.load(MODELS_DIR / 'customer churn prediction rf model.pkl')
    
    print("--- RUNNING STARTUP PIPELINE TEST ---")
    sample_customer = pd.DataFrame([{
        "gender": "Female", "SeniorCitizen": "No", "Partner": "Yes", "Dependents": "No",
        "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 70.70, "TotalCharges": 151.65
    }])
    
    try:
        preds, probs = prediction_pipeline(sample_customer)
        print(f"Predictions: {preds} | Probabilities: {probs}")
    except Exception as e:
        print(f"Test failed with error: {e}")
    print("-------------------------------------")

    yield
    ml_assets.clear()



app = FastAPI(title='Customer Churn Prediction API', lifespan=lifespan)

@app.get('/health')
def health():
    return {'status': 'healthy'}

def prediction_pipeline(customer_input):
    binary_cols = ['SeniorCitizen', 'Partner', 'Dependents',
                    'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        if customer_input[col].dtype == 'object':
            customer_input[col] = (customer_input[col]=='Yes').astype(int)

    customer_input['Contract'] = customer_input['Contract'].replace({'Month-to-month':0, 'One year':1, 'Two year':2})

    ohe_columns = ['gender','MultipleLines','InternetService','OnlineSecurity',
                    'OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies',
                    'PaymentMethod']
    encoded_df = ml_assets['oh_encoder'].transform(customer_input[ohe_columns])
    new_df = pd.concat([customer_input.drop(columns=ohe_columns), encoded_df], axis=1)

    new_df['num_services'] = new_df[['PhoneService','OnlineSecurity_Yes','OnlineBackup_Yes',
    'DeviceProtection_Yes','TechSupport_Yes','StreamingTV_Yes','StreamingMovies_Yes']].sum(axis=1)
    new_df['avg_monthly_spend'] = new_df['TotalCharges'] / new_df['tenure']
    new_df['charge_gap'] = new_df['MonthlyCharges'] - new_df['avg_monthly_spend']  # recent price hike?
    new_df['is_new_customer'] = (new_df['tenure'] <= 6).astype(int)
    new_df['autopay'] = 1 - new_df['PaymentMethod_Electronic check'] - new_df['PaymentMethod_Mailed check']

    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

    new_df[numerical_cols] = ml_assets['scaler'].transform(new_df[numerical_cols])

    new_df = new_df[ml_assets['model'].feature_names_in_]

    return ml_assets['model'].predict(new_df), ml_assets['model'].predict_proba(new_df)[:, 1]


    
# @app.post('/predict', response_model=ChurnResponse)
# def predict(customer: CustomerInput):
#     try:
#         input_df = pd.DataFrame([customer.model_dump()])

