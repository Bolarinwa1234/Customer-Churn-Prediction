from contextlib import asynccontextmanager
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
import joblib
import os
from pathlib import Path
from .schemas import CustomerInput, ChurnResponse
import sys

try:
    from .schemas import CustomerInput, ChurnResponse
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from schemas import CustomerInput, ChurnResponse

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

ml_assets = {}



@asynccontextmanager
async def lifespan(app: FastAPI):

    ml_assets['scaler'] = joblib.load(MODELS_DIR / 'scaler.pkl')
    ml_assets['oh_encoder'] = joblib.load(MODELS_DIR / 'oh_encoder.pkl')
    ml_assets['model'] = joblib.load(MODELS_DIR / 'customer churn prediction rf model.pkl')

    yield
    ml_assets.clear()



app = FastAPI(title='Customer Churn Prediction API', lifespan=lifespan)

@app.get('/')
def root():
    return {'status': 'ok', 'docs': '/docs'}

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

    safe_tenure = new_df['tenure'].replace(0, np.nan)
    new_df['num_services'] = new_df[['PhoneService','OnlineSecurity_Yes','OnlineBackup_Yes',
    'DeviceProtection_Yes','TechSupport_Yes','StreamingTV_Yes','StreamingMovies_Yes']].sum(axis=1)
    new_df['avg_monthly_spend'] = new_df['TotalCharges'] / new_df['tenure']
    new_df['charge_gap'] = new_df['MonthlyCharges'] - new_df['avg_monthly_spend']  # recent price hike?
    new_df['is_new_customer'] = (new_df['tenure'] <= 6).astype(int)
    new_df['autopay'] = 1 - new_df['PaymentMethod_Electronic check'] - new_df['PaymentMethod_Mailed check']

    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

    new_df[numerical_cols] = ml_assets['scaler'].transform(new_df[numerical_cols])

    new_df = new_df[ml_assets['model'].feature_names_in_]

    return ml_assets['model'].predict(new_df)[0], ml_assets['model'].predict_proba(new_df)[:, 1][0]


    
@app.post('/predict', response_model=ChurnResponse)
def predict(customer: CustomerInput):
    try:
        input_df = pd.DataFrame([customer.model_dump()])
        pred, prob = list(prediction_pipeline(input_df))
        risk = 'High' if prob>=0.5 else 'Low'
        top_factors = None

        return ChurnResponse(
            churn_probability=round(prob, 4),
            prediction = 'Churn' if int(pred)==1 else 'No Churn',
            risk_level=risk,
            # top_driving_factors= None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

        

