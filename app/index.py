import sys
import traceback
from pathlib import Path
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    report = {}
    models = Path(__file__).resolve().parent.parent / "models"

    try:
        import joblib
        scaler = joblib.load(models / "scaler.pkl")
        report["scaler"] = "ok"
        encoder = joblib.load(models / "oh_encoder.pkl")
        report["encoder"] = "ok"
        model = joblib.load(models / "customer churn prediction rf model.pkl")
        report["model"] = "ok"
        report["features"] = list(model.feature_names_in_)[:5]
    except Exception:
        report["error"] = traceback.format_exc()[-800:]

    return report