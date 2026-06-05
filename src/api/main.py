import os
import mlflow.pyfunc
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from src.api.pydantic_models import CustomerFeatures, RiskPredictionResponse

app = FastAPI(title="Customer Risk Prediction API", version="1.0")

MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "risk_model")
MODEL_STAGE = os.getenv("MLFLOW_MODEL_STAGE", "Production")

model = None


@app.on_event("startup")
def load_model():
    """
    Load model from MLflow Model Registry at startup.
    """
    global model
    try:
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        model = mlflow.pyfunc.load_model(model_uri)
        print(f"Loaded model from {model_uri}")
    except Exception as e:
        raise RuntimeError(f"Failed to load MLflow model: {e}")


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=RiskPredictionResponse)
def predict(features: CustomerFeatures):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Convert input to DataFrame (MLflow expects tabular input)
        input_data = pd.DataFrame([features.dict()])

        # Predict probability
        prediction = model.predict(input_data)

        # Handle different model output types
        risk_prob = float(np.array(prediction).reshape(-1)[0])

        return RiskPredictionResponse(
            risk_probability=risk_prob,
            risk_label="high_risk" if risk_prob > 0.5 else "low_risk",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))