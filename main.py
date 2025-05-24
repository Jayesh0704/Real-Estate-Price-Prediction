from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model, scaler, and feature columns
try:
    model = joblib.load("model.joblib")
    scaler = joblib.load("scaler.joblib")
    FEATURE_COLS = joblib.load("feature_columns.joblib")
    logger.info("Model, scaler, and feature columns loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model, scaler, or feature columns: {str(e)}")
    raise

# Define the request schema
class HouseFeatures(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    ocean_proximity: str

app = FastAPI(title="House Price Predictor")

@app.post("/predict")
async def predict_price(features: HouseFeatures):
    try:
        df = pd.DataFrame([features.dict()])
        df = pd.get_dummies(df, columns=["ocean_proximity"])
        df = df.reindex(columns=FEATURE_COLS, fill_value=0)
        X_scaled = scaler.transform(df)
        pred = model.predict(X_scaled)[0]
        logger.info(f"Prediction successful: {pred}")
        return {"predicted_price": float(pred)}
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
async def read_root():
    return {"message": "Welcome to the House Price Prediction API"}