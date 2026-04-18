import os
import time
import uuid
import numpy as np
import pandas as pd
import joblib
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Global state
ml_model = None
preprocessor = None
start_time = None
MODEL_VERSION = "1.0.0"
MODEL_PATH = os.getenv("MODEL_PATH", "models/rf_regressor.joblib")
PREPROCESSOR_PATH = os.getenv("PREPROCESSOR_PATH", "models/preprocessor.joblib")

# Load model and preprocessor once at startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    global ml_model, preprocessor, start_time
    ml_model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    start_time = time.time()
    print(f"Model loaded from       {MODEL_PATH}")
    print(f"Preprocessor loaded from {PREPROCESSOR_PATH}")
    yield
    print("Shutting down...")

# App
app = FastAPI(
    title="Taxi Tip Predictor",
    description="Predicts NYC taxi tip amount from trip features.",
    version=MODEL_VERSION,
    lifespan=lifespan,
)

# Pydantic schemas
class TaxiTripInput(BaseModel):
    passenger_count: int = Field(default=1, ge=1, le=6,
                                  description="Number of passengers (1-6)")
    trip_distance: float = Field(..., gt=0,
                                  description="Trip distance in miles")
    fare_amount: float = Field(..., ge=0,
                                description="Base fare in dollars")
    total_amount: float = Field(..., ge=0,
                                 description="Total charge including all fees")
    pickup_hour: int = Field(..., ge=0, le=23,
                              description="Hour of pickup (0-23)")
    pickup_day_of_week: int = Field(..., ge=0, le=6,
                                     description="Day of week (0=Monday, 6=Sunday)")
    is_weekend: int = Field(..., ge=0, le=1,
                             description="1 if Saturday or Sunday, else 0")
    trip_duration_minutes: float = Field(..., gt=0,
                                          description="Trip duration in minutes")
    trip_speed_mph: float = Field(..., ge=0,
                                   description="Average speed in miles per hour")
    pickup_borough: str = Field(...,
                                 description="Borough of pickup (e.g. Manhattan)")
    dropoff_borough: str = Field(...,
                                  description="Borough of dropoff (e.g. Brooklyn)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "passenger_count": 1,
                    "trip_distance": 3.5,
                    "fare_amount": 14.50,
                    "total_amount": 18.80,
                    "pickup_hour": 14,
                    "pickup_day_of_week": 2,
                    "is_weekend": 0,
                    "trip_duration_minutes": 12.5,
                    "trip_speed_mph": 16.8,
                    "pickup_borough": "Manhattan",
                    "dropoff_borough": "Brooklyn"
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    prediction_id: str
    predicted_tip_amount: float
    model_version: str


class BatchInput(BaseModel):
    records: List[TaxiTripInput] = Field(..., max_length=100)


class BatchResponse(BaseModel):
    predictions: List[PredictionResponse]
    count: int
    processing_time_ms: float


# Helper functions
def build_feature_dataframe(trip: TaxiTripInput) -> pd.DataFrame:
    """Convert a TaxiTripInput into a DataFrame matching the training feature set."""
    return pd.DataFrame([{
        "passenger_count":       trip.passenger_count,
        "trip_distance":         trip.trip_distance,
        "fare_amount":           trip.fare_amount,
        "total_amount":          trip.total_amount,
        "pickup_hour":           trip.pickup_hour,
        "pickup_day_of_week":    trip.pickup_day_of_week,
        "is_weekend":            trip.is_weekend,
        "trip_duration_minutes": trip.trip_duration_minutes,
        "trip_speed_mph":        trip.trip_speed_mph,
        "log_trip_distance":     np.log1p(trip.trip_distance),
        "fare_per_mile":         trip.fare_amount / trip.trip_distance,
        "fare_per_minute":       trip.fare_amount / trip.trip_duration_minutes,
        "pickup_borough":        trip.pickup_borough,
        "dropoff_borough":       trip.dropoff_borough,
    }])


def make_prediction(trip: TaxiTripInput) -> PredictionResponse:
    """Run the full pipeline (feature build → preprocess → predict) for one trip."""
    row_df = build_feature_dataframe(trip)
    row_processed = preprocessor.transform(row_df)
    raw_pred = ml_model.predict(row_processed)[0]

    return PredictionResponse(
        prediction_id=str(uuid.uuid4()),
        predicted_tip_amount=round(float(raw_pred), 2),
        model_version=MODEL_VERSION,
    )


# Endpoints
@app.get("/")
def root():
    return {"message": "Taxi Tip Predictor API is running", "docs": "/docs"}


@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: TaxiTripInput):
    """Predict tip amount for a single taxi trip."""
    return make_prediction(input_data)


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(batch: BatchInput):
    """Predict tip amounts for up to 100 trips in one request."""
    t_start = time.time()
    predictions = [make_prediction(record) for record in batch.records]
    elapsed_ms = round((time.time() - t_start) * 1000, 2)

    return BatchResponse(
        predictions=predictions,
        count=len(predictions),
        processing_time_ms=elapsed_ms,
    )


@app.get("/health")
def health_check():
    """Returns API and model status."""
    return {
        "status": "healthy",
        "model_loaded": ml_model is not None,
        "model_version": MODEL_VERSION,
        "uptime_seconds": round(time.time() - start_time, 1),
    }


@app.get("/model/info")
def model_info():
    """Returns metadata about the currently loaded model."""
    return {
        "model_name": "taxi-tip-regressor",
        "model_version": MODEL_VERSION,
        "features": [
            "passenger_count",
            "trip_distance",
            "fare_amount",
            "total_amount",
            "pickup_hour",
            "pickup_day_of_week",
            "is_weekend",
            "trip_duration_minutes",
            "trip_speed_mph",
            "log_trip_distance",
            "fare_per_mile",
            "fare_per_minute",
            "pickup_borough",
            "dropoff_borough",
        ],
        "metrics": {
            "mae":  0.1265,
            "rmse": 0.6846,
            "r2":   0.9676,
        },
        "trained_on": "NYC Yellow Taxi Trip Records — January 2023",
    }


# Global error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred. Please try again.",
        },
    )