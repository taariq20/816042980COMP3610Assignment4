import pytest
from fastapi.testclient import TestClient
from app import app

# Valid input fixture
VALID_TRIP = {
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
    "dropoff_borough": "Brooklyn",
}

# Module-scoped client - (Loads model + preprocessor)
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

# Happy path tests
def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "uptime_seconds" in data


def test_model_info(client):
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "features" in data
    assert "metrics" in data
    assert "model_name" in data


def test_predict_valid_input(client):
    response = client.post("/predict", json=VALID_TRIP)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_tip_amount" in data
    assert "prediction_id" in data
    assert "model_version" in data
    assert isinstance(data["predicted_tip_amount"], float)
    assert data["predicted_tip_amount"] >= 0


def test_predict_response_is_rounded(client):
    response = client.post("/predict", json=VALID_TRIP)
    assert response.status_code == 200
    tip = response.json()["predicted_tip_amount"]
    assert round(tip, 2) == tip


def test_predict_each_call_gets_unique_id(client):
    id1 = client.post("/predict", json=VALID_TRIP).json()["prediction_id"]
    id2 = client.post("/predict", json=VALID_TRIP).json()["prediction_id"]
    assert id1 != id2


def test_predict_batch_valid(client):
    batch = {"records": [VALID_TRIP] * 3}
    response = client.post("/predict/batch", json=batch)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 3
    assert len(data["predictions"]) == 3
    assert "processing_time_ms" in data


def test_predict_batch_single_record(client):
    batch = {"records": [VALID_TRIP]}
    response = client.post("/predict/batch", json=batch)
    assert response.status_code == 200
    assert response.json()["count"] == 1


# Validation tests - all must return 422
def test_predict_missing_required_field(client):
    """Omitting trip_distance should return 422."""
    bad = {k: v for k, v in VALID_TRIP.items() if k != "trip_distance"}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_predict_invalid_type(client):
    """String where float expected should return 422."""
    bad = {**VALID_TRIP, "trip_distance": "far"}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_predict_negative_distance(client):
    """trip_distance must be > 0."""
    bad = {**VALID_TRIP, "trip_distance": -1.0}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_predict_zero_distance(client):
    """trip_distance=0 must be rejected (gt=0 not ge=0)."""
    bad = {**VALID_TRIP, "trip_distance": 0.0}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_predict_invalid_pickup_hour(client):
    """pickup_hour must be 0-23."""
    bad = {**VALID_TRIP, "pickup_hour": 25}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_predict_negative_fare(client):
    """fare_amount must be >= 0."""
    bad = {**VALID_TRIP, "fare_amount": -5.0}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_predict_invalid_is_weekend(client):
    """is_weekend must be 0 or 1."""
    bad = {**VALID_TRIP, "is_weekend": 5}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_predict_invalid_passenger_count(client):
    """passenger_count must be 1-6."""
    bad = {**VALID_TRIP, "passenger_count": 10}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_batch_exceeds_limit(client):
    """Sending more than 100 records should return 422."""
    batch = {"records": [VALID_TRIP] * 101}
    response = client.post("/predict/batch", json=batch)
    assert response.status_code == 422


# Edge case tests
def test_predict_very_large_fare(client):
    """Extreme but valid fare should still return a prediction."""
    edge = {**VALID_TRIP, "fare_amount": 499.99, "total_amount": 520.00}
    response = client.post("/predict", json=edge)
    assert response.status_code == 200
    assert "predicted_tip_amount" in response.json()


def test_predict_long_trip(client):
    """Long duration trip should still return a valid prediction."""
    edge = {
        **VALID_TRIP,
        "trip_distance": 45.0,
        "trip_duration_minutes": 90.0,
        "trip_speed_mph": 30.0,
        "fare_amount": 120.0,
        "total_amount": 135.0,
    }
    response = client.post("/predict", json=edge)
    assert response.status_code == 200
    assert response.json()["predicted_tip_amount"] >= 0


def test_predict_weekend_trip(client):
    """Weekend flag set correctly should return a valid prediction."""
    weekend = {
        **VALID_TRIP,
        "pickup_day_of_week": 6,
        "is_weekend": 1,
    }
    response = client.post("/predict", json=weekend)
    assert response.status_code == 200