# COMP 3610 Assignment 4: MLOps & Model Deployment

## Prerequisites
- Python 3.10+
- Docker Desktop installed and running

## Setup

### 1. Clone the repository
git clone <your-repo-url>
cd assignment4

### 2. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

### 3. Install dependencies
pip install -r requirements.txt

### 4. Download the data and models
Run the notebook cells in assignment4.ipynb from the beginning.

## Running the API locally
uvicorn app:app --reload --port 8000

## Running with Docker Compose
docker compose up --build

## Running tests
pytest test_app.py -v

## API Endpoints
- GET  /health - API and model status
- GET  /model/info - Model metadata
- POST /predict - Single trip prediction
- POST /predict/batch - Batch predictions (up to 100)
- GET  /docs - Swagger UI documentation
