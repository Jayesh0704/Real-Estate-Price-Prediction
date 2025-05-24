# Real Estate Price Prediction

This project predicts house prices using an XGBoost model trained on the California Housing dataset. It includes a FastAPI app for serving predictions, Docker for containerization, and GitHub Actions for CI/CD to deploy the app to a remote server.

## Project Structure
- `main.py`: FastAPI app for predictions.
- `real_estate_prediction.ipynb`: Notebook for training the model and logging with MLflow.
- `xgb_specific.ipynb`: Additional notebook for XGBoost experiments.
- `Dockerfile`: Docker configuration.
- `requirements.txt`: Python dependencies.
- `.github/workflows/ci-cd.yml`: GitHub Actions workflow for CI/CD.

## Setup
1. Ensure secrets (`DOCKER_USERNAME`, `DOCKER_PASSWORD`, `SERVER_HOST`, `SERVER_USERNAME`, `SSH_KEY`) are set in GitHub Actions.
2. Push changes to the `main` branch to trigger the CI/CD pipeline.

## API Endpoints
- **GET /**: Welcome message.
- **POST /predict**: Predict house price.
  - Example: `curl -X POST "http://SERVER_HOST:8000/predict" -H "Content-Type: application/json" -d '{"longitude": -122.23, "latitude": 37.88, "housing_median_age": 41.0, "total_rooms": 880.0, "total_bedrooms": 129.0, "population": 322.0, "households": 126.0, "median_income": 8.3252, "ocean_proximity": "<1H OCEAN"}'`

## Deployment
The CI/CD pipeline deploys the app to a remote server on port 8000.
