# Paddy Yield Prediction

A Machine Learning Engineering project for predicting rice yield from agricultural, environmental, and cultivation-related factors. The project covers model development, experiment tracking, API-based inference, and containerized deployment.

## Overview

Rice yield prediction can help estimate expected production based on cultivation and environmental conditions. This project develops and evaluates several regression models to identify the best-performing approach for rice yield prediction.

The project compares:

- Linear Regression
- K-Nearest Neighbors (KNN)
- Random Forest

The best-performing model is an optimized Random Forest Regressor, which is served through a FastAPI inference API and can be accessed through the Streamlit application.

## Machine Learning Pipeline

The modeling workflow consists of:

1. Data preprocessing
2. Duplicate removal
3. Feature selection
4. One-hot encoding
5. Robust scaling
6. Train-test splitting
7. Hyperparameter tuning
8. 5-fold cross-validation
9. Model evaluation
10. Experiment tracking with MLflow
11. Model serving with FastAPI
12. Containerization with Docker

## Dataset

The project uses the Paddy Crop Dataset from the UCI Machine Learning Repository.

After preprocessing:

- Samples: 2,338
- Original features: 45
- Missing values: 0
- Duplicate records: Removed

## Feature Selection

Feature importance was analyzed using a Random Forest model to identify the most relevant variables for yield prediction.

The final model uses the following features:

- Hectares
- Variety
- Soil Types
- Seedrate (Kg)
- Urea_40Days
- Potassh_50Days
- 30DRain (mm)
- Relative Humidity_D1_D30

## Models and Hyperparameter Tuning

Three regression algorithms were evaluated:

- Linear Regression
- K-Nearest Neighbors (KNN)
- Random Forest

### K-Nearest Neighbors

The following hyperparameters were evaluated:

- `n_neighbors`: 3, 5, 7
- `weights`: uniform, distance

Best configuration:

```text
n_neighbors = 7
weights = uniform
```

### Random Forest

The following hyperparameters were evaluated:

- `n_estimators`: 50, 100, 150
- `max_depth`: None, 10, 20

Best configuration:

```text
n_estimators = 150
max_depth = None
```

## Model Performance

| Model | MAE (kg) | RMSE (kg) | P90 Error (kg) | R² |
|---|---:|---:|---:|---:|
| Optimized Random Forest | **657.62** | **912.78** | **1426.74** | **0.9903** |
| Optimized KNN | 682.38 | 937.67 | 1458.57 | 0.9897 |
| Linear Regression | 762.88 | 1021.45 | 1584.16 | 0.9878 |

The optimized Random Forest achieved the best overall performance with:

- MAE: **657.62 kg**
- RMSE: **912.78 kg**
- R²: **0.9903**

## Generalization Check

The difference between training and testing R² was analyzed to assess model generalization.

| Model | Train R² | Test R² | Gap |
|---|---:|---:|---:|
| Random Forest | 0.9922 | 0.9903 | 0.0019 |
| KNN | 0.9916 | 0.9897 | 0.0019 |
| Linear Regression | 0.9893 | 0.9878 | 0.0015 |

All models produced a train-test R² gap below 0.01 on this evaluation, suggesting no substantial performance gap between the training and test sets.

## Experiment Tracking with MLflow

MLflow is used to track and compare machine learning experiments.

The tracked information includes:

- Model parameters
- Hyperparameters
- MAE
- RMSE
- R²
- Model artifacts

This provides a systematic way to compare different model configurations and identify the best-performing model.

## FastAPI

The trained Random Forest model is served through a FastAPI application for inference.

### Available Endpoints

#### Health Check

```http
GET /health
```

Used to verify that the API is running correctly.

#### Prediction

```http
POST /predict
```

Accepts agricultural input features and returns the predicted rice yield.

Example response:

```json
{
  "prediction_kg": 23982.55
}
```

The API also provides interactive documentation through Swagger UI.

## Docker

The FastAPI inference service is containerized using Docker.

The Docker container packages:

- Python runtime
- Required dependencies
- Trained model
- Preprocessing artifacts
- FastAPI application

Build the Docker image:

```bash
docker build -t paddy-yield-api .
```

Run the container:

```bash
docker run -p 8000:8000 paddy-yield-api
```

The API will be available at:

```text
http://localhost:8000
```

Swagger UI:

```text
http://localhost:8000/docs
```

## Streamlit Application

The project includes a Streamlit interface for interactive prediction.

The application allows users to:

- Enter agricultural parameters
- Generate rice yield predictions
- View predicted productivity
- Interpret prediction results
- View recommendations based on prediction results

Run the Streamlit application:

```bash
streamlit run app.py
```

## System Architecture

```text
                    Paddy Dataset
                         |
                         v
                 Data Preprocessing
                         |
                         v
                  Feature Selection
                         |
                         v
              Model Training & Tuning
                         |
                         v
                      MLflow
                 Experiment Tracking
                         |
                         v
              Optimized Random Forest
                         |
                         v
                     FastAPI
                  /predict /health
                         |
                         v
                       Docker
                         |
                         v
                Inference Application
```

## Project Structure

```text
Model/
│
├── api/
│   └── main.py
│
├── data/
│   └── paddydataset.csv
│
├── models/
│   ├── best_random_forest_model.pkl
│   ├── robust_standard_scaler.pkl
│   └── feature_columns.pkl
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── evaluate.py
│
├── app.py
├── train_model.py
├── main.ipynb
│
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── requirements-api.txt
├── README.md
└── .gitignore
```

## Installation

Clone the repository:

```bash
git clone https://github.com/stargazingss/Model.git
```

Move into the project directory:

```bash
cd Model
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment on Windows:

```bash
.venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

### Streamlit

```bash
streamlit run app.py
```

### FastAPI

Install the API dependencies:

```bash
pip install -r requirements-api.txt
```

Run the API:

```bash
uvicorn api.main:app --reload
```

Open Swagger UI:

```text
http://localhost:8000/docs
```

### Docker

Build the image:

```bash
docker build -t paddy-yield-api .
```

Run the container:

```bash
docker run -p 8000:8000 paddy-yield-api
```

Open Swagger UI:

```text
http://localhost:8000/docs
```

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- MLflow
- FastAPI
- Uvicorn
- Streamlit
- Docker
- Git
- GitHub

## Limitations

- The model is trained and evaluated on the available Paddy Crop Dataset.
- Model performance may vary on data from different geographical regions, cultivation practices, or environmental conditions.
- The current system focuses on prediction and inference rather than continuous model monitoring or automated retraining.

## Future Improvements

Potential improvements include:

- Automated model retraining
- Model version management
- Cloud deployment
- Model monitoring
- CI/CD integration
- Additional agricultural and environmental data

## License

This project was developed for academic and portfolio purposes.