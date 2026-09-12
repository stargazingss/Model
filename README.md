# Paddy Yield Prediction

Machine Learning project for predicting paddy yield based on agricultural and environmental factors.

## Live Demo

[Streamlit App](https://stargazingss-model-app-tzj2im.streamlit.app/)

## Dataset

The project uses the **UCI Paddy Crop Dataset**, containing agricultural and environmental data related to paddy production.

### Selected Features

The modeling workflow uses selected agricultural and environmental features, including:

- Hectares
- Variety
- Soil Types
- Seedrate (Kg)
- Urea_40Days
- Potassh_50Days
- 30DRain (mm)
- Relative Humidity_D1_D30

## Models

Three regression models were implemented and evaluated:

- Random Forest
- K-Nearest Neighbors (KNN)
- Linear Regression

### Hyperparameter Tuning

**KNN**

- `n_neighbors`: 3, 5, 7
- `weights`: uniform, distance
- Best configuration: `n_neighbors=7`, `weights=uniform`

**Random Forest**

- `n_estimators`: 50, 100, 150
- `max_depth`: None, 10, 20
- Best configuration: `n_estimators=150`, `max_depth=None`

## Model Performance

| Model | MAE (kg) | RMSE (kg) | R² |
|---|---:|---:|---:|
| Random Forest | 657.62 | 912.78 | 0.9903 |
| KNN | 682.38 | 937.67 | 0.9897 |
| Linear Regression | 762.88 | 1021.45 | 0.9878 |

The optimized Random Forest achieved the best overall performance with:

- **MAE:** 657.62 kg
- **RMSE:** 912.78 kg
- **R²:** 0.9903

R² is a regression evaluation metric and should not be interpreted as prediction accuracy.

## Generalization Performance

| Model | Test R² | Train-Test R² Gap |
|---|---:|---:|
| Random Forest | 0.9903 | 0.0019 |
| KNN | 0.9897 | 0.0025 |
| Linear Regression | 0.9878 | 0.0018 |

The small train-test R² gaps indicate that the models maintain consistent performance between training and test data.

## MLflow

MLflow is used to track machine learning experiments, including:

- Model parameters
- MAE
- RMSE
- R²
- Model artifacts

This allows different experiments and model configurations to be compared systematically.

## FastAPI

A FastAPI inference service was implemented to expose the trained model through an API.

### Endpoints

**Health Check**

```text
GET /health
```

**Prediction**

```text
POST /predict
```

The `/predict` endpoint accepts agricultural input features and returns the predicted paddy yield in kilograms.

## Docker

The FastAPI service is containerized using Docker to provide a consistent runtime environment.

Build the image:

```bash
docker build -t paddy-yield-api .
```

Run the container:

```bash
docker run -p 8000:8000 paddy-yield-api
```

The API can then be accessed through:

```text
http://localhost:8000/docs
```

## Streamlit

A Streamlit application provides an interactive interface for paddy yield prediction.

The deployed Streamlit application loads the trained model and preprocessing artifacts directly for prediction.

### Deployment

The application is deployed using Streamlit Community Cloud.

**Live Demo:** [Paddy Yield Prediction](https://stargazingss-model-app-tzj2im.streamlit.app/)

## Architecture

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
MLflow Experiment Tracking
      |
      v
Optimized Random Forest
      |
      +----------------------+
      |                      |
      v                      v
Streamlit Application    FastAPI API
      |                      |
      v                      v
Streamlit Cloud           Docker
      |                      |
      v                      v
  Live Demo             API Inference
```

## Project Structure

```text
Model/
├── api/
│   └── main.py
├── data/
│   └── paddydataset.csv
├── models/
│   ├── random_forest_model.pkl
│   ├── knn_model.pkl
│   ├── linear_regression_model.pkl
│   ├── robust_standard_scaler.pkl
│   └── feature_columns.pkl
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── evaluate.py
├── app.py
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── requirements-api.txt
└── README.md
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

Run the interactive prediction application:

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

Build the Docker image:

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
- Model performance may vary when applied to data from different geographical regions, cultivation practices, or environmental conditions.
- The current system focuses on prediction and inference rather than continuous model monitoring or automated retraining.
- The Streamlit deployment is intended as an interactive prediction demo and does not represent a production-scale ML serving infrastructure.

## Future Improvements

Potential improvements include:

- Automated model retraining
- Model version management
- Model monitoring
- CI/CD integration
- Additional agricultural and environmental data
- Production cloud deployment for the FastAPI inference service

## License

This project was developed for academic and portfolio purposes.