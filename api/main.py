from pathlib import Path
import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"


model = joblib.load(
    MODEL_DIR / "best_random_forest_model.pkl"
)

scaler = joblib.load(
    MODEL_DIR / "robust_standard_scaler.pkl"
)

feature_columns = joblib.load(
    MODEL_DIR / "feature_columns.pkl"
)


app = FastAPI(
    title="Paddy Yield Prediction API",
    description="API for predicting paddy yield using a trained Random Forest model",
    version="1.0.0"
)


# 1. Health check
@app.get("/health")
def health_check():
    return {"status": "ok"}


# 2. Input schema
class PaddyInput(BaseModel):
    hectares: float
    micronutrients_70days: float
    potassh_50days: float
    urea_40days: float
    pest_60day: float
    lp_mainfield: float
    dap_20days: float
    trash: float
    seedrate: float
    lp_nurseryarea: float
    weed28d_thiobencarb: float
    nursery_area_cents: float


# 3. Prediction endpoint
@app.post("/predict")
def predict(data: PaddyInput):

    input_data = pd.DataFrame([{
        "Hectares ": data.hectares,
        "Micronutrients_70Days": data.micronutrients_70days,
        "Potassh_50Days": data.potassh_50days,
        "Urea_40Days": data.urea_40days,
        "Pest_60Day(in ml)": data.pest_60day,
        "LP_Mainfield(in Tonnes)": data.lp_mainfield,
        "DAP_20days": data.dap_20days,
        "Trash(in bundles)": data.trash,
        "Seedrate(in Kg)": data.seedrate,
        "LP_nurseryarea(in Tonnes)": data.lp_nurseryarea,
        "Weed28D_thiobencarb": data.weed28d_thiobencarb,
        "Nursery area (Cents)": data.nursery_area_cents
    }])

    input_data = input_data[feature_columns]

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    return {
        "prediction_kg": float(prediction[0])
    }