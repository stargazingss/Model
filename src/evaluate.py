from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocessing import preprocess_data


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"


X_train_scaled, X_test_scaled, y_train, y_test, scaler_final, feature_columns = preprocess_data()


rf_model = joblib.load(
    MODEL_DIR / "best_random_forest_model.pkl"
)

knn_model = joblib.load(
    MODEL_DIR / "knn_model.pkl"
)

lr_model = joblib.load(
    MODEL_DIR / "linear_regression_model.pkl"
)


models_final = {
    "Optimized Random Forest": rf_model,
    "Optimized KNN Regressor": knn_model,
    "Linear Regression (Baseline)": lr_model
}


performance_results = []


for name, model in models_final.items():
    y_pred = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    abs_errors = np.abs(y_test - y_pred)
    p90 = np.percentile(abs_errors, 90)

    if name == "Optimized Random Forest":
        best_params = {
            "max_depth": model.max_depth,
            "n_estimators": model.n_estimators
        }
    elif name == "Optimized KNN Regressor":
        best_params = {
            "n_neighbors": model.n_neighbors,
            "weights": model.weights
        }
    else:
        best_params = "N/A"

    performance_results.append({
        "Model Name": name,
        "Best Params": str(best_params),
        "MAE (Kg)": round(mae, 2),
        "P90 Error (Kg)": round(p90, 2),
        "RMSE (Kg)": round(rmse, 2),
        "R2 Score": round(r2, 4)
    })


df_report = pd.DataFrame(performance_results)


print("=== PERBANDINGAN MODEL ===")
print(df_report.to_string(index=False))