from pathlib import Path
import joblib
import mlflow
import numpy as np

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocessing import preprocess_data


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
MLFLOW_DB = BASE_DIR / "mlflow.db"

MODEL_DIR.mkdir(exist_ok=True)

mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
mlflow.set_experiment("Paddy Yield Prediction")


X_train_scaled, X_test_scaled, y_train, y_test, scaler_final, feature_columns = preprocess_data()


kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


param_rf = {
    "n_estimators": [50, 100, 150],
    "max_depth": [None, 10, 20]
}


param_knn = {
    "n_neighbors": [3, 5, 7],
    "weights": ["uniform", "distance"]
}


grid_rf = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_rf,
    cv=kf,
    scoring="r2",
    n_jobs=-1
)

grid_rf.fit(X_train_scaled, y_train)


grid_knn = GridSearchCV(
    KNeighborsRegressor(),
    param_knn,
    cv=kf,
    scoring="r2",
    n_jobs=-1
)

grid_knn.fit(X_train_scaled, y_train)


lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)


rf_model = grid_rf.best_estimator_
knn_model = grid_knn.best_estimator_


models = {
    "Random Forest": rf_model,
    "KNN": knn_model,
    "Linear Regression": lr_model
}


for name, model in models.items():

    with mlflow.start_run(run_name=name):

        y_pred = model.predict(X_test_scaled)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        abs_errors = np.abs(y_test - y_pred)
        p90 = np.percentile(abs_errors, 90)

        mlflow.log_param("model_type", name)
        mlflow.log_param("n_features", X_train_scaled.shape[1])
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("cv_folds", 5)

        if name == "Random Forest":
            mlflow.log_param("n_estimators", model.n_estimators)
            mlflow.log_param("max_depth", model.max_depth)

        elif name == "KNN":
            mlflow.log_param("n_neighbors", model.n_neighbors)
            mlflow.log_param("weights", model.weights)

        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)
        mlflow.log_metric("P90_Error", p90)

        model_path = MODEL_DIR / f"{name.lower().replace(' ', '_')}_model.pkl"

        joblib.dump(
            model,
            model_path
        )

        mlflow.log_artifact(
            str(model_path),
            artifact_path="model"
        )

        print(f"\n{name}")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R2: {r2:.4f}")
        print(f"P90 Error: {p90:.2f}")


joblib.dump(
    rf_model,
    MODEL_DIR / "best_random_forest_model.pkl"
)

joblib.dump(
    knn_model,
    MODEL_DIR / "knn_model.pkl"
)

joblib.dump(
    lr_model,
    MODEL_DIR / "linear_regression_model.pkl"
)

joblib.dump(
    scaler_final,
    MODEL_DIR / "robust_standard_scaler.pkl"
)

joblib.dump(
    feature_columns,
    MODEL_DIR / "feature_columns.pkl"
)


print(f"\nBest params RF: {grid_rf.best_params_}")
print(f"Best params KNN: {grid_knn.best_params_}")
print("Models and MLflow runs saved successfully.")