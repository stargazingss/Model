from pathlib import Path
import joblib
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

from preprocessing import preprocess_data


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


X_train_scaled, X_test_scaled, y_train, y_test, scaler_final, feature_columns = preprocess_data()


kf = KFold(n_splits=5, shuffle=True, random_state=42)


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


print(f"Best params RF: {grid_rf.best_params_}")
print(f"Best params KNN: {grid_knn.best_params_}")
print("Models and preprocessing artifacts saved successfully.")