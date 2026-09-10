import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_data():
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_PATH = BASE_DIR / "data" / "paddydataset.csv"

    df = pd.read_csv(DATA_PATH)

    print(f"Jumlah missing values sebelum cleaning: {df.isnull().sum().sum()}")

    df = df.dropna()
    df = df.drop_duplicates()

    print(f"Dimensi dataset setelah cleaning: {df.shape}")
    print(f"Jumlah missing values setelah cleaning: {df.isnull().sum().sum()}")

    selected_features = [
        'Hectares ',
        'Micronutrients_70Days',
        'Potassh_50Days',
        'Urea_40Days',
        'Pest_60Day(in ml)',
        'LP_Mainfield(in Tonnes)',
        'DAP_20days',
        'Trash(in bundles)',
        'Seedrate(in Kg)',
        'LP_nurseryarea(in Tonnes)',
        'Weed28D_thiobencarb',
        'Nursery area (Cents)'
    ]

    target = 'Paddy yield(in Kg)'

    X_final = df[selected_features]
    y_final = df[target]

    X_encoded_final = pd.get_dummies(X_final, drop_first=True)

    print(f"Jumlah fitur setelah encoding: {X_encoded_final.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded_final,
        y_final,
        test_size=0.2,
        random_state=42
    )

    scaler_final = StandardScaler()

    X_train_scaled = scaler_final.fit_transform(X_train)
    X_test_scaled = scaler_final.transform(X_test)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    feature_columns = X_encoded_final.columns.tolist()

    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler_final,
        feature_columns
    )