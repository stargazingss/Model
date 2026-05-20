import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

# contoh dummy dataset
data = {
    "hectares": [1,2,3,4,5],
    "rainfall": [100,120,130,140,150],
    "humidity": [70,72,75,78,80],
    "yield": [2000,2500,3000,3500,4000]
}

df = pd.DataFrame(data)

X = df[["hectares", "rainfall", "humidity"]]
y = df["yield"]

# train model
model = RandomForestRegressor()
model.fit(X, y)

# simpan model
pickle.dump(model, open("model.pkl", "wb"))

print("model.pkl berhasil dibuat")