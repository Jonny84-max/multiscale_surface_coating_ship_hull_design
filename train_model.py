import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

# load dataset (must be inside this folder)
df = pd.read_csv("biomimetic_dataset.csv")

X = df.drop(columns=["drag_reduction", "biofouling", "hydrophobicity", "durability"])
y = df[["drag_reduction", "biofouling", "hydrophobicity", "durability"]]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(model, "model.pkl")

print("Model trained successfully")
