import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("biomimetic_dataset.csv")

X = df.drop(columns=["drag_reduction", "biofouling", "hydrophobicity", "durability"])
y = df[["drag_reduction", "biofouling", "hydrophobicity", "durability"]]

model = RandomForestRegressor(n_estimators=80, random_state=42)
model.fit(X, y)

joblib.dump({
    "model": model,
    "columns": X.columns.tolist()
}, "model.pkl")

print("DONE - clean model saved")
