import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# 1. LOAD & CLEAN
df = pd.read_csv("biomimetic_simml_dataset.csv")
def rescale(s, mi, ma): return (s - s.min()) / (s.max() - s.min() + 1e-9) * (ma - mi) + mi

# 2. APPLY YOUR DESIGN SPECIFICATIONS
df["velocity"] = rescale(df["velocity"], 0, 25)
df["riblet_height"] = rescale(df["riblet_height"], 0.01, 0.3)
df["riblet_spacing"] = rescale(df["riblet_spacing"], 0.05, 1.0)
df["lotus_intensity"] = rescale(df["lotus_intensity"], 0.1, 0.9) # 0.9 = 100nm, 0.1 = 1000nm
df["durability"] = rescale(df["durability"].abs(), 30, 98)
df["drag_reduction"] = rescale(df["drag_reduction"], 0, 22)

# 3. PHYSICS ENGINEERING
df["aspect_ratio"] = df["riblet_height"] / (df["riblet_spacing"] + 1e-6)
df["solid_fraction_f"] = 1 / (1 + df["aspect_ratio"])
df["multiscale_index"] = df["lotus_intensity"] * (1 - (df["riblet_height"] * 0.1))
df["velocity_riblet_interact"] = df["velocity"] * df["aspect_ratio"]
df["cos_theta_eff"] = -1 + df["solid_fraction_f"] * (np.cos(np.radians(110)) + 1)
df["effective_contact_angle"] = np.degrees(np.arccos(np.clip(df["cos_theta_eff"], -1, 1)))
df["estimated_slip_length"] = (1 / (np.sqrt(df["solid_fraction_f"]) + 1e-6)) * df["riblet_spacing"]

# 4. TRAIN & SAVE
feature_columns = [
    "riblet_height", "riblet_spacing", "lotus_intensity", "velocity",
    "temperature", "salinity", "time", "material", "coating",
    "aspect_ratio", "multiscale_index", "velocity_riblet_interact",
    "solid_fraction_f", "cos_theta_eff", "effective_contact_angle",
    "estimated_slip_length"
]

X = df[feature_columns].fillna(0)
y = df["drag_reduction"].fillna(0)

model = RandomForestRegressor(n_estimators=145, max_depth=15, random_state=30)
model.fit(X, y)

joblib.dump(model, "shs_predictive_model.pkl")
joblib.dump(feature_columns, "feature_columns.pkl")
df.to_csv("biomimetic_simml_dataset.csv", index=False)
