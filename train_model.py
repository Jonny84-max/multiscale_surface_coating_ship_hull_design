import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# ==============================
# 1. LOAD DATA
# ==============================
df = pd.read_csv("biomimetic_dataset_physics_refined.csv")

# ==============================
# 2. CLEAN DATA
# ==============================
df = df.apply(pd.to_numeric, errors="coerce")
df = df.fillna(0)

# ==============================
# 3. ENCODE CATEGORICAL VARIABLES
# ==============================
df["material"] = df["material"].map({"GFRP": 0, "CFRP": 1, "Hybrid": 2})
df["coating"] = df["coating"].map({
    "Epoxy": 0, "Vinyl": 1, "PDMS": 2, "Fluoro": 3, "Sol-gel": 4
})

# ==============================
# 4. FEATURE ENGINEERING (MATCH STREAMLIT EXACTLY)
# ==============================
df["aspect_ratio"] = df["riblet_height"] / (df["riblet_spacing"] + 1e-6)
df["solid_fraction_f"] = 1 / (1 + df["aspect_ratio"])
df["multiscale_index"] = df["lotus_intensity"] * (1 - (df["riblet_height"] * 0.1))
df["velocity_riblet_interact"] = df["velocity"] * df["aspect_ratio"]

df["cos_theta_eff"] = -1 + df["solid_fraction_f"] * (
    np.cos(np.radians(110)) + 1
)

df["effective_contact_angle"] = np.degrees(
    np.arccos(np.clip(df["cos_theta_eff"], -1, 1))
)

df["estimated_slip_length"] = (
    1 / (np.sqrt(df["solid_fraction_f"]) + 1e-6)
) * df["riblet_spacing"]

# ==============================
# 5. FEATURE LIST (STRICT CHECK)
# ==============================
feature_cols = [
    "riblet_height", "riblet_spacing", "lotus_intensity",
    "velocity", "temperature", "salinity", "time",
    "material", "coating",
    "aspect_ratio", "multiscale_index",
    "velocity_riblet_interact",
    "solid_fraction_f",
    "cos_theta_eff",
    "effective_contact_angle",
    "estimated_slip_length"
]

missing = [c for c in feature_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in dataset: {missing}")

# ==============================
# 6. SPLIT DATA
# ==============================
X = df[feature_cols]
y = df["drag_reduction"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 7. TRAIN MODEL
# ==============================
model = RandomForestRegressor(
    n_estimators=150,
    random_state=18,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ==============================
# 8. EVALUATION
# ==============================
preds = model.predict(X_test)

print("\n--- MODEL PERFORMANCE ---")
print("R2 Score:", r2_score(y_test, preds))
print("MAE:", mean_absolute_error(y_test, preds))

# ==============================
# 9. SAVE EVERYTHING (STREAMLIT SAFE)
# ==============================
joblib.dump(model, "shs_predictive_model.pkl")
joblib.dump(feature_cols, "feature_columns.pkl")

print("\nSaved model + feature list successfully.")