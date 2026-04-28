import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor

# 1. SETUP PARAMETERS
n_samples = 5000
np.random.seed(42)

# --- Generate Input Ranges (Aligned with your Design) ---
velocity = np.random.uniform(0, 25, n_samples)
temp = np.random.uniform(2, 35, n_samples)
salinity = np.random.uniform(15, 40, n_samples)
time_days = np.random.uniform(1, 1095, n_samples)
riblet_h = np.random.uniform(0.01, 0.3, n_samples)
riblet_s = np.random.uniform(0.05, 1.0, n_samples)
lotus_intensity = np.random.uniform(0.1, 0.9, n_samples) # 0.9=100nm, 0.1=1000nm
material = np.random.randint(0, 3, n_samples)
coating = np.random.randint(0, 5, n_samples)

# --- Physics Calculations ---
aspect_ratio = riblet_h / (riblet_s + 1e-6)
solid_fraction_f = 1 / (1 + aspect_ratio)

# Cassie-Baxter Contact Angle
cos_theta_eff = -1 + solid_fraction_f * (np.cos(np.radians(110)) + 1)
eff_contact_angle = np.degrees(np.arccos(np.clip(cos_theta_eff, -1, 1)))

# Secondary features
multiscale_idx = lotus_intensity * (1 - (riblet_h * 0.1))
velocity_interact = velocity * aspect_ratio
slip_length = (1 / (np.sqrt(solid_fraction_f) + 1e-6)) * riblet_s

# --- Target Calculation (Drag Reduction %) ---
# Base physics: Riblets (max ~10%) + SHS/Lotus (max ~12%)
dr_riblet = 8 * np.sin(np.clip(aspect_ratio, 0, 1) * np.pi/2)
dr_shs = 12 * (eff_contact_angle - 90) / 90
if_shs_bonus = np.where(eff_contact_angle > 150, 1.2, 1.0) # Bonus for SHS state

# Environmental modifiers
temp_factor = 1 + (temp - 20) / 200
salinity_factor = 1 - (salinity - 35) / 500

# Degradation modifier
wear_rate = (velocity * 0.05 + 0.1) / 100
durability = np.clip(100 - (time_days * wear_rate), 0, 100)
time_factor = 0.6 + 0.4 * (durability / 100)

# Final Formula
drag_reduction = (dr_riblet + (dr_shs * if_shs_bonus)) * temp_factor * salinity_factor * time_factor
drag_reduction = np.clip(drag_reduction, 0, 25) # Cap at physical limit

# 2. CREATE DATASET
df = pd.DataFrame({
    "riblet_height": riblet_h, "riblet_spacing": riblet_s, "lotus_intensity": lotus_intensity,
    "velocity": velocity, "temperature": temp, "salinity": salinity, "time": time_days,
    "material": material, "coating": coating, "aspect_ratio": aspect_ratio,
    "multiscale_index": multiscale_idx, "velocity_riblet_interact": velocity_interact,
    "solid_fraction_f": solid_fraction_f, "cos_theta_eff": cos_theta_eff,
    "effective_contact_angle": eff_contact_angle, "estimated_slip_length": slip_length,
    "durability": durability, "drag_reduction": drag_reduction
})

# 3. DEFINE FEATURES & TRAIN
feature_columns = [
    "riblet_height", "riblet_spacing", "lotus_intensity", "velocity",
    "temperature", "salinity", "time", "material", "coating",
    "aspect_ratio", "multiscale_index", "velocity_riblet_interact",
    "solid_fraction_f", "cos_theta_eff", "effective_contact_angle",
    "estimated_slip_length"
]

X = df[feature_columns]
y = df["drag_reduction"]

model = RandomForestRegressor(n_estimators=130, max_depth=10, random_state=25)
model.fit(X, y)

# 4. SAVE ASSETS
joblib.dump(model, "shs_predictive_model.pkl")
joblib.dump(feature_columns, "feature_columns.pkl")
df.to_csv("biomimetic_opsimml_dataset.csv", index=False)

print("Training Complete. Model is now optimized for Temp, Salinity, and SHS Physics.")
