import numpy as np
import pandas as pd

np.random.seed(42)
data = []

for _ in range(2000):
    # --- 1. Basic Inputs (Stay the same) ---
    riblet_height = np.random.uniform(0.01, 0.3)
    riblet_spacing = np.random.uniform(0.05, 1.0)
    lotus_intensity = np.random.uniform(0, 1)
    velocity = np.random.uniform(0.5, 12)
    temperature = np.random.uniform(0, 40)
    salinity = np.random.uniform(10, 40)
    time = np.random.uniform(1, 1095)
    material = np.random.randint(0, 3)
    coating = np.random.randint(0, 5)

    # --- 2. Existing Logic (Stay the same) ---
    material_factor = [0.75, 0.9, 1.0][material]
    coating_factor = [0.8, 0.88, 1.0, 1.0, 0.95][coating]
    riblet_effect = riblet_height / (riblet_spacing + 1e-6)
    
    drag_reduction = (60 * riblet_effect + 0.4 * velocity + np.random.normal(0, 2))
    biofouling = (0.04 * time + 0.03 * temperature + 0.02 * salinity - 55 * lotus_intensity + np.random.normal(0, 2))
    hydrophobicity = (100 * lotus_intensity - 0.2 * temperature + np.random.normal(0, 2))
    durability = (100 - 25 * (1 - material_factor) * 100 - 25 * (1 - coating_factor) * 100 + np.random.normal(0, 3))

    # --- 3. NEW: Refined Physics Columns (ADD THESE) ---
    aspect_ratio = riblet_height / riblet_spacing
    multiscale_index = lotus_intensity * (1 - (riblet_height * 0.1)) # Example multiscale logic
    velocity_riblet_interact = velocity * riblet_effect
    
    # SHS Surface Science calculations
    solid_fraction_f = 1 / (1 + aspect_ratio)
    cos_theta_eff = -1 + solid_fraction_f * (np.cos(np.radians(110)) + 1)
    effective_contact_angle = np.degrees(np.arccos(np.clip(cos_theta_eff, -1, 1)))
    estimated_slip_length = (1 / (solid_fraction_f**0.5)) * riblet_spacing
    
    # Classification logic
    is_shs = 1 if effective_contact_angle > 145 else 0
    better_than_painting = 1 if drag_reduction > 15 else 0
    design_category = "Multi-scale" if multiscale_index > 0.6 else "Shark-like"

    # --- 4. Append ALL 25 columns ---
    data.append([
        riblet_height, riblet_spacing, lotus_intensity, velocity, temperature, salinity, time,
        material, coating, drag_reduction, biofouling, hydrophobicity, durability,
        aspect_ratio, multiscale_index, velocity_riblet_interact, drag_reduction, # composite/clipped proxies
        drag_reduction, is_shs, better_than_painting, design_category,
        solid_fraction_f, cos_theta_eff, effective_contact_angle, estimated_slip_length
    ])

# Define the full 25 column names
cols = [
    "riblet_height", "riblet_spacing", "lotus_intensity", "velocity", "temperature", "salinity", "time",
    "material", "coating", "drag_reduction", "biofouling", "hydrophobicity", "durability",
    "aspect_ratio", "multiscale_index", "velocity_riblet_interact", "composite_performance", 
    "drag_reduction_clipped", "is_shs", "better_than_painting", "design_category", 
    "solid_fraction_f", "cos_theta_eff", "effective_contact_angle", "estimated_slip_length"
]

df = pd.DataFrame(data, columns=cols)
df.to_csv("biomimetic_dataset_physics_refined.csv", index=False)
print("Refined Dataset generated with 25 columns")
