import numpy as np
import pandas as pd

np.random.seed(42)

data = []

for _ in range(2000):

    riblet_height = np.random.uniform(0.01, 0.3)
    riblet_spacing = np.random.uniform(0.05, 1.0)
    lotus_intensity = np.random.uniform(0, 1)

    velocity = np.random.uniform(0.5, 12)
    temperature = np.random.uniform(0, 40)
    salinity = np.random.uniform(10, 40)
    time = np.random.uniform(1, 1095)

    material = np.random.randint(0, 3)
    coating = np.random.randint(0, 5)

    material_factor = [0.75, 0.9, 1.0][material]
    coating_factor = [0.8, 0.88, 1.0, 1.0, 0.95][coating]

    riblet_effect = riblet_height / (riblet_spacing + 1e-6)

    drag_reduction = (
        60 * riblet_effect +
        0.4 * velocity +
        np.random.normal(0, 2)
    )

    biofouling = (
        0.04 * time +
        0.03 * temperature +
        0.02 * salinity -
        55 * lotus_intensity +
        np.random.normal(0, 2)
    )

    hydrophobicity = (
        100 * lotus_intensity -
        0.2 * temperature +
        np.random.normal(0, 2)
    )

    durability = (
        100 -
        25 * (1 - material_factor) * 100 -
        25 * (1 - coating_factor) * 100 +
        np.random.normal(0, 3)
    )

    data.append([
        riblet_height, riblet_spacing, lotus_intensity,
        velocity, temperature, salinity, time,
        material, coating,
        drag_reduction, biofouling, hydrophobicity, durability
    ])

df = pd.DataFrame(data, columns=[
    "riblet_height", "riblet_spacing", "lotus_intensity",
    "velocity", "temperature", "salinity", "time",
    "material", "coating",
    "drag_reduction", "biofouling", "hydrophobicity", "durability"
])

df.to_csv("biomimetic_dataset.csv", index=False)

print("Dataset generated")
