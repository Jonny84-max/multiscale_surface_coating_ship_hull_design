import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ================= LOAD MODEL =================
data = joblib.load("model.pkl")
model = data["model"]
columns = data["columns"]

st.title("Hybrid Biomimetic Hull Design System")

# ================= INPUT =================
st.sidebar.header("Design Inputs")

riblet_height = st.sidebar.slider("Riblet Height (mm)", 0.01, 0.3)
riblet_spacing = st.sidebar.slider("Riblet Spacing (mm)", 0.05, 1.0)
lotus_intensity = st.sidebar.slider("Lotus Intensity", 0.0, 1.0)

velocity = st.sidebar.slider("Velocity", 0.5, 12.0)
temperature = st.sidebar.slider("Temperature", 0, 40)
salinity = st.sidebar.slider("Salinity", 10, 40)
time = st.sidebar.slider("Time (days)", 1, 1095)

material = st.sidebar.selectbox("Material", ["GFRP", "CFRP", "Hybrid"])
coating = st.sidebar.selectbox("Coating", ["Epoxy", "Vinyl", "PDMS", "Fluoro", "Sol-gel"])

material_map = {"GFRP": 0, "CFRP": 1, "Hybrid": 2}
coating_map = {"Epoxy": 0, "Vinyl": 1, "PDMS": 2, "Fluoro": 3, "Sol-gel": 4}

# ================= FEATURE BUILDER =================
def build_input(v):
    input_dict = {
        "riblet_height": riblet_height,
        "riblet_spacing": riblet_spacing,
        "lotus_intensity": lotus_intensity,
        "velocity": v,
        "temperature": temperature,
        "salinity": salinity,
        "time": time,
        "material": material_map[material],
        "coating": coating_map[coating]
    }
    return np.array([[input_dict[col] for col in columns]])

# ================= PREDICTION =================
if st.button("Run Simulation"):

    X = build_input(velocity)
    pred = model.predict(X)[0]

    st.subheader("Performance Results")

    st.metric("Drag Reduction", f"{pred[0]:.2f}")
    st.metric("Biofouling Risk", f"{pred[1]:.2f}")
    st.metric("Hydrophobicity", f"{pred[2]:.2f}")
    st.metric("Durability", f"{pred[3]:.2f}")

# ================= SURFACE (REALISTIC HULL SHAPE) =================
st.subheader("Surface Microstructure (Biomimetic Hull)")

x = np.linspace(-3, 3, 140)
y = np.linspace(-1.5, 1.5, 140)
Xg, Yg = np.meshgrid(x, y)

# hull geometry (curved structure)
hull_base = 1 - (Yg**2) / (1.5**2)
hull_base = np.clip(hull_base, 0, 1)

# biomimetic textures
riblet_surface = riblet_height * np.sin(12 * Xg) * np.cos(3 * Yg)
lotus_surface = lotus_intensity * np.random.normal(0, 0.02, Xg.shape)

Z = hull_base + riblet_surface + lotus_surface

fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(Xg, Yg, Z, cmap="viridis", edgecolor="none")
ax.set_title("Biomimetic Hull Surface (Riblet + Lotus)")
st.pyplot(fig)

# ================= FLOW FIELD =================
st.subheader("Flow Field (CFD-style)")

velocity_field = 1 / (1 + np.abs(Z))

fig2, ax2 = plt.subplots()
ax2.contourf(Xg, Yg, velocity_field, levels=25)
st.pyplot(fig2)

# ================= DRAG CURVE =================
st.subheader("Drag vs Velocity")

vels = np.linspace(0.5, 12, 40)
drag_curve = []

for v in vels:
    X = build_input(v)
    drag_curve.append(model.predict(X)[0][0])

fig3, ax3 = plt.subplots()
ax3.plot(vels, drag_curve)
ax3.set_xlabel("Velocity")
ax3.set_ylabel("Drag Reduction")
st.pyplot(fig3)

# ================= COMPARISON =================
st.subheader("Surface Performance Comparison")

labels = ["Smooth Hull", "Riblet Surface", "Lotus Surface", "Hybrid Biomimetic"]

smooth = 40
riblet_val = 65
lotus_val = 60
hybrid = pred[0] if "pred" in locals() else 75

values = [smooth, riblet_val, lotus_val, hybrid]

fig4, ax4 = plt.subplots()
ax4.bar(labels, values)
ax4.set_ylabel("Performance Index")
st.pyplot(fig4)

# ================= INSIGHT =================
st.subheader("Engineering Insight")

st.write("""
This system integrates riblet-based drag reduction, lotus-inspired hydrophobicity,
and composite material selection to optimize hull performance under marine conditions.
""")
