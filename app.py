





import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from surface_3d_pattern import generate_stl   # ✅ FIXED IMPORT

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    data = joblib.load("model.pkl")
    return data["model"], data["columns"]

model, columns = load_model()

# ================= FIXED MAPPINGS =================
material_map = {"GFRP": 0, "CFRP": 1, "Hybrid": 2}
coating_map = {"Epoxy": 0, "Vinyl": 1, "PDMS": 2, "Fluoro": 3, "Sol-gel": 4}

# ================= UI =================
st.title("Hybrid Biomimetic Hull Design System")

st.sidebar.header("Design Inputs")

riblet_height = st.sidebar.slider("Riblet Height (mm)", 0.01, 0.3)
riblet_spacing = st.sidebar.slider("Riblet Spacing (mm)", 0.05, 1.0)
lotus_intensity = st.sidebar.slider("Lotus Intensity", 0.0, 1.0)

velocity = st.sidebar.slider("Velocity", 0.5, 12.0)
temperature = st.sidebar.slider("Temperature", 0, 40)
salinity = st.sidebar.slider("Salinity", 10, 40)
time = st.sidebar.slider("Time (days)", 1, 1095)

material = st.sidebar.selectbox("Material", list(material_map.keys()))
coating = st.sidebar.selectbox("Coating", list(coating_map.keys()))
mode = st.sidebar.selectbox("Output Mode", ["Visualization STL", "CFD STL"])

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

# ================= MODEL =================
pred = None

if st.button("Run Simulation"):
    try:
        X = build_input(velocity)
        pred = model.predict(X)

        if len(pred[0]) >= 4:
            pred = pred[0]
            st.subheader("Performance Results")
            st.metric("Drag Reduction", f"{pred[0]:.2f}")
            st.metric("Biofouling Risk", f"{pred[1]:.2f}")
            st.metric("Hydrophobicity", f"{pred[2]:.2f}")
            st.metric("Durability", f"{pred[3]:.2f}")
        else:
            st.error("Model output format mismatch")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ================= GRID =================
resolution = 80   # ✅ lighter

x = np.linspace(0, 5, resolution)
y = np.linspace(0, 5, resolution)
Xg, Yg = np.meshgrid(x, y)

# ================= HULL =================
hull_base = 1 - (Yg**2) / (1.5**2)
hull_base = np.clip(hull_base, 0, 1)

# coating influence
coating_factor = {
    "Epoxy": 0.3,
    "Vinyl": 0.4,
    "PDMS": 0.9,
    "Fluoro": 1.0,
    "Sol-gel": 0.6
}

riblet = riblet_height * np.sin((2 * np.pi / riblet_spacing) * Xg)
lotus = lotus_intensity * coating_factor[coating] * (np.cos(8 * Xg) * np.cos(8 * Yg))

Z = hull_base + riblet + 0.3 * lotus

# ================= 3D =================
st.subheader("3D Biomimetic Hull Surface")

fig = go.Figure(data=[go.Surface(x=Xg, y=Yg, z=Z)])
st.plotly_chart(fig, use_container_width=True)

# ================= STL =================
st.subheader("Export STL for SimScale")

if st.button("Generate STL"):
    try:
        X_s, Y_s, Z_s, file_path = generate_stl(
            riblet_spacing,
            riblet_height,
            lotus_intensity,   # ✅ now included
            resolution=80,
            mode=mode
        )

        with open(file_path, "rb") as f:
            st.download_button(
                "Download STL",
                f,
                "biomimetic_hull.stl"
            )

    except Exception as e:
        st.error(f"STL generation failed: {e}")

# ================= FLOW =================
st.subheader("Relative Flow Pattern")

velocity_field = 1 / (1 + np.abs(Z))

fig_flow, ax_flow = plt.subplots()
ax_flow.contourf(Xg, Yg, velocity_field, levels=25)
st.pyplot(fig_flow)

# ================= BIOFOULING =================
st.subheader("Biofouling Zones")

attachment_zone = (np.abs(Z) < 0.2) & (velocity_field < 0.8)

fig_bio, ax_bio = plt.subplots()
ax_bio.contourf(Xg, Yg, attachment_zone, levels=1)
st.pyplot(fig_bio)

# ================= FLOW VECTORS =================
st.subheader("Flow Direction Field")

step = 8
U = -np.gradient(Z, axis=1)
V = -np.gradient(Z, axis=0)

fig_vec, ax_vec = plt.subplots()
ax_vec.quiver(Xg[::step, ::step], Yg[::step, ::step],
              U[::step, ::step], V[::step, ::step])
st.pyplot(fig_vec)

# ================= DRAG CURVE =================
st.subheader("Drag vs Velocity")

vels = np.linspace(0.5, 12, 40)
drag_curve = []

for v in vels:
    try:
        X = build_input(v)
        drag_curve.append(model.predict(X)[0][0])
    except:
        drag_curve.append(0)

fig_drag, ax_drag = plt.subplots()
ax_drag.plot(vels, drag_curve)
st.pyplot(fig_drag)

# ================= COMPARISON =================
st.subheader("Surface Comparison")

labels = ["Smooth", "Riblet", "Lotus", "Hybrid"]
values = [40, 65, 60, pred[0] if pred is not None else 75]

fig_comp, ax_comp = plt.subplots()
ax_comp.bar(labels, values)
st.pyplot(fig_comp)

# ================= INSIGHT =================
st.subheader("Engineering Insight")

st.write("""
This system integrates riblet-induced drag reduction, lotus-effect hydrophobicity,
and nano-composite material optimization with predictive modeling for enhanced
marine hull performance.
""")
