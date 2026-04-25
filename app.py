import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import time
from surface_3d_pattern import generate_stl

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    data = joblib.load("model.pkl")
    return data["model"], data["columns"]

model, columns = load_model()

# ================= MAPPINGS =================
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
time_days = st.sidebar.slider("Time (days)", 1, 365)

material = st.sidebar.selectbox("Material", list(material_map.keys()))
coating = st.sidebar.selectbox("Coating", list(coating_map.keys()))
mode = st.sidebar.selectbox("Output Mode", ["Visualization STL", "CFD STL"])

run_sim = st.sidebar.checkbox("Run Time Simulation")
speed = st.sidebar.slider("Simulation Speed (sec per step)", 0.1, 1.5, 0.3)

time_display = st.empty()

# ================= FEATURE BUILDER =================
def build_input(v):
    input_dict = {
        "riblet_height": riblet_height,
        "riblet_spacing": riblet_spacing,
        "lotus_intensity": lotus_intensity,
        "velocity": v,
        "temperature": temperature,
        "salinity": salinity,
        "time": 0,
        "material": material_map[material],
        "coating": coating_map[coating]
    }

    df = pd.DataFrame([input_dict])
    return df[columns]  # ensures correct order

# ================= SIMULATION =================
if st.button("Run Simulation"):

    try:
        progress = st.progress(0)

        cumulative_bio = 0

        for t in range(1, time_days + 1):

            time_display.write(f"Simulation Day: {t}")
            progress.progress(t / time_days)

            X = build_input(velocity)
            X.loc[0, "time"] = t

            pred = model.predict(X)[0]

            drag = max(pred[0], 0)
            daily_bio = np.clip(pred[1], 0, 1)

            # cumulative biofouling growth
            cumulative_bio += daily_bio * 0.05
            bio = min(cumulative_bio, 1)

            hydro = max(pred[2], 0)
            durability = max(pred[3], 0)

            st.subheader("Performance Results")

            st.metric(
                "Drag Reduction",
                f"{drag:.2f} %",
                delta=f"{drag - 50:.2f}% vs baseline"
            )

            st.metric("Biofouling Accumulation", f"{bio:.2f} (0–1)")
            st.metric("Hydrophobicity (Contact Angle)", f"{hydro:.2f} °")
            st.metric("Durability Index", f"{durability:.2f}")

            if run_sim:
                time.sleep(speed)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ================= BIOFOULING OVER TIME =================
st.subheader("Cumulative Biofouling Growth")

times = np.arange(1, time_days + 1)
bio_curve = []
cumulative = 0

for t in times:
    try:
        X = build_input(velocity)
        X.loc[0, "time"] = t

        pred_vals = model.predict(X)
        daily_bio = np.clip(pred_vals[0][1], 0, 1)

        cumulative += daily_bio * 0.05
        bio_curve.append(min(cumulative, 1))

    except:
        bio_curve.append(0)

fig_time, ax_time = plt.subplots()
ax_time.plot(times, bio_curve)
ax_time.set_title("Cumulative Biofouling Risk Over Time")
ax_time.set_xlabel("Days")
ax_time.set_ylabel("Accumulated Risk (0–1)")
st.pyplot(fig_time)

# ================= MULTISCALE SURFACE =================
resolution = 100
x = np.linspace(0, 5, resolution)
y = np.linspace(0, 5, resolution)
Xg, Yg = np.meshgrid(x, y)

hull_base = 1 - (Yg**2) / (1.5**2)
hull_base = np.clip(hull_base, 0, 1)

riblet = riblet_height * np.sin((2 * np.pi / riblet_spacing) * Xg)

nano_freq = 40
nano_amp = 0.08 * lotus_intensity

np.random.seed(1)
noise = 0.02 * np.random.randn(*Xg.shape)

lotus = nano_amp * (np.cos(nano_freq * Xg) * np.cos(nano_freq * Yg)) + noise

Z = hull_base + riblet + lotus

# ================= 3D =================
st.subheader("3D Biomimetic Surface")

fig = go.Figure(data=[go.Surface(x=Xg, y=Yg, z=Z)])
st.plotly_chart(fig, width='stretch')

# ================= STL =================
st.subheader("Export STL")

if st.button("Generate STL"):
    try:
        X_s, Y_s, Z_s, file_path = generate_stl(
            riblet_spacing,
            riblet_height,
            lotus_intensity,
            resolution=80,
            mode=mode
        )

        with open(file_path, "rb") as f:
            st.download_button("Download STL", f, "biomimetic_hull.stl")

    except Exception as e:
        st.error(f"STL generation failed: {e}")

# ================= FLOW =================
st.subheader("Flow Interaction")

dZdx = np.gradient(Z, axis=1)
dZdy = np.gradient(Z, axis=0)

U = 1 - np.abs(dZdx) * 2
V = -dZdy * 0.5

velocity_field = np.sqrt(U**2 + V**2)

fig_flow, ax_flow = plt.subplots()
ax_flow.contourf(Xg, Yg, velocity_field, levels=30)
st.pyplot(fig_flow)

# ================= INSIGHT =================
st.subheader("Engineering Insight")

st.write("""
This system integrates riblet drag reduction, lotus-inspired nano-textures,
and machine learning prediction to design intelligent marine hull coatings.
""")
if st.button("Show Engineering Interpretation"):
    st.write("""
    - Riblet structures reduce turbulent drag by controlling boundary layer flow.
    - Lotus-inspired nano-textures enhance hydrophobicity and reduce adhesion.
    - Combined system improves antifouling performance and fuel efficiency.
    - Predictive ML enables time-dependent biofouling estimation.
    """)
