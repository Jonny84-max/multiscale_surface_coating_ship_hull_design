import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import time as time_lib  # Alias to prevent conflict with 'time' variable
from surface_3d_pattern import generate_stl

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    # Ensure model.pkl is in the same directory
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
days_input = st.sidebar.slider("Time (days)", 1, 1095) # Renamed variable

material = st.sidebar.selectbox("Material", list(material_map.keys()))
coating = st.sidebar.selectbox("Coating", list(coating_map.keys()))
mode = st.sidebar.selectbox("Output Mode", ["Visualization STL", "CFD STL"])

time_display = st.empty()
run_sim = st.sidebar.checkbox("Run Time Simulation")
speed = st.sidebar.slider("Simulation Speed (sec per step)", 0.1, 2.0, 0.5)


# ================= FEATURE BUILDER =================
def build_input(v, t_val):
    input_dict = {
        "riblet_height": riblet_height,
        "riblet_spacing": riblet_spacing,
        "lotus_intensity": lotus_intensity,
        "velocity": v,
        "temperature": temperature,
        "salinity": salinity,
        "time": t_val,
        "material": material_map[material],
        "coating": coating_map[coating]
    }
    df = pd.DataFrame([input_dict])
    return df[columns]

# ================= MODEL EXECUTION =================
pred = None

if st.button("Run Simulation"):
    try:
        if run_sim:
            cumulative_bio = 0
            for t in range(1, days_input + 1):
                time_display.write(f"Simulation Day: {t}")

                X = build_input(velocity, t)
                pred = model.predict(X)[0]
                
                drag = max(pred[0], 0)
                daily_bio = np.clip(pred[1], 0, 1)
                cumulative_bio += daily_bio * 0.05
                bio = min(cumulative_bio, 1)
                hydro = max(pred[2], 0)
                durability = max(pred[3], 0)

                st.subheader(f"Performance Results (Day {t})")
                col1, col2 = st.columns(2)
                col1.metric("Drag Reduction", f"{drag:.2f} %", delta=f"{drag-50:.2f}% vs baseline")
                col1.metric("Biofouling Risk", f"{bio:.2f}")
                col2.metric("Hydrophobicity", f"{hydro:.2f} °")
                col2.metric("Durability Index", f"{durability:.2f}")
                
                time_lib.sleep(speed)
        else:
            X = build_input(velocity, days_input)
            pred = model.predict(X)[0]
            drag, bio, hydro, durability = max(pred[0], 0), np.clip(pred[1], 0, 1), max(pred[2], 0), max(pred[3], 0)

            st.subheader("Performance Results")
            st.metric("Drag Reduction", f"{drag:.2f} %")
            st.metric("Biofouling Risk Index", f"{bio:.2f}")
            st.metric("Hydrophobicity", f"{hydro:.2f} °")
            st.metric("Durability Index", f"{durability:.2f}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ================= BIOFOULING OVER TIME =================
st.subheader("Biofouling Risk Over Time")

times = np.arange(1, days_input + 1)
bio_curve = []

for t in times:
    try:
        X_t = build_input(velocity, t)
        pred_t = model.predict(X_t)[0][1]
        bio_curve.append(np.clip(pred_t, 0, 1))
    except:
        bio_curve.append(0)

fig_time, ax_time = plt.subplots()
ax_time.plot(times, bio_curve, color='teal')
ax_time.set_title("Biofouling Risk Projection")
ax_time.set_xlabel("Days")
ax_time.set_ylabel("Risk Index (0–1)")
st.pyplot(fig_time)
        
# ================= MULTISCALE SURFACE GENERATION =================
resolution = 100
x = np.linspace(0, 5, resolution)
y = np.linspace(0, 5, resolution)
Xg, Yg = np.meshgrid(x, y)

hull_base = np.clip(1 - (Yg**2) / (1.5**2), 0, 1)
riblet = riblet_height * np.sin((2 * np.pi / riblet_spacing) * Xg)
nano_freq = 40
lotus = (0.08 * lotus_intensity) * (np.cos(nano_freq * Xg) * np.cos(nano_freq * Yg))
Z = hull_base + riblet + lotus

# ================= 3D PLOT =================
st.subheader("3D Biomimetic Hull Surface (Multiscale)")
fig_3d = go.Figure(data=[go.Surface(x=Xg, y=Yg, z=Z, colorscale='Viridis')])
st.plotly_chart(fig_3d, use_container_width=True)

# ================= STL EXPORT =================
st.subheader("Export STL for SimScale")
if st.button("Generate STL"):
    try:
        _, _, _, file_path = generate_stl(
            riblet_spacing, riblet_height, lotus_intensity,
            resolution=80, mode=mode
        )
        with open(file_path, "rb") as f:
            st.download_button("Download STL", f, "biomimetic_hull.stl")
    except Exception as e:
        st.error(f"STL generation failed: {e}")

# ================= FLOW FIELD ANALYSIS =================
st.subheader("Flow Interaction Field (Denticle-Driven)")
dZdx = np.gradient(Z, axis=1)
dZdy = np.gradient(Z, axis=0)
U = 1 - np.abs(dZdx) * 2
V = -dZdy * 0.5
velocity_field = np.sqrt(U**2 + V**2)

fig_flow, ax_flow = plt.subplots()
cf = ax_flow.contourf(Xg, Yg, velocity_field, levels=30, cmap='RdYlBu_r')
plt.colorbar(cf, ax=ax_flow, label="Velocity Magnitude")
st.pyplot(fig_flow)

# ================= FLOW VECTORS =================
st.subheader("Flow Direction Field")
step = 6
fig_vec, ax_vec = plt.subplots()
ax_vec.quiver(Xg[::step, ::step], Yg[::step, ::step], U[::step, ::step], V[::step, ::step], color='navy')
st.pyplot(fig_vec)

# ================= BIOFOULING ZONES =================
st.subheader("Biofouling Risk Zones")
low_flow = velocity_field < np.percentile(velocity_field, 40)
low_lotus = np.abs(lotus) < np.percentile(np.abs(lotus), 40)
attachment_zone = low_flow & low_lotus

fig_bio, ax_bio = plt.subplots()
ax_bio.contourf(Xg, Yg, attachment_zone, cmap='Reds')
ax_bio.set_title("Predicted High Fouling Attachment Zones")
st.pyplot(fig_bio)

# ================= DRAG CURVE =================
st.subheader("Drag vs Velocity")
vels = np.linspace(0.5, 12, 40)
drag_curve = []
for v in vels:
    try:
        X_v = build_input(v, days_input)
        drag_val = model.predict(X_v)[0][0]
        drag_curve.append(max(drag_val, 0))
    except:
        drag_curve.append(0)
        
fig_drag, ax_drag = plt.subplots()
ax_drag.plot(vels, drag_curve, 'r--')
ax_drag.set_xlabel("Velocity")
ax_drag.set_ylabel("Drag Reduction %")
st.pyplot(fig_drag)

# ================= COMPARISON =================
st.subheader("Surface Comparison")
labels = ["Smooth", "Riblet", "Lotus", "Hybrid"]
# Use current prediction if available, else a baseline
current_drag = max(pred[0], 0) if pred is not None else 75
values = [40, 65, 60, current_drag]

fig_comp, ax_comp = plt.subplots()
ax_comp.bar(labels, values, color=['gray', 'blue', 'green', 'orange'])
ax_comp.set_ylabel("Efficiency Score")
st.pyplot(fig_comp)

# ================= INSIGHT =================
st.subheader("Engineering Insight")
st.info("This system integrates riblet-induced drag reduction and lotus-inspired nano-scale hydrophobicity.")

if st.button("Show Engineering Interpretation"):
    st.write("""
    - **Riblets:** Control boundary layer flow to reduce turbulent drag.
    - **Lotus Effect:** Nano-textures minimize the liquid-solid contact area, preventing bio-adhesion.
    - **Hybrid Advantage:** The combination addresses both fluid dynamics (short term) and fouling (long term).
    """)
