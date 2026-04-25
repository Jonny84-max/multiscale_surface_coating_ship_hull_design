import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import time as time_lib
from surface_3d_pattern import generate_stl 

# ================= CONFIGURATION =================
st.set_page_config(page_title="Biomimetic Design System", layout="wide")

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    try:
        # UPDATED: Using your specific model filename
        data = joblib.load("shs_predictive_model.pkl")
        
        # Checking if it's a dictionary (containing model + columns) or a direct model
        if isinstance(data, dict) and "model" in data:
            return data["model"], data.get("columns", None)
        return data, None
    except Exception as e:
        st.error(f"⚠️ Model Load Error: {e}. Please ensure 'shs_predictive_model.pkl' is in the app folder.")
        return None, None

model, model_columns = load_model()

# ================= MAPPINGS =================
material_map = {"GFRP": 0, "CFRP": 1, "Hybrid": 2}
coating_map = {"Epoxy": 0, "Vinyl": 1, "PDMS": 2, "Fluoro": 3, "Sol-gel": 4}

# ================= UI SIDEBAR =================
st.sidebar.header("Design Inputs")
riblet_height = st.sidebar.slider("Riblet Height (mm)", 0.01, 0.3, 0.15)
riblet_spacing = st.sidebar.slider("Riblet Spacing (mm)", 0.05, 1.0, 0.5)
lotus_intensity = st.sidebar.slider("Lotus Intensity", 0.0, 1.0, 0.5)

st.sidebar.header("Simulation Settings")
velocity = st.sidebar.slider("Velocity", 0.5, 25.0, 12.0)
temperature = st.sidebar.slider("Temperature", 0, 40, 20)
salinity = st.sidebar.slider("Salinity", 10, 40, 35)
days_input = st.sidebar.slider("Time (days)", 1, 1095, 30)

material = st.sidebar.selectbox("Material", list(material_map.keys()))
coating = st.sidebar.selectbox("Coating", list(coating_map.keys()))
mode = st.sidebar.selectbox("STL Mode", ["Visualization STL", "CFD STL"])
run_sim = st.sidebar.checkbox("Run Time Simulation")
speed = st.sidebar.slider("Sim Speed", 0.1, 1.5, 0.4)

# ================= REFINED FEATURE BUILDER =================
def build_input(v, t_val):
    asp = riblet_height / (riblet_spacing + 1e-6)
    f = 1 / (1 + asp)
    ca = np.degrees(np.arccos(np.clip(-1 + f * (np.cos(np.radians(110)) + 1), -1, 1)))
    
    input_dict = {
        "riblet_height": riblet_height, "riblet_spacing": riblet_spacing, "lotus_intensity": lotus_intensity,
        "velocity": v, "temperature": temperature, "salinity": salinity, "time": t_val,
        "material": material_map[material], "coating": coating_map[coating],
        "aspect_ratio": asp, "multiscale_index": lotus_intensity * (1 - (riblet_height * 0.1)),
        "velocity_riblet_interact": v * asp, "solid_fraction_f": f,
        "effective_contact_angle": ca, "estimated_slip_length": (1 / (np.sqrt(f) + 1e-6)) * riblet_spacing,
        "is_shs": 1 if ca > 145 else 0, "better_than_painting": 1 if asp > 0.2 else 0,
        "design_category": 1 if lotus_intensity > 0.6 else 0
    }
    df = pd.DataFrame([input_dict])
    if model_columns:
        for col in model_columns:
            if col not in df.columns: df[col] = 0
        return df[model_columns]
    return df

# ================= PERFORMANCE EXECUTION =================
st.title("🚢 Hybrid Biomimetic Hull Design System")
results_card = st.empty()
last_pred = [0, 0, 0, 0]

if st.button("Run Prediction"):
    if model is None:
        st.error("Cannot run simulation: Model not loaded. Check the sidebar error for details.")
    else:
        try:
            progress_bar = st.progress(0)
            t_range = range(1, days_input + 1) if run_sim else [days_input]
            for t in t_range:
                X = build_input(velocity, t)
                raw_pred = model.predict(X)[0]
                last_pred = raw_pred
                with results_card.container():
                    st.subheader(f"📊 Performance: Day {t}")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Drag Red.", f"{np.clip(raw_pred[0], 0, 95):.1f}%")
                    c2.metric("Bio-Accum.", f"{min(1.0, t*0.001 + raw_pred[1]*0.01):.3f}")
                    c3.metric("Contact Angle", f"{raw_pred[2]:.1f}°")
                    c4.metric("Durability", f"{max(0, raw_pred[3]):.0f} pts")
                progress_bar.progress(t / days_input)
                if run_sim: time_lib.sleep(speed)
        except Exception as e:
            st.error(f"Prediction Error: {e}")

# ================= COMPLETE VISUAL SUITE =================
res = 100
x = np.linspace(0, 5, res); y = np.linspace(0, 5, res)
Xg, Yg = np.meshgrid(x, y)
Z = np.clip(1 - (Yg**2) / (1.5**2), 0, 1) + \
    (2 * riblet_height / np.pi) * np.arcsin(np.sin((2 * np.pi / riblet_spacing) * Xg)) + \
    (0.06 * lotus_intensity) * (np.cos(45 * Xg) * np.cos(45 * Yg))

# 1. 3D View
st.subheader("3D Surface Morphology")
st.plotly_chart(go.Figure(data=[go.Surface(z=Z, colorscale='Viridis')]), use_container_width=True)

# 2. Flow Fields
st.subheader("Flow Interaction & Quiver")
dZdx = np.gradient(Z, axis=1)
v_field = 1 - np.abs(dZdx) * 2
col1, col2 = st.columns(2)
with col1:
    fig_f, ax_f = plt.subplots(); ax_f.contourf(Xg, Yg, v_field, cmap='RdYlBu_r'); st.pyplot(fig_f)
with col2:
    step = 6
    fig_q, ax_q = plt.subplots(); ax_q.quiver(Xg[::step, ::step], Yg[::step, ::step], v_field[::step, ::step], -np.gradient(Z, axis=0)[::step, ::step], color='navy')
    st.pyplot(fig_q)

# 3. Biofouling & Drag
col3, col4 = st.columns(2)
with col3:
    st.subheader("Biofouling Risk Zones")
    fig_b, ax_b = plt.subplots(); ax_b.contourf(Xg, Yg, (v_field < 0.4), cmap='Reds'); st.pyplot(fig_b)
with col4:
    st.subheader("Drag vs Velocity")
    if model is not None:
        v_range = np.linspace(0.5, 25, 20)
        d_curve = [model.predict(build_input(v, days_input))[0][0] for v in v_range]
        fig_d, ax_d = plt.subplots(); ax_d.plot(v_range, d_curve, 'r-'); st.pyplot(fig_d)
    else:
        st.warning("Load model to see drag curve.")

# 4. STL & Comparison
st.divider()
if st.button("Generate STL"):
    _, _, _, path = generate_stl(riblet_spacing, riblet_height, lotus_intensity, mode=mode)
    with open(path, "rb") as f:
        st.download_button("Download STL", f, "hull.stl")

st.subheader("Efficiency Comparison")
labels = ["Smooth", "Riblet", "Lotus", "Hybrid"]
vals = [40, 65, 60, max(last_pred[0], 0)]
fig_c, ax_c = plt.subplots(); ax_c.bar(labels, vals, color=['gray', 'blue', 'green', 'orange']); st.pyplot(fig_c)
