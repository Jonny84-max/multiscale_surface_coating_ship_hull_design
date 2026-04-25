import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import time as time_lib
from surface_3d_pattern import generate_stl

# ================= CONFIGURATION =================
st.set_page_config(page_title="Hybrid Biomimetic Hull Design", layout="wide")

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    try:
        data = joblib.load("model.pkl")
        if isinstance(data, dict):
            return data["model"], data.get("columns", None)
        return data, None
    except Exception as e:
        st.error(f"Model File Error: {e}")
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
velocity = st.sidebar.slider("Velocity (knots)", 0.5, 25.0, 12.0)
temperature = st.sidebar.slider("Sea Temp (°C)", 0, 40, 20)
salinity = st.sidebar.slider("Salinity (PSU)", 10, 40, 35)
days_input = st.sidebar.slider("Time (days)", 1, 1095, 30)

material = st.sidebar.selectbox("Substrate", list(material_map.keys()))
coating = st.sidebar.selectbox("Coating", list(coating_map.keys()))
mode = st.sidebar.selectbox("STL Output Mode", ["Visualization STL", "CFD STL"])

run_sim = st.sidebar.checkbox("Animate Time Simulation")
speed = st.sidebar.slider("Animation Speed", 0.1, 1.5, 0.4)

# ================= REFINED FEATURE BUILDER =================
def build_input(v, t_val):
    """
    Engineers 25 features to match biomimetic_dataset_physics_refined.csv 
    """
    asp_ratio = riblet_height / (riblet_spacing + 1e-6)
    f = 1 / (1 + asp_ratio) # Solid fraction [cite: 1]
    cos_theta_eff = -1 + f * (np.cos(np.radians(110)) + 1)
    eff_ca = np.degrees(np.arccos(np.clip(cos_theta_eff, -1, 1))) # Effective Contact Angle [cite: 1]
    
    input_dict = {
        "riblet_height": riblet_height, "riblet_spacing": riblet_spacing, 
        "lotus_intensity": lotus_intensity, "velocity": v, 
        "temperature": temperature, "salinity": salinity, "time": t_val,
        "material": material_map[material], "coating": coating_map[coating],
        "aspect_ratio": asp_ratio, 
        "multiscale_index": lotus_intensity * (1 - (riblet_height * 0.1)),
        "velocity_riblet_interact": v * asp_ratio, 
        "solid_fraction_f": f,
        "effective_contact_angle": eff_ca, 
        "cos_theta_eff": cos_theta_eff,
        "estimated_slip_length": (1 / (np.sqrt(f) + 1e-6)) * riblet_spacing,
        "is_shs": 1 if eff_ca > 145 else 0, 
        "better_than_painting": 1 if asp_ratio > 0.2 else 0,
        "design_category": 1 if lotus_intensity > 0.6 else 0
    }
    
    df = pd.DataFrame([input_dict])
    # Match exact column list from trained model [cite: 1]
    if model_columns:
        for col in model_columns:
            if col not in df.columns: df[col] = 0
        return df[model_columns]
    return df

# ================= PERFORMANCE SIMULATION =================
st.title("Hybrid Biomimetic Hull Design System")
results_card = st.empty()
last_pred = [0, 0, 0, 0]

if st.button("🚀 Run Simulation Analysis", use_container_width=True):
    try:
        progress_bar = st.progress(0)
        cumulative_bio = 0
        t_range = range(1, days_input + 1) if run_sim else [days_input]
        
        for t in t_range:
            X = build_input(velocity, t)
            raw_pred = model.predict(X)[0]
            last_pred = raw_pred
            
            # Physics scaling for visual display
            drag_red = np.clip(raw_pred[0], 0, 95)
            cumulative_bio = min(1.0, (t * 0.001) + (raw_pred[1] * 0.01))
            hydro = raw_pred[2]
            durability = max(0, raw_pred[3])

            with results_card.container():
                st.subheader(f"📊 Performance Analysis: Day {t}")
                if hydro > 150: st.success("✨ Superhydrophobic (SHS) Active")
                elif hydro > 90: st.info("💧 Hydrophobic State")
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Drag Reduc.", f"{drag_red:.1f}%")
                c2.metric("Bio-Accumulation", f"{cumulative_bio:.3f}")
                c3.metric("Contact Angle", f"{hydro:.1f}°")
                c4.metric("Durability", f"{durability:.0f} pts")
            
            progress_bar.progress(t / days_input)
            if run_sim: time_lib.sleep(speed)
    except Exception as e:
        st.error(f"Simulation Error: {e}")

# ================= 3D SURFACE & ANALYSIS =================
res = 100
x = np.linspace(0, 5, res)
y = np.linspace(0, 5, res)
Xg, Yg = np.meshgrid(x, y)

hull_base = np.clip(1 - (Yg**2) / (1.5**2), 0, 1)
riblet = riblet_height * np.sin((2 * np.pi / riblet_spacing) * Xg)
nano = (0.08 * lotus_intensity) * (np.cos(40 * Xg) * np.cos(40 * Yg))
Z = hull_base + riblet + nano

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("3D Surface Morphology")
    fig_3d = go.Figure(data=[go.Surface(z=Z, colorscale='Viridis')])
    st.plotly_chart(fig_3d, use_container_width=True)

with col_b:
    st.subheader("Flow Interaction Field")
    dZdx = np.gradient(Z, axis=1)
    velocity_field = 1 - np.abs(dZdx) * 2
    fig_flow, ax_flow = plt.subplots()
    cf = ax_flow.contourf(Xg, Yg, velocity_field, levels=30, cmap='RdYlBu_r')
    plt.colorbar(cf, label="Velocity Magnitude")
    st.pyplot(fig_flow)

# ================= DETAILED VISUALS =================
col_c, col_d = st.columns(2)

with col_c:
    st.subheader("Biofouling Risk Zones")
    attachment_zone = (velocity_field < 0.4) & (np.abs(nano) < 0.02)
    fig_bio, ax_bio = plt.subplots()
    ax_bio.contourf(Xg, Yg, attachment_zone, cmap='Reds')
    st.pyplot(fig_bio)

with col_d:
    st.subheader("Drag vs Velocity Curve")
    vels = np.linspace(0.5, 25, 20)
    drag_vals = [model.predict(build_input(v, days_input))[0][0] for v in vels]
    fig_drag, ax_drag = plt.subplots()
    ax_drag.plot(vels, drag_vals, 'r-o')
    ax_drag.set_ylabel("Drag Reduction %")
    st.pyplot(fig_drag)

# ================= STL EXPORT =================
st.divider()
st.subheader("Export CAD Data")
if st.button("Generate STL for SimScale", use_container_width=True):
    _, _, _, path = generate_stl(riblet_spacing, riblet_height, lotus_intensity, mode=mode)
    with open(path, "rb") as f:
        st.download_button("📥 Download STL File", f, "hull_design.stl")

# ================= COMPARISON =================
st.subheader("Surface Efficiency Comparison")
labels = ["Smooth", "Riblet", "Lotus", "Hybrid (Current)"]
comp_values = [40, 65, 60, max(last_pred[0], 5)] # Uses current prediction
fig_comp, ax_comp = plt.subplots(figsize=(8, 3))
ax_comp.bar(labels, comp_values, color=['#bdc3c7', '#3498db', '#2ecc71', '#e67e22'])
st.pyplot(fig_comp)
