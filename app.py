import streamlit as st
import numpy as np
import joblib
import pandas as pd
import plotly.graph_objects as go
import time
from surface_3d_pattern import generate_stl 

# ================= CONFIGURATION =================
st.set_page_config(page_title="Biomimetic Hull Design", layout="wide")

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    try:
        # Assumes model.pkl is a trained model or a dict containing metadata
        data = joblib.load("model.pkl")
        if isinstance(data, dict):
            return data["model"], data.get("columns", None)
        return data, None
    except Exception as e:
        st.error(f"Missing 'model.pkl'. Please ensure the model file is uploaded. Error: {e}")
        return None, None

model, model_columns = load_model()

# ================= MAPPINGS =================
material_map = {"GFRP": 0, "CFRP": 1, "Hybrid": 2}
coating_map = {"Epoxy": 0, "Vinyl": 1, "PDMS": 2, "Fluoro": 3, "Sol-gel": 4}

# ================= UI SIDEBAR =================
st.sidebar.header("Design Parameters")
riblet_height = st.sidebar.slider("Riblet Height (mm)", 0.01, 0.3, 0.15)
riblet_spacing = st.sidebar.slider("Riblet Spacing (mm)", 0.05, 1.0, 0.5)
lotus_intensity = st.sidebar.slider("Lotus Intensity", 0.0, 1.0, 0.5)

st.sidebar.header("Environment & Material")
velocity = st.sidebar.slider("Velocity (knots)", 0.5, 25.0, 12.0)
temperature = st.sidebar.slider("Sea Temp (°C)", 0, 40, 20)
salinity = st.sidebar.slider("Salinity (PSU)", 10, 40, 35)
days_input = st.sidebar.slider("Simulation Time (days)", 1, 1095, 30)

material = st.sidebar.selectbox("Substrate Material", list(material_map.keys()))
coating = st.sidebar.selectbox("Surface Coating", list(coating_map.keys()))
run_sim = st.sidebar.checkbox("Animate Time Simulation")
speed = st.sidebar.slider("Animation Speed", 0.1, 1.5, 0.4)

# ================= FEATURE BUILDER =================
def build_input(v, t_val):
    """
    Engineers 25 features to match the refined physics dataset.
    """
    asp_ratio = riblet_height / (riblet_spacing + 1e-6)
    f = 1 / (1 + asp_ratio) # Solid fraction calculation
    # Effective contact angle calculation for hydrophobic surfaces
    cos_theta_eff = -1 + f * (np.cos(np.radians(110)) + 1)
    eff_ca = np.degrees(np.arccos(np.clip(cos_theta_eff, -1, 1)))
    
    input_dict = {
        "riblet_height": riblet_height, 
        "riblet_spacing": riblet_spacing, 
        "lotus_intensity": lotus_intensity,
        "velocity": v, 
        "temperature": temperature, 
        "salinity": salinity, 
        "time": t_val,
        "material": material_map[material], 
        "coating": coating_map[coating],
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
    # Ensure all 25 features are present and in correct order
    if model_columns:
        for col in model_columns:
            if col not in df.columns: df[col] = 0
        return df[model_columns]
    return df

# ================= MAIN INTERFACE =================
st.title("Hybrid Biomimetic Hull Design System")

col_viz, col_metrics = st.columns([2, 1])

with col_metrics:
    st.subheader("Performance Metrics")
    results_card = st.empty()
    
    if st.button("Run Prediction Analysis", use_container_width=True):
        if model is None:
            st.error("Model not loaded.")
        else:
            try:
                progress_bar = st.progress(0)
                t_range = range(1, days_input + 1) if run_sim else [days_input]
                
                for t in t_range:
                    X = build_input(velocity, t)
                    raw_pred = model.predict(X)[0]
                    
                    with results_card.container():
                        st.metric("Drag Reduction", f"{np.clip(raw_pred[0], 0, 95):.1f}%")
                        st.metric("Bio-Accumulation Index", f"{min(1.0, t*0.001 + raw_pred[1]*0.01):.3f}")
                        st.metric("Effective Contact Angle", f"{raw_pred[2]:.1f}°")
                        st.metric("Durability Score", f"{raw_pred[3]:.0f} pts")
                    
                    progress_bar.progress(t / days_input)
                    if run_sim: time.sleep(speed)
            except Exception as e:
                st.error(f"Prediction Error: {e}")

with col_viz:
    st.subheader("3D Surface Morphology")
    res = 80
    x_v = np.linspace(0, 5, res)
    y_v = np.linspace(0, 5, res)
    Xv, Yv = np.meshgrid(x_v, y_v)
    
    # Matching the physics model geometry logic
    z_base = np.clip(1 - (Yv**2) / (1.5**2), 0, 1)
    z_riblet = (2 * riblet_height / np.pi) * np.arcsin(np.sin((2 * np.pi / riblet_spacing) * Xv))
    z_nano = (0.05 * lotus_intensity) * (np.cos(40 * Xv) * np.cos(40 * Yv))
    Z_viz = z_base + z_riblet + z_nano

    fig = go.Figure(data=[go.Surface(z=Z_viz, colorscale='Blues', showscale=False)])
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), height=500)
    st.plotly_chart(fig, use_container_width=True)

    if st.button("Generate CAD (STL) File", use_container_width=True):
        with st.spinner("Building high-resolution mesh..."):
            _, _, _, path = generate_stl(riblet_spacing, riblet_height, lotus_intensity)
            with open(path, "rb") as f:
                st.download_button("Download STL for SimScale", f, "hull_design.stl")
