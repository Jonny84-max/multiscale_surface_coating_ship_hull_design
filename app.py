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
    try:
        data = joblib.load("shs_predictive_model.pkl")
        if isinstance(data, dict) and "model" in data:
            return data["model"], data.get("columns", None)
        return data, None
    except Exception as e:
        st.error(f"Model Load Error: {e}")
        return None, None

model, model_columns = load_model()

# ================= FEATURE ALIGNER =================
def build_input(v, t_val):
    asp = riblet_height / (riblet_spacing + 1e-6)
    f = 1 / (1 + asp)
    cos_theta_eff = -1 + f * (np.cos(np.radians(110)) + 1)
    ca = np.degrees(np.arccos(np.clip(cos_theta_eff, -1, 1)))
    
    input_dict = {
        "riblet_height": riblet_height, "riblet_spacing": riblet_spacing, 
        "lotus_intensity": lotus_intensity, "velocity": v, 
        "temperature": temperature, "salinity": salinity, "time": t_val,
        "material": material_map.get(material, 0), "coating": coating_map.get(coating, 0),
        "aspect_ratio": asp, "multiscale_index": lotus_intensity * (1 - (riblet_height * 0.1)),
        "velocity_riblet_interact": v * asp, "solid_fraction_f": f,
        "cos_theta_eff": cos_theta_eff, "effective_contact_angle": ca, 
        "estimated_slip_length": (1 / (np.sqrt(f) + 1e-6)) * riblet_spacing,
        "is_shs": 1 if ca > 145 else 0, "better_than_painting": 1 if asp > 0.2 else 0,
        "design_category": 1 if lotus_intensity > 0.6 else 0
    }
    
    df = pd.DataFrame([input_dict])
    if model is not None:
        expected = list(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else list(df.columns)
        for col in expected:
            if col not in df.columns: df[col] = 0
        return df[expected]
    return df

# ================= UI SETUP =================
st.set_page_config(page_title="Biomimetic Design System", layout="wide")
st.sidebar.header("Design Inputs")
riblet_height = st.sidebar.slider("Riblet Height (mm)", 0.01, 0.3, 0.15)
riblet_spacing = st.sidebar.slider("Riblet Spacing (mm)", 0.05, 1.0, 0.5)
lotus_intensity = st.sidebar.slider("Lotus Intensity", 0.0, 1.0, 0.5)

velocity = st.sidebar.slider("Velocity", 0.5, 25.0, 12.0)
temperature = st.sidebar.slider("Temperature", 0, 40, 20)
salinity = st.sidebar.slider("Salinity", 10, 40, 35)
days_input = st.sidebar.slider("Time (days)", 1, 1095, 30)

material_map = {"GFRP": 0, "CFRP": 1, "Hybrid": 2}
coating_map = {"Epoxy": 0, "Vinyl": 1, "PDMS": 2, "Fluoro": 3, "Sol-gel": 4}
material = st.sidebar.selectbox("Material", list(material_map.keys()))
coating = st.sidebar.selectbox("Coating", list(coating_map.keys()))
run_sim = st.sidebar.checkbox("Run Time Simulation")

# ================= MAIN APP =================
st.title("🚢 Hybrid Biomimetic Hull Design System")
results_card = st.empty()
main_metric = 0.0 # To store for the bar chart later

if st.button("Run Prediction"):
    if model is not None:
        try:
            t_range = range(1, days_input + 1) if run_sim else [days_input]
            for t in t_range:
                X = build_input(velocity, t)
                prediction = model.predict(X)[0]
                
                # Check if prediction is a single number or a list/array
                if np.isscalar(prediction) or prediction.ndim == 0:
                    main_metric = float(prediction)
                    display_metrics = {"Drag Reduction": f"{main_metric:.2f}%"}
                else:
                    main_metric = float(prediction[0])
                    display_metrics = {
                        "Drag Reduction": f"{main_metric:.2f}%",
                        "Bio-Accum": f"{prediction[1]:.3f}",
                        "Contact Angle": f"{prediction[2]:.1f}°"
                    }

                with results_card.container():
                    st.subheader(f"📊 Performance: Day {t}")
                    cols = st.columns(len(display_metrics))
                    for i, (label, val) in enumerate(display_metrics.items()):
                        cols[i].metric(label, val)
                if run_sim: time.sleep(0.1)
        except Exception as e:
            st.error(f"Prediction logic error: {e}")

# ================= VISUALS =================
res = 100
x = np.linspace(0, 5, res); y = np.linspace(0, 5, res)
Xg, Yg = np.meshgrid(x, y)
Z = np.clip(1 - (Yg**2) / (1.5**2), 0, 1) + \
    (2 * riblet_height / np.pi) * np.arcsin(np.sin((2 * np.pi / riblet_spacing) * Xg)) + \
    (0.06 * lotus_intensity) * (np.cos(45 * Xg) * np.cos(45 * Yg))

st.subheader("3D Surface Morphology")
st.plotly_chart(go.Figure(data=[go.Surface(z=Z, colorscale='Viridis')]), use_container_width=True)

st.subheader("Efficiency Comparison")
# Comparison using the actual value from the model
labels = ["Smooth", "Riblet", "Lotus", "Hybrid"]
vals = [0.0, 8.5, 5.2, main_metric] # Standard baselines vs your AI result
fig_c, ax_c = plt.subplots(figsize=(8, 4))
ax_c.bar(labels, vals, color=['gray', 'blue', 'green', 'orange'])
ax_c.set_ylabel("Drag Reduction %")
st.pyplot(fig_c)

if st.button("Generate STL"):
    _, _, _, path = generate_stl(riblet_spacing, riblet_height, lotus_intensity)
    with open(path, "rb") as f:
        st.download_button("Download STL", f, "hull_design.stl")
