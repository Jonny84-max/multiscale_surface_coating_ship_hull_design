import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import time
from surface_3d_pattern import generate_stl # Ensure this script uses the Triangle Wave logic

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    try:
        data = joblib.load("model.pkl")
        # Captures the model and the specific 25 columns from your Refined Dataset
        return data["model"], data["columns"]
    except:
        st.error("Model Alignment Error: Ensure model.pkl contains 'model' and 'columns' keys.")
        return None, None

model, columns = load_model()

# ================= MAPPINGS =================
material_map = {"GFRP": 0, "CFRP": 1, "Hybrid": 2}
coating_map = {"Epoxy": 0, "Vinyl": 1, "PDMS": 2, "Fluoro": 3, "Sol-gel": 4}

# ================= UI =================
st.set_page_config(page_title="Refined Biomimetic Hull", layout="wide")
st.title("🚢 Hybrid Biomimetic Hull Design System")

st.sidebar.header("Design Inputs")
riblet_height = st.sidebar.slider("Riblet Height (mm)", 0.01, 0.3, 0.15)
riblet_spacing = st.sidebar.slider("Riblet Spacing (mm)", 0.05, 1.0, 0.5)
lotus_intensity = st.sidebar.slider("Lotus Intensity", 0.0, 1.0, 0.5)

velocity = st.sidebar.slider("Velocity", 0.5, 25.0, 12.0)
temperature = st.sidebar.slider("Temperature", 0, 40, 20)
salinity = st.sidebar.slider("Salinity", 10, 40, 35)
days_input = st.sidebar.slider("Time (days)", 1, 1095, 30)

material = st.sidebar.selectbox("Material", list(material_map.keys()))
coating = st.sidebar.selectbox("Coating", list(coating_map.keys()))
mode = st.sidebar.selectbox("Output Mode", ["Visualization STL", "CFD STL"])

run_sim = st.sidebar.checkbox("Run Time Simulation")
speed = st.sidebar.slider("Simulation Speed (sec per step)", 0.1, 1.5, 0.4)

# ================= REFINED FEATURE BUILDER (The Brain) =================
def build_input(v, t_val):
    """
    SECTION CAPTURED: This now generates all 25 features required by the 
    Refined Physics model to prevent 'Feature Mismatch' errors.
    """
    # Base Inputs
    h = riblet_height
    s = riblet_spacing
    li = lotus_intensity
    
    # Physics Engineering (Matching your Refined CSV)
    asp_ratio = h / (s + 1e-6)
    ms_index = li * (1 - (h * 0.1))
    v_r_interact = v * asp_ratio
    f = 1 / (1 + asp_ratio)
    cos_theta = -1 + f * (np.cos(np.radians(110)) + 1)
    eff_ca = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
    slip_len = (1 / (np.sqrt(f) + 1e-6)) * s
    
    input_dict = {
        "riblet_height": h, "riblet_spacing": s, "lotus_intensity": li,
        "velocity": v, "temperature": temperature, "salinity": salinity, "time": t_val,
        "material": material_map[material], "coating": coating_map[coating],
        "aspect_ratio": asp_ratio, "multiscale_index": ms_index, 
        "velocity_riblet_interact": v_r_interact,
        "solid_fraction_f": f, "cos_theta_eff": cos_theta, 
        "effective_contact_angle": eff_ca, "estimated_slip_length": slip_len,
        "is_shs": 1 if eff_ca > 145 else 0,
        "better_than_painting": 1 if asp_ratio > 0.2 else 0,
        "design_category": "Multi-scale" if ms_index > 0.6 else "Shark-like",
        # Placeholders for targets
        "composite_performance": 0.5, "drag_reduction_clipped": 0.0,
        "drag_reduction": 0.0, "biofouling": 0.0, "hydrophobicity": 0.0, "durability": 0.0
    }
    
    df = pd.DataFrame([input_dict])
    return df[columns] # Forces alignment with the model's training columns

# ================= MODEL EXECUTION =================
pred = [0, 0, 0, 0] 

if st.button("Run Simulation"):
    try:
        results_card = st.empty()
        progress_bar = st.progress(0)
        cumulative_bio = 0
        
        loop_range = range(1, days_input + 1) if run_sim else range(days_input, days_input + 1)
        
        for t in loop_range:
            X = build_input(velocity, t)
            raw_pred = model.predict(X)[0]
            
            # Pack results for the comparison chart
            pred = raw_pred 
            
            drag_red = np.clip(raw_pred[0], 0, 95)
            daily_risk = max(0, raw_pred[1]) 
            cumulative_bio += daily_risk * 0.02
            total_bio = min(1.0, cumulative_bio)
            
            wear_factor = max(0.5, 1 - (t / 2000)) 
            hydro = max(0, raw_pred[2] * wear_factor)
            durability = max(0, raw_pred[3])

            with results_card.container():
                st.subheader(f"📊 Hull Performance Analysis: Day {t}")
                if hydro > 145: st.success("✨ Status: Superhydrophobic (SHS)")
                elif hydro > 90: st.info("💧 Status: Hydrophobic")
                else: st.warning("⚠️ Status: Hydrophilic (High Drag)")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Drag Reduc.", f"{drag_red:.1f}%")
                c2.metric("Bio-Accum.", f"{total_bio:.2f}", delta=f"{daily_risk:.3f}")
                c3.metric("Contact Angle", f"{hydro:.1f}°")
                c4.metric("Durability", f"{durability:.0f} pts")

            progress_bar.progress(t / days_input)
            if run_sim: time.sleep(speed)
            
    except Exception as e:
        st.error(f"Prediction Error: {e}")

# ================= MULTISCALE SURFACE GENERATION (The Geometry) =================
# SECTION CAPTURED: Replaced Sine wave with Triangle wave to match Refined Physics
resolution = 100
x = np.linspace(0, 5, resolution)
y = np.linspace(0, 5, resolution)
Xg, Yg = np.meshgrid(x, y)

hull_base = np.clip(1 - (Yg**2) / (1.5**2), 0, 1)
# TRIANGLE WAVE CALCULATION
phase = (2 * np.pi / riblet_spacing) * Xg
riblet = (2 * riblet_height / np.pi) * np.arcsin(np.sin(phase))

nano_freq = 45
nano_amp = 0.07 * lotus_intensity
np.random.seed(1)
lotus = nano_amp * (np.cos(nano_freq * Xg) * np.cos(nano_freq * Yg)) + (0.01 * np.random.randn(*Xg.shape))

Z = hull_base + riblet + lotus

# ================= VISUALS =================
st.subheader("3D Biomimetic Hull Surface (Triangle Profile)")
fig_3d = go.Figure(data=[go.Surface(x=Xg, y=Yg, z=Z, colorscale='Viridis')])
st.plotly_chart(fig_3d, use_container_width=True)

# ... [Flow Field, Biofouling Zones, and Comparison Charts code remain here] ...

# ================= INSIGHT =================
st.subheader("Engineering Insight")
st.info("This system uses refined slip-length and aspect-ratio physics to predict SHS performance.")

if st.button("Show Engineering Interpretation"):
    st.write("""
    - **Triangle Riblets:** Sharper peaks (calculated via Arcsin) provide superior vortex control compared to Sine waves.
    - **Aspect Ratio:** The ratio of height to spacing is the primary driver of the Cassie-Baxter state.
    - **Slip Length:** The model predicts the 'virtual' fluid slide over air pockets trapped in the nano-texture.
    """)
