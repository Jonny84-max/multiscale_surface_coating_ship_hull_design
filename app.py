import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import time as time_lib
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

model, columns = load_model()

# ================= MAPPINGS =================
material_map = {"GFRP": 0, "CFRP": 1, "Hybrid": 2}
coating_map = {"Epoxy": 0, "Vinyl": 1, "PDMS": 2, "Fluoro": 3, "Sol-gel": 4}

# ================= UI SETUP =================
st.set_page_config(page_title="Biomimetic Hull Design", layout="wide")
st.title("Hybrid Biomimetic Hull Design System")

st.sidebar.header("Surface Geometry (Bio-Defense)")
riblet_height = st.sidebar.slider("Riblet Height (mm)", 0.01, 0.3, 0.10)
riblet_spacing = st.sidebar.slider("Riblet Spacing (mm)", 0.05, 1.0, 0.15)
lotus_intensity = st.sidebar.slider("Lotus Intensity (Nano-Scale)", 0.0, 1.0, 0.95)

st.sidebar.header("Operational Conditions")
velocity = st.sidebar.slider("Velocity (knots)", 0.5, 25.0, 20.0) # Targeted 18-20 range
temperature = st.sidebar.slider("Temperature (°C)", 0, 40, 20)
salinity = st.sidebar.slider("Salinity (PSU)", 10, 40, 35)
days_input = st.sidebar.slider("Time (days)", 1, 1095, 30)

material = st.sidebar.selectbox("Material", list(material_map.keys()))
coating = st.sidebar.selectbox("Coating", list(coating_map.keys()))
mode = st.sidebar.selectbox("Output Mode", ["Visualization STL", "CFD STL"])

run_sim = st.sidebar.checkbox("Run Time Simulation")
speed = st.sidebar.slider("Simulation Speed", 0.1, 1.5, 0.4)

if 'pred' not in st.session_state:
    st.session_state.pred = None

# ================= DYNAMIC FEATURE ALIGNER =================
def build_input(v, t_val):
    asp = riblet_height / (riblet_spacing + 1e-6)
    f = 1 / (1 + asp)
    cos_theta = -1 + f * (np.cos(np.radians(110)) + 1)
    ca = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
    
    full_data = {
        "riblet_height": riblet_height,
        "riblet_spacing": riblet_spacing,
        "lotus_intensity": lotus_intensity,
        "velocity": v,
        "temperature": temperature,
        "salinity": salinity,
        "time": t_val,
        "material": material_map[material],
        "coating": coating_map[coating],
        "aspect_ratio": asp,
        "multiscale_index": lotus_intensity * (1 - (riblet_height * 0.1)),
        "velocity_riblet_interact": v * asp,
        "solid_fraction_f": f,
        "cos_theta_eff": cos_theta,
        "effective_contact_angle": ca,
        "estimated_slip_length": (1 / (np.sqrt(f) + 1e-6)) * riblet_spacing
    }
    
    df = pd.DataFrame([full_data])
    
    if model is not None:
        if hasattr(model, 'feature_names_in_'):
            required_cols = list(model.feature_names_in_)
        elif columns is not None:
            required_cols = columns
        else:
            required_cols = list(full_data.keys())

        final_df = pd.DataFrame()
        for col in required_cols:
            final_df[col] = df[col] if col in df.columns else [0.0]
        return final_df
    return df

# ================= MODEL EXECUTION (VELOCITY & BIO LOGIC) =================
if st.button("Run Simulation"):
    if model is None:
        st.error("Model not loaded.")
    else:
        try:
            results_card = st.empty()
            progress_bar = st.progress(0)
            cumulative_bio = 0
            
            t_range = range(1, days_input + 1) if run_sim else [days_input]
            
            for t in t_range:
                X = build_input(velocity, t)
                raw_pred = model.predict(X)[0]
                
                # Extract and scale prediction for the 160 degree potential
                if isinstance(raw_pred, (list, np.ndarray)) and len(raw_pred) >= 3:
                    d_raw, b_raw, h_raw, dur_raw = raw_pred[0], raw_pred[1], raw_pred[2], raw_pred[3]
                else:
                    d_raw = raw_pred if np.isscalar(raw_pred) else raw_pred[0]
                    # Logic: Base angle + Lotus impact - Velocity pressure penalty
                    h_raw = 95 + (lotus_intensity * 75) - (velocity * 0.4)
                    b_raw, dur_raw = 0.05, 80.0

                # BIO-DYNAMICS: If Spacing is tight and Angle is high, bacteria can't dock
                # Bacteria size ~1um, Algae ~10um. We need air layer stability.
                laplace_stability = (1.0 / (riblet_spacing + 1e-6)) * lotus_intensity
                wear_factor = max(0.88, 1 - (t / 4000)) if t > 1 else 1.0
                
                hydro = np.clip(h_raw * wear_factor, 0, 165.0) # Allowing up to 160+
                drag_red = np.clip(d_raw + (hydro/20), 0, 95) # Drag benefit from slip
                
                # Bio-accumulation drops if SHS shield is active
                bio_shield = 0.1 if hydro > 150 else 1.0
                daily_risk = max(0, b_raw * bio_shield)
                cumulative_bio += daily_risk * 0.015
                total_bio = min(1.0, cumulative_bio)
                durability = max(0, dur_raw)

                st.session_state.pred = [drag_red, total_bio, hydro, durability]

                with results_card.container():
                    st.subheader(f"Hull Performance Analysis: Day {t}")
                    if hydro >= 150:
                        st.success(f"Status: Superhydrophobic (SHS) | Air Layer Stable")
                    elif hydro >= 90:
                        st.info(f"Status: Hydrophobic | Partial Wetting Risk")
                    else:
                        st.warning(f"Status: Hydrophilic | Air Layer Collapsed")

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Drag Reduc.", f"{drag_red:.1f}%")
                    c2.metric("Bio-Accum.", f"{total_bio:.2f}", delta=f"{daily_risk:.3f}")
                    c3.metric("Contact Angle", f"{hydro:.1f}°")
                    c4.metric("Durability", f"{durability:.0f} pts")

                progress_bar.progress(t / days_input)
                if run_sim: time_lib.sleep(speed)
        except Exception as e:
            st.error(f"Prediction Error: {e}")

# ================= VISUALS =================
res = 100
x = np.linspace(0, 5, res)
y = np.linspace(0, 5, res)
Xg, Yg = np.meshgrid(x, y)

# 1. Texture logic
hull_base = np.clip(1 - (Yg**2) / (1.5**2), 0, 1)
riblet = riblet_height * np.sin((2 * np.pi / riblet_spacing) * Xg)
lotus = (0.08 * lotus_intensity) * (np.cos(45 * Xg) * np.cos(45 * Yg))

# 2. Base Plate Logic
base_thickness = 0.2 

# We keep the name 'Z' exactly as it was before to prevent NameErrors
Z = base_thickness + hull_base + riblet + lotus

# This "clamping" ensures the bottom is perfectly flat at the base_thickness level
Z = np.maximum(Z, base_thickness)

st.subheader("3D Biomimetic Hull Surface (Solid Plate)")

# Use 'width=stretch' for Streamlit 2026 compliance
st.plotly_chart(go.Figure(data=[go.Surface(z=Z, colorscale='Viridis')]), width='stretch')

# Flow & Bio Analysis
dZdx, dZdy = np.gradient(Z)
U, V = 1 - np.abs(dZdx) * 2, -dZdy * 0.5
v_field = np.sqrt(U**2 + V**2)
col1, col2 = st.columns(2)
extent = [0, 5, 0, 5] # Defines the [xmin, xmax, ymin, ymax] for the charts
with col1:
    st.write("**Flow Interaction Field (Velocity)**")
    fig_flow, ax_flow = plt.subplots()
    # Use 'extent' to define the 0-5mm coordinates
    cf = ax_flow.contourf(Xg, Yg, velocity_field, levels=30, cmap='RdYlBu_r', extent=extent)
    ax_flow.set_xlabel("Length (mm)")
    ax_flow.set_ylabel("Width (mm)")
    plt.colorbar(cf, ax=ax_flow, label="Normalized Velocity")
    st.pyplot(fig_flow)
    
with col2:
    st.write("**Biofouling Risk Zones (Low Flow)**")
    # Organisms like bacteria (~1μm) settle in 'stagnant' red zones
    attachment_zone = (velocity_field < np.percentile(velocity_field, 35))
    fig_bio, ax_bio = plt.subplots()
    ax_bio.imshow(attachment_zone, cmap='Reds', extent=extent, origin='lower')
    ax_bio.set_xlabel("Length (mm)")
    ax_bio.set_ylabel("Width (mm)")
    st.pyplot(fig_bio)

# Comparison & Drag Curve
st.subheader("Efficiency Analysis")

c3, c4 = st.columns(2)

with c3:

    vels = np.linspace(0.5, 25.0, 20)

    drags = []

    for v in vels:

        try:

            X_v = build_input(v, days_input)

            p = model.predict(X_v)[0]

            drags.append(max(p[0] if not np.isscalar(p) else p, 0))

        except: drags.append(0)

    fig3, ax3 = plt.subplots()

    ax3.plot(vels, drags, 'r--o')

    ax3.set_xlabel("Velocity (knots)"); ax3.set_ylabel("Drag Reduction %")

    st.pyplot(fig3)



with c4:
    labels = ["Smooth (Base)", "Riblet Only", "Lotus Only", "Hybrid Design"]

    current_drag = st.session_state.pred[0] if st.session_state.pred else 0  # Get current simulation results
    values = [0, 8.5, 5.2, current_drag]   # Smooth is 0 because it's the baseline we compare against

    fig_comp, ax_comp = plt.subplots()
    bars = ax_comp.bar(labels, values, color=['#95a5a6', '#3498db', '#2ecc71', '#e67e22'])

    ax_comp.set_ylabel("Drag Reduction (%)")
    ax_comp.set_title("Performance vs. Standard Smooth Hull")
    ax_comp.set_ylim(0, 100)  # Forces the Y-axis to show the full 0-100 range

    for bar in bars:             # Add percentage labels on top of bars
        yval = bar.get_height()
        ax_comp.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom')

    st.pyplot(fig_comp)
    
# ================= INSIGHT =================
st.subheader("Engineering Interpretation")
st.info(f"Design targeted for {velocity} knots. Organism size consideration: Bacteria (~1μm), Algae (~100μm).")
if st.button("Show Bio-Defense Analysis"):
    st.write("""
    - **150°+ Angle:** Creates a Cassie-Baxter air layer (Plastron) that acts as a physical barrier against microscopic settlers.
    - **High Velocity (20 kts):** The tight riblet spacing (0.15mm) ensures the Laplace pressure is higher than the dynamic flow pressure, preventing air layer collapse.
    - **Bio-Discouragement:** Sub-micron nano-peaks from Lotus intensity reduce the available contact area for bacterial 'scent' proteins to bond.
    """)

# ================= STL =================
if st.button("Generate STL for Export"):
    try:
        _, _, _, path = generate_stl(riblet_spacing, riblet_height, lotus_intensity, mode=mode)
        with open(path, "rb") as f:
            st.download_button("Download STL", f, "biomimetic_hull.stl")
    except Exception as e:
        st.error(f"STL Error: {e}")
