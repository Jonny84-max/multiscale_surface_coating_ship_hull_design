import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import time
# Ensure your local file is named surface_3d_pattern.py
from surface_3d_pattern import generate_stl 

# ================= LOAD MODEL (ROBUST) =================
@st.cache_resource
def load_model():
    try:
        data = joblib.load("model.pkl")
        if isinstance(data, dict) and "model" in data:
            return data["model"], data.get("columns", None)
        else:
            # If the pkl is just the model object itself
            return data, None
    except Exception as e:
        st.error(f"Model Load Error: {e}")
        return None, None

model, model_columns = load_model()

# ================= MAPPINGS =================
material_map = {"GFRP": 0, "CFRP": 1, "Hybrid": 2}
coating_map = {"Epoxy": 0, "Vinyl": 1, "PDMS": 2, "Fluoro": 3, "Sol-gel": 4}

# ================= UI =================
st.set_page_config(page_title="Biomimetic Design System", layout="wide")
st.title("🚢 Hybrid Biomimetic Hull Design System")

st.sidebar.header("Design Inputs")
riblet_height = st.sidebar.slider("Riblet Height (mm)", 0.01, 0.3, 0.15)
riblet_spacing = st.sidebar.slider("Riblet Spacing (mm)", 0.05, 1.0, 0.5)
lotus_intensity = st.sidebar.slider("Lotus Intensity", 0.0, 1.0, 0.5)

st.sidebar.markdown("---")
velocity = st.sidebar.slider("Velocity", 0.5, 25.0, 12.0)
temperature = st.sidebar.slider("Temperature", 0, 40, 20)
salinity = st.sidebar.slider("Salinity", 10, 40, 35)
days_input = st.sidebar.slider("Time (days)", 1, 1095, 30)

material = st.sidebar.selectbox("Material", list(material_map.keys()))
coating = st.sidebar.selectbox("Coating", list(coating_map.keys()))
mode = st.sidebar.selectbox("Output Mode", ["Visualization STL", "CFD STL"])

run_sim = st.sidebar.checkbox("Run Time Simulation")
speed = st.sidebar.slider("Simulation Speed", 0.1, 1.5, 0.4)

# ================= REFINED FEATURE BUILDER =================
def build_input(v, t_val):
    # Physics Engineering matching the 'Refined' CSV logic
    asp_ratio = riblet_height / (riblet_spacing + 1e-6)
    ms_index = lotus_intensity * (1 - (riblet_height * 0.1))
    v_r_interact = v * asp_ratio
    f = 1 / (1 + asp_ratio)
    cos_theta = -1 + f * (np.cos(np.radians(110)) + 1)
    eff_ca = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
    slip_len = (1 / (np.sqrt(f) + 1e-6)) * riblet_spacing
    
    input_dict = {
        "riblet_height": riblet_height, "riblet_spacing": riblet_spacing, "lotus_intensity": lotus_intensity,
        "velocity": v, "temperature": temperature, "salinity": salinity, "time": t_val,
        "material": material_map[material], "coating": coating_map[coating],
        "aspect_ratio": asp_ratio, "multiscale_index": ms_index, 
        "velocity_riblet_interact": v_r_interact, "solid_fraction_f": f, 
        "cos_theta_eff": cos_theta, "effective_contact_angle": eff_ca, "estimated_slip_length": slip_len,
        "is_shs": 1 if eff_ca > 145 else 0, "better_than_painting": 1 if asp_ratio > 0.2 else 0,
        "design_category": 1 if ms_index > 0.6 else 0, # Categorical as Int
        "composite_performance": 0, "drag_reduction_clipped": 0, # Placeholders
        "drag_reduction": 0, "biofouling": 0, "hydrophobicity": 0, "durability": 0
    }
    
    df = pd.DataFrame([input_dict])
    # If we have the column list from the model, use it to sort/filter
    if model_columns is not None:
        return df[model_columns]
    return df

# ================= MODEL EXECUTION =================
# Track final state for plots
current_metrics = [0, 0, 0, 0] 

if st.button("Run Simulation"):
    try:
        results_card = st.empty()
        progress_bar = st.progress(0)
        cumulative_bio = 0
        
        t_range = range(1, days_input + 1) if run_sim else [days_input]
        
        for t in t_range:
            X = build_input(velocity, t)
            raw_pred = model.predict(X)[0]
            current_metrics = raw_pred # Update for charts
            
            drag_red = np.clip(raw_pred[0], 0, 95)
            daily_risk = max(0, raw_pred[1]) 
            cumulative_bio += daily_risk * 0.02
            total_bio = min(1.0, cumulative_bio)
            
            wear_factor = max(0.5, 1 - (t / 2000)) 
            hydro = max(0, raw_pred[2] * wear_factor)
            durability = max(0, raw_pred[3])

            with results_card.container():
                st.subheader(f"📊 Performance Analysis: Day {t}")
                if hydro > 145: st.success("✨ Status: Superhydrophobic (SHS)")
                elif hydro > 90: st.info("💧 Status: Hydrophobic")
                else: st.warning("⚠️ Status: Wetted (High Drag)")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Drag Reduc.", f"{drag_red:.1f}%")
                c2.metric("Bio-Accum.", f"{total_bio:.2f}", delta=f"{daily_risk:.3f}")
                c3.metric("Contact Angle", f"{hydro:.1f}°")
                c4.metric("Durability", f"{durability:.0f} pts")

            progress_bar.progress(t / days_input)
            if run_sim: time.sleep(speed)
            
    except Exception as e:
        st.error(f"Prediction Error: {e}")

# ================= 3D SURFACE GENERATION =================
resolution = 100
x = np.linspace(0, 5, resolution)
y = np.linspace(0, 5, resolution)
Xg, Yg = np.meshgrid(x, y)

hull_base = np.clip(1 - (Yg**2) / (1.5**2), 0, 1)
# Sharp Triangle Profile logic
phase = (2 * np.pi / riblet_spacing) * Xg
riblet = (2 * riblet_height / np.pi) * np.arcsin(np.sin(phase))

nano_freq = 45
nano_amp = 0.07 * lotus_intensity
np.random.seed(1)
lotus = nano_amp * (np.cos(nano_freq * Xg) * np.cos(nano_freq * Yg)) + (0.01 * np.random.randn(*Xg.shape))
Z = hull_base + riblet + lotus

# ================= ALL ORIGINAL VISUAL SECTIONS RESTORED =================

# 1. 3D Plot
st.subheader("3D Biomimetic Hull Surface (Multiscale)")
fig_3d = go.Figure(data=[go.Surface(x=Xg, y=Yg, z=Z, colorscale='Viridis')])
st.plotly_chart(fig_3d, use_container_width=True)

# 2. STL Export
st.subheader("Export STL for SimScale")
if st.button("Generate STL"):
    try:
        _, _, _, file_path = generate_stl(riblet_spacing, riblet_height, lotus_intensity, resolution=80)
        with open(file_path, "rb") as f:
            st.download_button("Download STL", f, "biomimetic_hull.stl")
    except Exception as e:
        st.error(f"STL generation failed: {e}")

# 3. Flow Field Analysis
st.subheader("Flow Interaction & Direction")
dZdx = np.gradient(Z, axis=1)
dZdy = np.gradient(Z, axis=0)
U = 1 - np.abs(dZdx) * 2
V = -dZdy * 0.5
velocity_field = np.sqrt(U**2 + V**2)

col1, col2 = st.columns(2)
with col1:
    fig_flow, ax_flow = plt.subplots()
    cf = ax_flow.contourf(Xg, Yg, velocity_field, levels=30, cmap='RdYlBu_r')
    plt.colorbar(cf, ax=ax_flow, label="Velocity Magnitude")
    st.pyplot(fig_flow)
with col2:
    step = 6
    fig_vec, ax_vec = plt.subplots()
    ax_vec.quiver(Xg[::step, ::step], Yg[::step, ::step], U[::step, ::step], V[::step, ::step], color='navy')
    st.pyplot(fig_vec)

# 4. Biofouling Zones
st.subheader("Biofouling Risk Zones")
attachment_zone = (velocity_field < np.percentile(velocity_field, 40)) & (np.abs(lotus) < np.percentile(np.abs(lotus), 40))
fig_bio, ax_bio = plt.subplots()
ax_bio.contourf(Xg, Yg, attachment_zone, cmap='Reds')
st.pyplot(fig_bio)

# 5. Drag Curve
st.subheader("Drag vs Velocity")
vels = np.linspace(0.5, 25, 40)
drag_curve = [max(0, model.predict(build_input(v, days_input))[0][0]) for v in vels]
fig_drag, ax_drag = plt.subplots()
ax_drag.plot(vels, drag_curve, 'r--')
ax_drag.set_xlabel("Velocity")
ax_drag.set_ylabel("Drag Reduction %")
st.pyplot(fig_drag)

# 6. Surface Comparison
st.subheader("Efficiency Comparison")
labels = ["Smooth", "Riblet Only", "Lotus Only", "Hybrid (Current)"]
values = [40, 65, 60, max(current_metrics[0], 0)]
fig_comp, ax_comp = plt.subplots()
ax_comp.bar(labels, values, color=['gray', 'blue', 'green', 'orange'])
ax_comp.set_ylim(0, 100)
st.pyplot(fig_comp)

# 7. Engineering Insight
st.subheader("Engineering Interpretation")
st.info("Uses Arcsin-based triangle riblets and Cassie-Baxter contact angle logic.")
if st.button("Show Details"):
    st.write("- **Riblets:** High aspect ratio reduces skin friction.\n- **Lotus:** Nano-roughness creates air pockets (plastrons).")
