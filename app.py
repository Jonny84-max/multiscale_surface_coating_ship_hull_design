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
st.title("🚢 Hybrid Biomimetic Hull Design System")

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
mode = st.sidebar.selectbox("Output Mode", ["Visualization STL", "CFD STL"])

run_sim = st.sidebar.checkbox("Run Time Simulation")
speed = st.sidebar.slider("Simulation Speed (sec per step)", 0.1, 1.5, 0.4)

if 'pred' not in st.session_state:
    st.session_state.pred = None

# ================= DYNAMIC FEATURE ALIGNER =================
def build_input(v, t_val):
    # 1. Calculate all possible physics features
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
    
    # 2. IDENTIFY REQUIRED COLUMNS: Ask the model what it wants
    if model is not None:
        if hasattr(model, 'feature_names_in_'):
            required_cols = list(model.feature_names_in_)
        elif columns is not None:
            required_cols = columns
        else:
            # Emergency fallback to base features
            required_cols = ["riblet_height", "riblet_spacing", "lotus_intensity", "velocity", "time"]

        # 3. SELECT AND REORDER: Only take what the model recognizes
        # If a required column is missing from our dict, we initialize it as 0
        final_df = pd.DataFrame()
        for col in required_cols:
            if col in df.columns:
                final_df[col] = df[col]
            else:
                final_df[col] = [0.0]
        return final_df
    
    return df

# ================= MODEL EXECUTION =================
if st.button("Run Simulation"):
    if model is None:
        st.error("Model not loaded. Check 'shs_predictive_model.pkl'.")
    else:
        try:
            results_card = st.empty()
            progress_bar = st.progress(0)
            cumulative_bio = 0
            
            t_range = range(1, days_input + 1) if run_sim else [days_input]
            
            for t in t_range:
                X = build_input(velocity, t)
                raw_pred = model.predict(X)[0]
                
                # Check for scalar vs array prediction
                if isinstance(raw_pred, (list, np.ndarray)) and len(raw_pred) >= 4:
                    d_raw, b_raw, h_raw, dur_raw = raw_pred[0], raw_pred[1], raw_pred[2], raw_pred[3]
                else:
                    d_raw = raw_pred if np.isscalar(raw_pred) else raw_pred[0]
                    b_raw, h_raw, dur_raw = 0.05, 110.0, 75.0 
                
                drag_red = np.clip(d_raw, 0, 95)
                daily_risk = max(0, b_raw)
                cumulative_bio += daily_risk * 0.02
                total_bio = min(1.0, cumulative_bio)
                wear_factor = max(0.5, 1 - (t / 2000))
                hydro = max(0, h_raw * wear_factor)
                durability = max(0, dur_raw)

                st.session_state.pred = [drag_red, total_bio, hydro, durability]

                with results_card.container():
                    st.subheader(f"📊 Hull Performance Analysis: Day {t}")
                    if hydro > 150: st.success("✨ Surface Status: Superhydrophobic (SHS)")
                    elif hydro > 90: st.info("💧 Surface Status: Hydrophobic")
                    else: st.warning("⚠️ Surface Status: Hydrophilic (High Drag)")

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Drag Reduc.", f"{drag_red:.1f}%")
                    c2.metric("Bio-Accum.", f"{total_bio:.2f}", delta=f"{daily_risk:.3f}")
                    c3.metric("Contact Angle", f"{hydro:.1f}°")
                    c4.metric("Durability", f"{durability:.0f} pts")

                progress_bar.progress(t / days_input)
                if run_sim: time_lib.sleep(speed)
        except Exception as e:
            st.error(f"Prediction Error: {e}")

# ================= SURFACE GENERATION =================
resolution = 100
x = np.linspace(0, 5, resolution)
y = np.linspace(0, 5, resolution)
Xg, Yg = np.meshgrid(x, y)
hull_base = np.clip(1 - (Yg**2) / (1.5**2), 0, 1)
riblet = riblet_height * np.sin((2 * np.pi / riblet_spacing) * Xg)
nano_amp = 0.08 * lotus_intensity
lotus = nano_amp * (np.cos(40 * Xg) * np.cos(40 * Yg)) + (0.02 * np.random.randn(resolution, resolution))
Z = hull_base + riblet + lotus

st.subheader("3D Biomimetic Hull Surface (Multiscale)")
st.plotly_chart(go.Figure(data=[go.Surface(x=Xg, y=Yg, z=Z, colorscale='Viridis')]), width='stretch')

# ================= FLOW ANALYSIS =================
col_f1, col_f2 = st.columns(2)
dZdx, dZdy = np.gradient(Z)
U, V = 1 - np.abs(dZdx) * 2, -dZdy * 0.5
velocity_field = np.sqrt(U**2 + V**2)

with col_f1:
    st.subheader("Flow Interaction Field")
    fig_flow, ax_flow = plt.subplots()
    ax_flow.contourf(Xg, Yg, velocity_field, levels=30, cmap='RdYlBu_r')
    st.pyplot(fig_flow)

with col_f2:
    st.subheader("Biofouling Risk Zones")
    attachment_zone = (velocity_field < np.percentile(velocity_field, 40))
    fig_bio, ax_bio = plt.subplots()
    ax_bio.contourf(Xg, Yg, attachment_zone, cmap='Reds')
    st.pyplot(fig_bio)

# ================= DRAG CURVE =================
st.subheader("Drag vs Velocity")
vels = np.linspace(0.5, 25.0, 20)
drag_curve = []
for v in vels:
    try:
        X_v = build_input(v, days_input)
        p = model.predict(X_v)[0]
        drag_val = p[0] if not np.isscalar(p) else p
        drag_curve.append(max(drag_val, 0))
    except: drag_curve.append(0)

fig_drag, ax_drag = plt.subplots()
ax_drag.plot(vels, drag_curve, 'r--o')
ax_drag.set_xlabel("Velocity"); ax_drag.set_ylabel("Drag Reduction %")
st.pyplot(fig_drag)

# ================= COMPARISON =================
st.subheader("Surface Comparison")
labels = ["Smooth", "Riblet", "Lotus", "Hybrid"]
current_drag = st.session_state.pred[0] if st.session_state.pred else 0
values = [0, 8.5, 5.2, current_drag]

fig_comp, ax_comp = plt.subplots()
ax_comp.bar(labels, values, color=['gray', 'blue', 'green', 'orange'])
ax_comp.set_ylabel("Drag Reduction %")
ax_comp.set_ylim(0, 100)
if st.session_state.pred is None:
    ax_comp.text(3, 10, "Run Simulation\nto see Hybrid", ha='center', color='red')
st.pyplot(fig_comp)

# ================= INSIGHT =================
st.subheader("Engineering Insight")
st.info("This system integrates riblet-induced drag reduction and lotus-inspired nano-scale hydrophobicity.")

if st.button("Show Engineering Interpretation"):
    st.write("""
    - **Riblets:** Control boundary layer flow to reduce turbulent drag by minimizing cross-stream vortex motion.
    - **Lotus Effect:** Nano-textures minimize the liquid-solid contact area (Cassie-Baxter state), preventing bio-adhesion.
    - **Hybrid Advantage:** Combined geometric and chemical properties address both fluid dynamics and biological fouling.
    """)

# ================= STL EXPORT =================
st.divider()
if st.button("Generate STL"):
    try:
        _, _, _, path = generate_stl(riblet_spacing, riblet_height, lotus_intensity, mode=mode)
        with open(path, "rb") as f:
            st.download_button("Download STL File", f, "hull_design.stl")
    except Exception as e:
        st.error(f"STL Error: {e}")
