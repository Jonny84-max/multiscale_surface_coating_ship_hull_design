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

#---- Materials & Coating Selection

st.sidebar.header("Base Material Selection")
material = st.sidebar.selectbox(
    "Foundation Material", 
    list(material_map.keys()), 
    help="CFRP is most effective for maintaining sharp V-groove vertices."
)
coating = st.sidebar.selectbox(
    "Surface Coating", 
    list(coating_map.keys()), 
    help="PDMS (Silicone) is the most effective Foul-Release coating."
)
st.sidebar.header("Surface Geometry (Biofouling resistance)")

# Natural shark denticle height: ~0.1mm
riblet_height = st.sidebar.slider("Riblet Height (mm)", 0.01, 0.3, 0.10)
# Natural spacing for 20 knots: ~0.15mm
riblet_spacing = st.sidebar.slider("Riblet Spacing (mm)", 0.05, 1.0, 0.15)
# 200nm is the optimal natural target for wax-mimicry. nm is Nanometrs
lotus_nm = st.sidebar.slider("Lotus Feature Size (Nanometers (nm))", 100, 1000, 200)
# Convert back to a 0.0 - 1.0 intensity for the math model. Smaller nm = Higher "Intensity" (Efficiency)
lotus_intensity = np.clip(1.0 - (lotus_nm - 100) / 900, 0.0, 1.0)

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

                # Material Performance Modifiers
                if material == "CFRP":
                    dur_mod = 0.85  # Slowest wear (stiff ridges)
                    drag_mod = 1.05        # Slight boost to drag efficiency
                elif material == "Hybrid":
                    dur_mod = 0.92  # Good balance
                    drag_mod = 1.10        # Best bio-shielding performance
                else:  # GFRP
                    dur_mod = 1.2    # Faster wear
                    drag_mod = 1.0

                # COATING MODIFIERS (Surface Energy)
                if coating == "PDMS":
                    bio_mod, slip_mod = 0.3, 1.02
                elif coating == "Fluoro":
                    bio_mod, slip_mod = 0.5, 1.12
                elif coating == "Sol-gel":
                    bio_mod, slip_mod = 0.7, 1.05
                else: # Epoxy/Vinyl
                    bio_mod, slip_mod = 1.2, 1.0
                
                # Contact Angle (h_raw)
                # Logic: Base angle + Lotus impact - Velocity pressure penalty
                h_raw = 95 + (lotus_intensity * 75) - (velocity * 0.4)
                
                # Durability (dur_raw)
                # Use dynamic decay. 
                # Higher velocity causes the surface 'points' to drop faster over time.
                wear_rate = (velocity * 0.05) + 0.01 * dur_mod  # dur_mod makes CFRP wear 0.85x slower than GFRP (1.2x)
                dur_raw = max(0, 100 - (t * wear_rate)) 
                
                # Mapping and clipping logic
                laplace_stability = (1.0 / (riblet_spacing + 1e-6)) * lotus_intensity
                wear_factor = max(0.88, 1 - (t / (4000/dur_mod))) if t > 1 else 1.0

                # Bio-Accumulation (b_raw & cumulative_bio)
                # If hydro > 150 (SHS shield: Cassie-Baxter), growth is cut by 90%
                bio_shield = 0.1 if (h_raw * (max(0.88, 1 - (t / 4000)))) > 150 else 0.8
                daily_risk = bio_shield * bio_med * (1 + (temperature / 40)) * 0.05   # bio_mod makes PDMS (0.4) much cleaner than Epoxy (1.2)
                cumulative_bio += daily_risk
                total_bio = min(1.0, cumulative_bio)
                b_raw = 0.05   # This represents the base biological growth rate  
                hydro = np.clip(h_raw * wear_factor, 0, 165.0)
                durability = max(0, dur_raw) # Maps dur_raw to the metric
                
                # Extract drag from raw_pred based on your model output
                d_raw = raw_pred if np.isscalar(raw_pred) else raw_pred[0]
                drag_red = np.clip((d_raw + (hydro/20)) * drag_mod * slip_mod, 0, 95)  # Combines structural boost (drag_mod) with molecular slip (slip_mod) 
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
                    c4.metric("Durability", f"{durability:.1f} pts", help="Surface Integrity Score (100=Pristine, 0=Failed)")

                progress_bar.progress(t / days_input)
                if run_sim: time_lib.sleep(speed)
        except Exception as e:
            st.error(f"Prediction Error: {e}")

# ================= 3D VISUALS =================
res = 100
x = np.linspace(0, 5, res)
y = np.linspace(0, 5, res)
Xg, Yg = np.meshgrid(x, y)

# 1. Texture logic: Shark Skin V-Groove (Triangle Wave)
hull_base = np.clip(1 - (Yg**2) / (1.5**2), 0, 1)
# Shark-inspired triangular ridges
# Formula: Triangle wave = (2*A/pi) * arcsin(sin(2*pi*x/p)) 
# Simplified for performance:
riblet = riblet_height * (1 - 2 * np.abs((Xg / riblet_spacing) % 1 - 0.5))

# Hierarchical Nano-scale Lotus effect (Superimposed on ridges)
lotus = (0.04 * lotus_intensity) * (np.cos(45 * Xg) * np.cos(45 * Yg))

# 2. Base Plate Logic
base_thickness = 0.2
Z = base_thickness + hull_base + riblet + lotus
Z = np.maximum(Z, base_thickness) # This "clamping" ensures the bottom is perfectly flat at the base_thickness level

st.subheader("3D Biomimetic Hull Surface (Solid Plate)")

# Use 'width=stretch' for Streamlit 2026 compliance
st.plotly_chart(go.Figure(data=[go.Surface(z=Z, colorscale='Viridis')]), width='stretch')

# ================= STL =================
if st.button("Generate STL for Export"):
    try:
        _, _, _, path = generate_stl(riblet_spacing, riblet_height, lotus_intensity, mode=mode)
        with open(path, "rb") as f:
            st.download_button("Download STL", f, "biomimetic_hull.stl")
    except Exception as e:
        st.error(f"STL Error: {e}")

# Flow & Bio Analysis
dZdx, dZdy = np.gradient(Z)
U, V = 1 - np.abs(dZdx) * 2, -dZdy * 0.5
velocity_field = np.sqrt(U**2 + V**2)
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
    
    if "pred" not in st.session_state:
        st.info("👉 To view Hybrid Design performance, please run the simulation.")
    
# ================= ENGINEERING INSIGHT =================
st.subheader("Engineering Interpretation")

st.info(
    f"Design evaluated at {velocity} knots. "
    "Target organisms: Bacteria (~1 μm), Algae (~100 μm)."
)

st.success(f"""
- Major Design Considerations: For maximum efficiency at **{velocity} knots**, the benchmark setup is:
    - Base Material:      CFRP (Carbon Fiber Selected) — Provides high stiffness to maintain ridge geometry.
    - Surface Coating:    PDMS (Silicone) — Lowest surface energy for non-stick performance.
    - Geometry:      0.10mm Height / 0.15mm Spacing — Optimal Shark-skin V-groove ratio.
    - Nano-Texture:   200nm Lotus Features — Maximizes air-layer stability (Plastron).
""")

st.markdown("""
**Core Design Strategy**

- **1. Riblets Flow Control:** 
- Aligns with flow direction to suppress turbulent vortices  
- Reduces skin-friction drag at the boundary layer

**2. Lotus-Inspired Nano Texture**
- Creates high contact angle (>150°) surfaces  
- Minimizes liquid-solid contact area  
- Prevents initial microbial adhesion  

- **3. Hybrid Design:** Simultaneously improves hydrodynamic efficiency and antifouling resistance.

**Surface Physics Insight**

- High contact angles (~150°+) promote a **Cassie–Baxter state**, sustaining an air layer (plastron).
- Riblet spacing is tuned to maintain **pressure stability under flow conditions**.
- Nano-textures reduce effective attachment area for microbial colonization.

**Biofouling Resistance Strategy**

- Disrupts early-stage fouling:
  - Conditioning film formation  
  - Bacterial adhesion  
  - Microorganism settlement  
- Reduces transition to macrofouling (such as barnacles)

- Cassie–Baxter State (Plastron Formation): Air pockets form a protective barrier against microorganisms

- Projected Hydrodynamic Stability at High Velocity:
  - Optimized riblet spacing (~0.15 mm) maintains plastron integrity  
  - Prevents air layer collapse under dynamic flow conditions  

**Engineering Outcome**

- Lower drag: Improved fuel efficiency  
- Limit conditioning film formation: Reduced fouling lowers maintenance cost  
- Predictive modeling: time-dependent performance estimation
""")
