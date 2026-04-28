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
def load_assets():
    try:
        # Load the model and the feature list
        model = joblib.load("shs_predictive_model.pkl")
        feature_columns = joblib.load("feature_columns.pkl")
        return model, feature_columns
    except Exception as e:
        st.error(f"Asset Load Error: {e}")
        return None, None

model, feature_columns = load_assets()

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
    help="CFRP is Considered most effective for maintaining sharp V-groove vertices in this design."
)

coating = st.sidebar.selectbox(
    "Surface Coating",
    list(coating_map.keys()),
    help="PDMS (Silicone) is considered most effective for Foul-Release coating in this design."
)

st.sidebar.header("Surface Geometry (Biofouling resistance)")

riblet_height = st.sidebar.slider("Riblet Height (mm)", 0.01, 0.3, 0.10)
riblet_spacing = st.sidebar.slider("Riblet Spacing (mm)", 0.05, 1.0, 0.15)

lotus_nm = st.sidebar.slider("Lotus Feature Size (Nanometers (nm))", 100, 1000, 200)
lotus_intensity = np.clip(1.0 - (lotus_nm - 100) / 900, 0.0, 1.0)

st.sidebar.header("Operational Conditions")

velocity = st.sidebar.slider("Velocity (knots)", 0.5, 25.0, 20.0)
temperature = st.sidebar.slider("Temperature (°C)", 0, 40, 20)
salinity = st.sidebar.slider("Salinity (PSU)", 10, 40, 35)
days_input = st.sidebar.slider("Time (days)", 1, 1095, 30)
asp = riblet_height / (riblet_spacing + 1e-6)
f = 1 / (1 + asp)
cos_theta = -1 + f * (np.cos(np.radians(110)) + 1)
ca = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
mode = st.sidebar.selectbox("Output Mode", ["Visualization STL", "CFD STL"])
run_sim = st.sidebar.checkbox("Run Time Simulation")
speed = st.sidebar.slider("Simulation Speed", 0.1, 1.5, 0.4)

if 'pred' not in st.session_state:
    st.session_state.pred = None

# ================= DYNAMIC FEATURE ALIGNER =================
def build_input(v, t_val):           # This dictionary should match the feature_columns.pkl exactly
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
        "aspect_ratio": asp, # Using the global 'asp' defined above
        "multiscale_index": lotus_intensity * (1 - (riblet_height * 0.1)),
        "velocity_riblet_interact": v * asp, # 'v' is the input speed, 'asp' is global
        "solid_fraction_f": f,
        "cos_theta_eff": cos_theta,
        "effective_contact_angle": ca,
        "estimated_slip_length": (1 / (np.sqrt(f) + 1e-6)) * riblet_spacing
    }

    df = pd.DataFrame([full_data])
    return df.reindex(columns=feature_columns, fill_value=0)   # Reindex ensures the columns are in the EXACT order the model was trained on
	
# ================= MODEL EXECUTION =================
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

                # Material Performance
                if material == "CFRP":
                    dur_mod = 0.85
                    drag_mod = 1.05
                elif material == "Hybrid":
                    dur_mod = 0.92
                    drag_mod = 1.10
                else:
                    dur_mod = 1.2
                    drag_mod = 1.0

                # Coating Effects
                if coating == "PDMS":
                    bio_mod, slip_mod = 0.3, 1.02
                elif coating == "Fluoro":
                    bio_mod, slip_mod = 0.5, 1.12
                elif coating == "Sol-gel":
                    bio_mod, slip_mod = 0.7, 1.05
                else:
                    bio_mod, slip_mod = 1.2, 1.0

                h_raw = 95 + (lotus_intensity * 75) - (velocity * 0.4)

                wear_rate = ((velocity * 0.05) + 0.01) * dur_mod
                dur_raw = max(0, 100 - (t * wear_rate))

                wear_factor = max(0.88, 1 - (t / (4000 / dur_mod))) if t > 1 else 1.0

                bio_shield = 0.1 if (h_raw * wear_factor) > 150 else 0.8

                daily_risk = bio_shield * bio_mod * (1 + (temperature / 40)) * 0.05
                cumulative_bio += daily_risk
                total_bio = min(1.0, cumulative_bio)

                hydro = np.clip(h_raw * wear_factor, 0, 165.0)
                durability = max(0, dur_raw)

                d_raw = raw_pred

                drag_red = np.clip(
                    (d_raw + (hydro / 20)) * drag_mod * slip_mod,
                    0, 95
                )

                st.session_state.pred = {
    			"drag": drag_red,
    			"bio": total_bio,
    			"hydro": hydro,
    			"durability": durability
		}

                with results_card.container():

                    st.subheader(f"Hull Performance Analysis: Day {t}")

                    if hydro >= 150:
                        st.success("Status: Superhydrophobic (SHS) | Air Layer Stable")
                    elif hydro >= 90:
                        st.info("Status: Hydrophobic | Partial Wetting Risk")
                    else:
                        st.warning("Status: Hydrophilic | Air Layer Collapsed")

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Drag Reduc.", f"{drag_red:.1f}%")
                    c2.metric("Bio-Accum.", f"{total_bio:.2f}", delta=f"{daily_risk:.3f}")
                    c3.metric("Contact Angle", f"{hydro:.1f}°")
                    c4.metric(
                        label="Durability",
                        value=f"{durability:.1f} pts", 
                        help="Surface Integrity Score (100=Pristine, 0=Failed). Higher velocity and time cause wear on the riblet peaks."
                    )
                progress_bar.progress(t / days_input)

                if run_sim:
                    time_lib.sleep(speed)

        except Exception as e:
            st.error(f"Prediction Error: {e}")

# ================= 3D VISUALS =================
res = 100
x = np.linspace(0, 5, res)
y = np.linspace(0, 5, res)
Xg, Yg = np.meshgrid(x, y)

hull_base = np.clip(1 - (Yg**2) / (1.5**2), 0, 1)

riblet = riblet_height * (1 - 2 * np.abs((Xg / (riblet_spacing + 1e-6)) % 1 - 0.5))

lotus = (0.04 * lotus_intensity) * (np.cos(45 * Xg) * np.cos(45 * Yg))

base_thickness = 0.2
Z = base_thickness + hull_base + riblet + lotus
Z = np.maximum(Z, base_thickness)

st.subheader("3D Biomimetic Hull Surface (Solid Plate)")

st.plotly_chart(
    go.Figure(data=[go.Surface(z=Z, colorscale='Viridis')]),
    use_container_width=True
)

# ================= STL =================
if st.button("Generate STL for Export"):
    try:
        _, _, _, path = generate_stl(
            riblet_spacing,
            riblet_height,
            lotus_intensity,
            mode=mode
        )

        with open(path, "rb") as f:
            st.download_button("Download STL", f, "biomimetic_hull.stl")

    except Exception as e:
        st.error(f"STL Error: {e}")

# ================= FLOW + BIO =================
dZdx, dZdy = np.gradient(Z)
U, V = 1 - np.abs(dZdx) * 2, -dZdy * 0.5
velocity_field = np.sqrt(U**2 + V**2)

col1, col2 = st.columns(2)
extent = [0, 5, 0, 5]

with col1:
    st.write("Flow Field")
    fig_flow, ax_flow = plt.subplots()
    cf = ax_flow.contourf(Xg, Yg, velocity_field, levels=30, cmap='RdYlBu_r', extent=extent)
    plt.colorbar(cf, ax=ax_flow)
    st.pyplot(fig_flow)

with col2:
    st.write("Biofouling Risk Zones")
    attachment_zone = (velocity_field < np.percentile(velocity_field, 35))
    fig_bio, ax_bio = plt.subplots()
    ax_bio.imshow(attachment_zone, cmap='Reds', extent=extent, origin='lower')
    st.pyplot(fig_bio)

# ================= COMPARISON =================
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
        except:
            drags.append(0)

    fig3, ax3 = plt.subplots()
    ax3.plot(vels, drags, 'r--o')
    st.pyplot(fig3)

with c4:
    labels = ["Smooth (Base)", "Riblet Only", "Lotus Only", "Hybrid Design"]

    current_drag = st.session_state.pred["drag", 0] if st.session_state.pred else 0
    values = [0, 8.5, 5.2, current_drag]

    fig_comp, ax_comp = plt.subplots()
    bars = ax_comp.bar(labels, values)

    ax_comp.set_ylabel("Drag Reduction (%)")
    ax_comp.set_title("Performance vs. Standard Smooth Hull")
    ax_comp.set_ylim(0, 100)

    for bar in bars:
        yval = bar.get_height()
        ax_comp.text(bar.get_x() + bar.get_width()/2, yval + 1,
                      f'{yval:.1f}%', ha='center', va='bottom')

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
    - Base Material:  CFRP (Carbon Fiber Selected)
    - Surface Coating: PDMS (Silicone)
    - Geometry: 0.10mm Height / 0.15mm Spacing
    - Nano-Texture: 200nm Lotus Features
""")

st.markdown("""
**Core Design Strategy**

- **1. Riblets Flow Control:**
  - Aligns with flow direction to suppress turbulent vortices
  - Reduces skin-friction drag at the boundary layer

- **2. Lotus-Inspired Nano Texture**
  - Creates high contact angle (>150°) surfaces
  - Minimizes liquid-solid contact area
  - Prevents initial microbial adhesion

- **3. Hybrid Design:**
  Simultaneously improves hydrodynamic efficiency and antifouling resistance.

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

- Cassie–Baxter State (Plastron Formation):
  Air pockets form a protective barrier against microorganisms

- Projected Hydrodynamic Stability at High Velocity:
  - Optimized riblet spacing (~0.15 mm) maintains plastron integrity
  - Prevents air layer collapse under dynamic flow conditions

**Engineering Outcome**

- Lower drag: Improved fuel efficiency
- Limit conditioning film formation: Reduced fouling lowers maintenance cost
- Predictive modeling: time-dependent performance estimation
""")
