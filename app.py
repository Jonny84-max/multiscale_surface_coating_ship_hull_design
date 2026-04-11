import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

model = joblib.load("model.pkl")

st.title("Hybrid Biomimetic Hull Design System")

# ================= INPUT =================
st.sidebar.header("Design Inputs")

riblet_height = st.sidebar.slider("Riblet Height (mm)", 0.01, 0.3)
riblet_spacing = st.sidebar.slider("Riblet Spacing (mm)", 0.05, 1.0)
lotus_intensity = st.sidebar.slider("Lotus Intensity", 0.0, 1.0)

velocity = st.sidebar.slider("Velocity", 0.5, 12.0)
temperature = st.sidebar.slider("Temperature", 0, 40)
salinity = st.sidebar.slider("Salinity", 10, 40)
time = st.sidebar.slider("Time (days)", 1, 1095)

material = st.sidebar.selectbox("Material", ["GFRP", "CFRP", "Hybrid"])
coating = st.sidebar.selectbox("Coating", ["Epoxy", "Vinyl", "PDMS", "Fluoro", "Sol-gel"])

material_map = {"GFRP": 0, "CFRP": 1, "Hybrid": 2}
coating_map = {"Epoxy": 0, "Vinyl": 1, "PDMS": 2, "Fluoro": 3, "Sol-gel": 4}

# ================= PREDICTION =================
if st.button("Run Simulation"):

    X = np.array([[riblet_height, riblet_spacing, lotus_intensity,
                   velocity, temperature, salinity, time,
                   material_map[material], coating_map[coating]]])

    pred = model.predict(X)[0]

    st.subheader("Performance Results")

    st.metric("Drag Reduction", f"{pred[0]:.2f}")
    st.metric("Biofouling Risk", f"{pred[1]:.2f}")
    st.metric("Hydrophobicity", f"{pred[2]:.2f}")
    st.metric("Durability", f"{pred[3]:.2f}")

# ================= SURFACE =================
st.subheader("Surface Microstructure")

x = np.linspace(0, 5, 100)
y = np.linspace(0, 5, 100)
Xg, Yg = np.meshgrid(x, y)

riblet = riblet_height * np.sin(6 * Xg)
lotus = lotus_intensity * np.random.rand(*Xg.shape)
Z = riblet + lotus

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(Xg, Yg, Z)
st.pyplot(fig)

# ================= FLOW FIELD =================
st.subheader("Flow Field (CFD-style)")

velocity_field = 1 / (1 + np.abs(Z))

fig2, ax2 = plt.subplots()
ax2.contourf(Xg, Yg, velocity_field, levels=20)
st.pyplot(fig2)

# ================= DRAG CURVE =================
st.subheader("Drag vs Velocity")

vels = np.linspace(0.5, 12, 40)
drag_curve = []

for v in vels:
    X = np.array([[riblet_height, riblet_spacing, lotus_intensity,
                   v, temperature, salinity, time,
                   material_map[material], coating_map[coating]]])
    drag_curve.append(model.predict(X)[0][0])

plt.figure()
plt.plot(vels, drag_curve)
st.pyplot(plt)

# ================= COMPARISON VIEW (NEW ADDITION) =================
st.subheader("Surface Performance Comparison")

labels = ["Smooth Hull", "Riblet Surface", "Lotus Surface", "Hybrid Biomimetic"]

smooth = 40
riblet = 65
lotus = 60
hybrid = pred[0] if 'pred' in locals() else 75

values = [smooth, riblet, lotus, hybrid]

plt.figure()
plt.bar(labels, values)
plt.ylabel("Performance Index")
st.pyplot(plt)

# ================= INSIGHT =================
st.subheader("Engineering Insight")

st.write("""
This system integrates riblet-based drag reduction, lotus-inspired hydrophobicity,
and composite material selection to optimize hull performance under marine conditions.
""")
