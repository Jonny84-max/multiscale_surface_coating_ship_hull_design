import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from stl import mesh
import tempfile

def export_stl(X, Y, Z):
    vertices = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
    faces = []

    nx, ny = X.shape

    for i in range(nx - 1):
        for j in range(ny - 1):
            v1 = i * ny + j
            v2 = (i + 1) * ny + j
            v3 = (i + 1) * ny + (j + 1)
            v4 = i * ny + (j + 1)

            faces.append([v1, v2, v3])
            faces.append([v1, v3, v4])

    faces = np.array(faces)

    surface = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))

    for i, f in enumerate(faces):
        for j in range(3):
            surface.vectors[i][j] = vertices[f[j]]

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".stl")
    surface.save(tmp.name)

    return tmp.name

# ================= LOAD MODEL =================
data = joblib.load("model.pkl")
model = data["model"]
columns = data["columns"]

st.title("Hybrid Biomimetic Hull Design System")

# ================= INPUT =================
st.sidebar.header("Design Inputs")

riblet_height = st.sidebar.slider("Riblet Height (mm)", 0.01, 0.3)
riblet_spacing = st.sidebar.slider("Riblet Spacing (mm)", 0.05, 1.0)
lotus_intensity = st.sidebar.slider("Lotus Intensity", 0.0, 1.0)
resolution = st.sidebar.slider("Mesh Resolution", 30, 200, 100)
velocity = st.sidebar.slider("Velocity", 0.5, 12.0)
temperature = st.sidebar.slider("Temperature", 0, 40)
salinity = st.sidebar.slider("Salinity", 10, 40)
time = st.sidebar.slider("Time (days)", 1, 1095)

material = st.sidebar.selectbox("Material", ["GFRP", "CFRP", "Hybrid"])
coating = st.sidebar.selectbox("Coating", ["Epoxy", "Vinyl", "PDMS", "Fluoro", "Sol-gel"])

material_map = {"GFRP": 0, "CFRP": 1, "Hybrid": 2}
coating_map = {"Epoxy": 0, "Vinyl": 1, "PDMS": 2, "Fluoro": 3, "Sol-gel": 4}

# ================= FEATURE BUILDER =================
def build_input(v):
    input_dict = {
        "riblet_height": riblet_height,
        "riblet_spacing": riblet_spacing,
        "lotus_intensity": lotus_intensity,
        "velocity": v,
        "temperature": temperature,
        "salinity": salinity,
        "time": time,
        "material": material_map[material],
        "coating": coating_map[coating]
    }
    return np.array([[input_dict[col] for col in columns]])

# ================= PREDICTION =================
if st.button("Run Simulation"):
    X = build_input(velocity)
    pred = model.predict(X)[0]

    st.subheader("Performance Results")
    st.metric("Drag Reduction", f"{pred[0]:.2f}")
    st.metric("Biofouling Risk", f"{pred[1]:.2f}")
    st.metric("Hydrophobicity", f"{pred[2]:.2f}")
    st.metric("Durability", f"{pred[3]:.2f}")

# ================= GRID =================
x = np.linspace(0, 5, resolution)
y = np.linspace(0, 5, resolution)
Xg, Yg = np.meshgrid(x, y)

# ================= HULL BASE =================
hull_base = 1 - (Yg**2) / (1.5**2)
hull_base = np.clip(hull_base, 0, 1)

# ================= RIBLETS =================
riblet = riblet_height * np.sin((2 * np.pi / riblet_spacing) * Xg)

# ================= LOTUS =================
lotus = lotus_intensity * (np.cos(8 * Xg) * np.cos(8 * Yg))

# ================= FINAL SURFACE =================
Z = hull_base + riblet + 0.3 * lotus

# ================= 3D SURFACE =================
st.subheader("3D Biomimetic Hull Surface")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(Xg, Yg, Z)
ax.set_title("Riblet + Lotus Hybrid Surface")
st.pyplot(fig)

# ================= INTERACTIVE 3D =================
st.subheader("Interactive 3D Viewer")

fig_plotly = go.Figure(data=[go.Surface(x=Xg, y=Yg, z=Z)])

fig_plotly.update_layout(
    scene=dict(
        xaxis_title="Flow Direction",
        yaxis_title="Hull Width",
        zaxis_title="Surface Height"
    )
)

st.plotly_chart(fig_plotly, use_container_width=True)

# ================= STL EXPORT =================
st.subheader("Export STL")

stl_file = export_stl(Xg, Yg, Z)

with open(stl_file, "rb") as f:
    st.download_button(
        label="Download STL for SimScale",
        data=f,
        file_name="biomimetic_hull.stl",
        mime="application/octet-stream"
    )
# ================= FLOW FIELD =================
st.subheader("Velocity Field")

velocity_field = 1 / (1 + np.abs(Z))

fig_flow, ax_flow = plt.subplots()
contour = ax_flow.contourf(Xg, Yg, velocity_field, levels=25)
plt.colorbar(contour)
ax_flow.set_title("Flow Distribution")
st.pyplot(fig_flow)

# ================= BIOFOULING =================
st.subheader("Biofouling Attachment Zones")

organism_size = 0.2
attachment_zone = np.abs(Z) < organism_size

fig_bio, ax_bio = plt.subplots()
ax_bio.contourf(Xg, Yg, attachment_zone, levels=1)
ax_bio.set_title("Attachment Risk Zones")
st.pyplot(fig_bio)

# ================= VECTOR FLOW =================
st.subheader("Flow Direction Field")

step = 8
U = -np.gradient(Z, axis=1)
V = -np.gradient(Z, axis=0)

fig_vec, ax_vec = plt.subplots()
ax_vec.quiver(Xg[::step, ::step], Yg[::step, ::step],
              U[::step, ::step], V[::step, ::step])
ax_vec.set_title("Flow Vectors")
st.pyplot(fig_vec)

# ================= DRAG CURVE =================
st.subheader("Drag vs Velocity")

vels = np.linspace(0.5, 12, 40)
drag_curve = []

for v in vels:
    X = build_input(v)
    drag_curve.append(model.predict(X)[0][0])

fig_drag, ax_drag = plt.subplots()
ax_drag.plot(vels, drag_curve)
ax_drag.set_xlabel("Velocity")
ax_drag.set_ylabel("Drag Reduction")
st.pyplot(fig_drag)

# ================= COMPARISON =================
st.subheader("Surface Comparison")

labels = ["Smooth", "Riblet", "Lotus", "Hybrid"]
smooth = 40
riblet_val = 65
lotus_val = 60
hybrid = pred[0] if 'pred' in locals() else 75

values = [smooth, riblet_val, lotus_val, hybrid]

fig_comp, ax_comp = plt.subplots()
ax_comp.bar(labels, values)
st.pyplot(fig_comp)

# ================= INSIGHT =================
st.subheader("Engineering Insight")

st.write("""
This system integrates riblet-induced drag reduction, lotus-inspired hydrophobicity,
and material-coating optimization to reduce biofouling and improve marine hull efficiency.
""")
