import numpy as np
from stl import mesh

def generate_stl(riblet_spacing, riblet_height, lotus_intensity, resolution=150, mode="Visualization STL"):
    """
    Generates a 3D STL mesh for a biomimetic hull surface.
    Uses triangular riblets to match the refined physics dataset.
    """
    x = np.linspace(0, 5, resolution)
    y = np.linspace(0, 5, resolution)
    Xg, Yg = np.meshgrid(x, y)

    # Base Hull Geometry
    hull_base = np.clip(1 - (Yg**2) / (1.5**2), 0, 1)

    # Sharp Triangular Riblets
    phase = (2 * np.pi / riblet_spacing) * Xg
    riblet = (2 * riblet_height / np.pi) * np.arcsin(np.sin(phase))

    # Lotus Nano-texture
    nano_freq = 50 
    nano_amp = 0.05 * lotus_intensity
    np.random.seed(42)
    noise = 0.01 * np.random.randn(*Xg.shape) * lotus_intensity
    lotus = nano_amp * (np.cos(nano_freq * Xg) * np.cos(nano_freq * Yg)) + noise

    Z = hull_base + riblet + lotus

    # STL Mesh Construction
    vertices = []
    faces = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            v0 = [Xg[i,j], Yg[i,j], Z[i,j]]
            v1 = [Xg[i+1,j], Yg[i+1,j], Z[i+1,j]]
            v2 = [Xg[i,j+1], Yg[i,j+1], Z[i,j+1]]
            v3 = [Xg[i+1,j+1], Yg[i+1,j+1], Z[i+1,j+1]]
            idx = len(vertices)
            vertices.extend([v0, v1, v2, v3])
            faces.append([idx, idx+1, idx+2])
            faces.append([idx+1, idx+3, idx+2])

    vertices = np.array(vertices)
    faces = np.array(faces)
    hull_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            hull_mesh.vectors[i][j] = vertices[f[j],:]

    file_path = "biomimetic_hull.stl"
    hull_mesh.save(file_path)
    return Xg, Yg, Z, file_path
