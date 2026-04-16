import numpy as np
from stl import mesh

def generate_stl(riblet_spacing, riblet_height, lotus_intensity, resolution=80, mode="Visualization STL"):

    np.random.seed(1)

    x = np.linspace(0, 5, resolution)
    y = np.linspace(0, 5, resolution)
    Xg, Yg = np.meshgrid(x, y)

    # Base hull
    hull_base = 1 - (Yg**2) / (1.5**2)
    hull_base = np.clip(hull_base, 0, 1)

    # Riblets
    riblet = riblet_height * np.sin((2 * np.pi / riblet_spacing) * Xg)

    # Lotus nano-scale
    nano_freq = 40
    nano_amp = 0.08 * lotus_intensity
    noise = 0.02 * np.random.randn(*Xg.shape)

    lotus = nano_amp * (np.cos(nano_freq * Xg) * np.cos(nano_freq * Yg)) + noise

    Z = hull_base + riblet + lotus

    # Build mesh
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
