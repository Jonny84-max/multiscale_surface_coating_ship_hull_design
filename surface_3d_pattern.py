import numpy as np
from stl import mesh

def generate_stl(riblet_spacing, riblet_height, resolution=50, thickness=0.02):

    x = np.linspace(0, 5, resolution)
    y = np.linspace(0, 5, resolution)
    X, Y = np.meshgrid(x, y)

    riblets = riblet_height * np.sin((2*np.pi / riblet_spacing) * X)
    lotus = 0.05 * np.cos(6*X) * np.cos(6*Y)

    Z_top = riblets + lotus
    Z_bottom = Z_top * 0   # perfectly smooth inner hull

    top_vertices = np.column_stack((X.flatten(), Y.flatten(), Z_top.flatten()))
    bottom_vertices = np.column_stack((X.flatten(), Y.flatten(), Z_bottom.flatten()))

    vertices = np.vstack((top_vertices, bottom_vertices))

    nx, ny = X.shape
    faces = []

    for i in range(nx - 1):
        for j in range(ny - 1):
            v1 = i * ny + j
            v2 = (i + 1) * ny + j
            v3 = (i + 1) * ny + (j + 1)
            v4 = i * ny + (j + 1)

            faces.append([v1, v2, v3])
            faces.append([v1, v3, v4])

    offset = nx * ny

    for i in range(nx - 1):
        for j in range(ny - 1):
            v1 = offset + i * ny + j
            v2 = offset + (i + 1) * ny + j
            v3 = offset + (i + 1) * ny + (j + 1)
            v4 = offset + i * ny + (j + 1)

            faces.append([v4, v3, v2])
            faces.append([v4, v2, v1])

    faces = np.array(faces)

    surface = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))

    for i, f in enumerate(faces):
        for j in range(3):
            surface.vectors[i][j] = vertices[f[j]]

    file_path = "biomimetic_hull.stl"
    surface.save(file_path)

    return X, Y, Z_top, file_path
