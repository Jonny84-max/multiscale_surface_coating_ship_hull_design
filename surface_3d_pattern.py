import numpy as np
from stl import mesh

def generate_stl(riblet_spacing, riblet_height, resolution=40, thickness=0.05):

    x = np.linspace(0, 0.7, resolution)
    y = np.linspace(0, 0.7, resolution)
    X, Y = np.meshgrid(x, y)

    riblets = riblet_height * np.sin((2*np.pi / riblet_spacing) * X)
    lotus = 0.05 * np.cos(6*X) * np.cos(6*Y)

    Z_top = riblets + lotus

    if mode == "Visualization STL":
    # ONLY TOP SURFACE
    vertices = np.column_stack((X.flatten(), Y.flatten(), Z_top.flatten()))

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

else:
    # CFD MODE → FULL SOLID BODY
    top_vertices = np.column_stack((X.flatten(), Y.flatten(), Z_top.flatten()))

    vertices = np.vstack((top_vertices))

    nx, ny = X.shape
    faces = []

    # TOP
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

    file_path = "biomimetic_hull.stl"
    surface.save(file_path)

    return X, Y, Z_top, file_path
