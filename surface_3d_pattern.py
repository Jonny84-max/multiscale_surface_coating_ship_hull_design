import numpy as np
from stl import mesh

def generate_refined_shs_stl(riblet_spacing, riblet_height, lotus_intensity, resolution=150):
    """
    Generates a 3D STL mesh for a biomimetic hull surface.
    Adjusted for Refined Physics: Uses triangular riblets and nano-scale lotus effects.
    """
    # 1. Set up the Grid
    # Increased resolution to 150 to capture the lotus-scale textures
    x = np.linspace(0, 5, resolution)
    y = np.linspace(0, 5, resolution)
    Xg, Yg = np.meshgrid(x, y)

    # 2. Base Hull Geometry
    # Circular hull profile
    hull_base = 1 - (Yg**2) / (1.5**2)
    hull_base = np.clip(hull_base, 0, 1)

    # 3. Refined Riblets (Triangular/Blade Profile)
    # Replaced np.sin with a triangle wave to match 'aspect_ratio' physics
    # Period is riblet_spacing; Amplitude is riblet_height
    phase = (2 * np.pi / riblet_spacing) * Xg
    riblet = (2 * riblet_height / np.pi) * np.arcsin(np.sin(phase))

    # 4. Lotus Nano-scale Texture
    # Frequency is high to simulate superhydrophobic bumps
    nano_freq = 50 
    nano_amp = 0.05 * lotus_intensity
    # Controlled noise to simulate manufacturing roughness
    noise = 0.01 * np.random.randn(*Xg.shape) * lotus_intensity

    lotus = nano_amp * (np.cos(nano_freq * Xg) * np.cos(nano_freq * Yg)) + noise

    # 5. Combine Layers
    Z = hull_base + riblet + lotus

    # 6. Build STL Mesh Structure
    vertices = []
    faces = []

    for i in range(resolution - 1):
        for j in range(resolution - 1):
            # Define 4 corners of a grid square
            v0 = [Xg[i,j], Yg[i,j], Z[i,j]]
            v1 = [Xg[i+1,j], Yg[i+1,j], Z[i+1,j]]
            v2 = [Xg[i,j+1], Yg[i,j+1], Z[i,j+1]]
            v3 = [Xg[i+1,j+1], Yg[i+1,j+1], Z[i+1,j+1]]

            idx = len(vertices)
            vertices.extend([v0, v1, v2, v3])

            # Two triangles per grid square
            faces.append([idx, idx+1, idx+2])
            faces.append([idx+1, idx+3, idx+2])

    vertices = np.array(vertices)
    faces = np.array(faces)

    # 7. Create the Mesh Object
    hull_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            hull_mesh.vectors[i][j] = vertices[f[j],:]

    # 8. Export for SimScale
    file_path = "refined_biomimetic_hull.stl"
    hull_mesh.save(file_path)

    print(f"STL Generated: {file_path}")
    print(f"Calculated Aspect Ratio: {riblet_height/riblet_spacing:.3f}")
    
    return Xg, Yg, Z, file_path

# --- EXAMPLE USAGE ---
# Testing with values from your refined dataset
generate_refined_shs_stl(riblet_spacing=0.5, riblet_height=0.2, lotus_intensity=0.8)
