import numpy as np
import random

def create_grass_plane_full_sh_ply(file_path, plane_width=10, plane_depth=10, num_gaussians=50000, max_sh_degree=3):
    """
    Generates a .ply file for a grass plane with full Spherical Harmonics (SH)
    up to a specified degree.
    """
    
    vertices = []

    # --- Parameters ---
    base_color = [40, 180, 99]
    color_variation = 25
    opacity = 0.999
    base_scale = [0.03, 0.1, 0.03] #[0.01, 0.1, 0.01]

    scale_variation = 0.005
    SH_C0 = 0.28209479177387814

    # --- SH Degree Calculation ---
    if max_sh_degree > 0:
        num_sh_coeffs = (max_sh_degree + 1) ** 2
        # We have 3 DC components (f_dc_0, 1, 2) and the rest are f_rest
        num_f_rest_coeffs = (num_sh_coeffs * 3) - 3
    else:
        num_f_rest_coeffs = 0

    # --- Generate Gaussians ---
    for i in range(num_gaussians):
        # Position
        x = random.uniform(-plane_width / 2, plane_width / 2)
        z = random.uniform(-plane_depth / 2, plane_depth / 2)
        y = base_scale[1] / 2
        
        # Color and SH DC components
        r = min(255, max(0, base_color[0] + random.randint(-color_variation, color_variation)))
        g = min(255, max(0, base_color[1] + random.randint(-color_variation, color_variation)))
        b = min(255, max(0, base_color[2] + random.randint(-color_variation, color_variation)))
        
        f_dc_0 = (r / 255.0 - 0.5) * SH_C0
        f_dc_1 = (g / 255.0 - 0.5) * SH_C0
        f_dc_2 = (b / 255.0 - 0.5) * SH_C0

        # Higher order SH components (f_rest) are set to 0 for diffuse objects
        f_rest = [0.0] * num_f_rest_coeffs

        # Scale
        sx = base_scale[0] + random.uniform(-scale_variation, scale_variation)
        sy = base_scale[1] + random.uniform(-scale_variation * 5, scale_variation * 5)
        sz = base_scale[2] + random.uniform(-scale_variation, scale_variation)
        log_sx, log_sy, log_sz = np.log(sx), np.log(sy), np.log(sz)

        # Rotation
        angle = random.uniform(-np.pi / 16, np.pi / 16)
        qx, qy, qz, qw = 0, np.sin(angle / 2), 0, np.cos(angle / 2)

        # Opacity
        logit_opacity = np.log(opacity / (1.0 - opacity))

        # Assemble vertex data
        vertex = [x, y, z, 0, 0, 0, f_dc_0, f_dc_1, f_dc_2]
        vertex.extend(f_rest)
        vertex.extend([logit_opacity, log_sx, log_sy, log_sz, qx, qy, qz, qw])
        
        vertices.append(tuple(vertex))

    # --- Write to .ply File ---
    
    # Generate the header dynamically based on SH degree
    header = f"""ply
format binary_little_endian 1.0
element vertex {len(vertices)}
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float f_dc_0
property float f_dc_1
property float f_dc_2
"""
    for i in range(num_f_rest_coeffs):
        header += f"property float f_rest_{i}\n"
    
    header += """property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
"""

    # Generate the dtype for the structured array dynamically
    dtype_list = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4')
    ]
    dtype_list.extend([(f'f_rest_{i}', 'f4') for i in range(num_f_rest_coeffs)])
    dtype_list.extend([
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')
    ])
    dtype = np.dtype(dtype_list)

    structured_array = np.array(vertices, dtype=dtype)
    
    with open(file_path, 'wb') as f:
        f.write(header.encode('utf-8'))
        f.write(structured_array.tobytes())
        
    print(f"Successfully generated '{file_path}' with {len(vertices)} Gaussians (SH degree {max_sh_degree}).")

# --- Main Execution ---
if __name__ == "__main__":
    output_file = "grass_sh_deg3.ply"
    # Set the desired max_sh_degree here
    create_grass_plane_full_sh_ply(output_file, max_sh_degree=3, num_gaussians=100000)