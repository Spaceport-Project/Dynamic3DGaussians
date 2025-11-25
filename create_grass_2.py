import numpy as np
import random

def create_natural_grass_plane_ply(file_path, plane_width=10, plane_depth=10, num_gaussians=100000, max_sh_degree=3):
    """
    Generates a .ply file for a more natural-looking grass plane.
    - Darker color
    - More variation in color, height, and rotation.
    """
    
    vertices = []

    # --- Parameters for a More Natural Look ---
    
    # 1. Darker Base Color
    base_color_green = [0, 177, 50]
    
    # 2. Secondary Color for Variation (yellowish-brown)
    base_color_yellow = [110, 100, 30]
    color_variation = 20 # Reduced variation per-blade, as we now have two base colors

    # Opacity and Scale
    opacity = 0.999
    base_scale = [0.03, 0.1, 0.03] # Keep the scale that worked before
    
    # 3. Increased Height Variation
    scale_variation_xy = 0.005
    scale_variation_y = 0.04 # Increased from ~0.025 to give more height difference

    # SH Constant
    SH_C0 = 0.28209479177387814

    # SH Degree Calculation
    num_f_rest_coeffs = ((max_sh_degree + 1) ** 2 * 3) - 3 if max_sh_degree > 0 else 0

    # --- Generate Gaussians ---
    for i in range(num_gaussians):
        # Position
        x = random.uniform(-plane_width / 2, plane_width / 2)
        z = random.uniform(-plane_depth / 2, plane_depth / 2)
        y = base_scale[1] / 2
        
        # Color: 85% chance of being green, 15% chance of being yellow/brown
        if random.random() < 0.85:
            base_color = base_color_green
        else:
            base_color = base_color_yellow
            
        r = min(255, max(0, base_color[0] + random.randint(-color_variation, color_variation)))
        g = min(255, max(0, base_color[1] + random.randint(-color_variation, color_variation)))
        b = min(255, max(0, base_color[2] + random.randint(-color_variation, color_variation)))
        
        f_dc_0 = (r / 255.0 - 0.5) * SH_C0
        f_dc_1 = (g / 255.0 - 0.5) * SH_C0
        f_dc_2 = (b / 255.0 - 0.5) * SH_C0

        f_rest = [0.0] * num_f_rest_coeffs

        # Scale with more height variation
        sx = base_scale[0] + random.uniform(-scale_variation_xy, scale_variation_xy)
        sy = base_scale[1] + random.uniform(-scale_variation_y, scale_variation_y)
        sz = base_scale[2] + random.uniform(-scale_variation_xy, scale_variation_xy)
        log_sx, log_sy, log_sz = np.log(max(0.001, sx)), np.log(max(0.001, sy)), np.log(max(0.001, sz))

        # 4. Increased Rotational Variation
        angle = random.uniform(-np.pi / 6, np.pi / 6) # Increased range from pi/16 to pi/6
        qx, qy, qz, qw = 0, np.sin(angle / 2), 0, np.cos(angle / 2)

        # Opacity
        logit_opacity = np.log(opacity / (1.0 - opacity))

        # Assemble vertex data
        vertex = [x, y, z, 0, 0, 0, f_dc_0, f_dc_1, f_dc_2]
        vertex.extend(f_rest)
        vertex.extend([logit_opacity, log_sx, log_sy, log_sz, qx, qy, qz, qw])
        
        vertices.append(tuple(vertex))

    # --- Write to .ply File ---
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
        
    print(f"Successfully generated '{file_path}' with more natural variation.")

# --- Main Execution ---
if __name__ == "__main__":
    output_file = "grass_final.ply"
    create_natural_grass_plane_ply(output_file, num_gaussians=100000)