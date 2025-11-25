# import numpy as np
# from plyfile import PlyData, PlyElement
# import os

# C0 = 0.28209479177387814

# def RGB2SH(rgb):
#     if rgb.max() > 1.0:  
#         rgb = rgb / 255.0  
#     return (rgb - 0.5) / C0
# def create_ply_from_splat_data(
#     points_file,
#     colors_dc_file, # Assuming f_dc_0, f_dc_1, f_dc_2 are in one file
#     log_scales_file,
#     logit_opacities_file,
#     rotations_file,
#     output_ply_file
# ):
#     """
#     Creates a .ply file for Gaussian splatting from .npy files.
#     """
#     # Load the data from .npy files
#     try:
#         points = np.load(points_file)
#         colors_dc = np.load(colors_dc_file)
#         log_scales = np.load(log_scales_file)
#         logit_opacities = np.load(logit_opacities_file)
#         rotations = np.load(rotations_file)
#         points = points[0]
#         colors_dc_logit = colors_dc[0]
#         log_scales = log_scales[0]
#         logit_opacities = logit_opacities[0]
#         rotations = rotations[0]

#     except FileNotFoundError as e:
#         print(f"Error: {e}. Please make sure all input .npy files exist.")
#         return

#     # --- Data Transformations ---
#     # Convert log scales to scales
#     scales = np.exp(log_scales)
    
#     # Convert logit opacities to opacities using the sigmoid function
#     opacities = 1 / (1 + np.exp(-logit_opacities))
#     colors_dc = 1 / (1 + np.exp(-colors_dc_logit)) 

#     sh = RGB2SH(colors_dc)
#     # sh = colors_dc - 0.5  

#     # Normalize rotation quaternions
#     rotations = rotations / np.linalg.norm(rotations, axis=1, keepdims=True)

#     # --- Prepare data for PLY file ---
#     num_points = points.shape[0]

#     sh_features = []
#     for i in range(45):
#         sh_features.append((f'f_rest_{i}', 'f4'))
    
#     # Define the structure of the PLY file's 'vertex' element
#     # dtype = [
#     #     ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
#     #     ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
#     #     ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
#     #     ('opacity', 'f4'),
#     #     ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
#     #     ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')
#     # ]

#     # --- UPDATED: Define the structure of the PLY file's 'vertex' element ---  
#     dtype = [  
#         ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),  
#         ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),  
#         ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),  
#         # Add the new SH features to the dtype  
#     ] + sh_features + [  
#         ('opacity', 'f4'),  
#         ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),  
#         ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')  
#     ]  
    
#     vertices = np.empty(num_points, dtype=dtype)

#     # Assign the data to the structured array
#     vertices['x'] = points[:, 0]
#     vertices['y'] = points[:, 1]
#     vertices['z'] = points[:, 2]

#     # Normals are often not used in Gaussian splatting, set to 0
#     vertices['nx'] = np.zeros(num_points, dtype=np.float32)
#     vertices['ny'] = np.zeros(num_points, dtype=np.float32)
#     vertices['nz'] = np.zeros(num_points, dtype=np.float32)

#     for i in range(45):
#         vertices[f'f_rest_{i}'] = np.zeros(num_points, dtype=np.float32)

#     vertices['f_dc_0'] = sh[:, 0]
#     vertices['f_dc_1'] = sh[:, 1]
#     vertices['f_dc_2'] = sh[:, 2]

#     vertices['opacity'] = opacities.squeeze()

#     vertices['scale_0'] = scales[:, 0]
#     vertices['scale_1'] = scales[:, 1]
#     vertices['scale_2'] = scales[:, 2]

#     # Note: The rotation quaternion is often stored as (w, x, y, z)
#     # The PLY format for some viewers might expect (x, y, z, w)
#     # Here we assume the order is rot_0, rot_1, rot_2, rot_3 as w, x, y, z
#     vertices['rot_0'] = rotations[:, 0] # w
#     vertices['rot_1'] = rotations[:, 1] # x
#     vertices['rot_2'] = rotations[:, 2] # y
#     vertices['rot_3'] = rotations[:, 3] # z

#     # --- Write to PLY file ---
#     vertex_element = PlyElement.describe(vertices, 'vertex')
#     ply_data = PlyData([vertex_element], text=False) # Use binary format for efficiency
#     ply_data.write(output_ply_file)
    
#     print(f"Successfully created PLY file at: {output_ply_file}")


# ### How to Use the Script

# # 1.  Save the code above as a Python file (e.g., `create_ply.py`).
# # 2.  Place the script in the same directory as your `.npy` files.
# # 3.  Modify the file names in the example usage section at the bottom of the script to match your files.
# # 4.  Run the script from your terminal: `python create_ply.py`

# # Here is an example of how you would call the function:

# # --- Example Usage ---

# # Create some dummy .npy files for demonstration if they don't exist
# # if not os.path.exists('points.npy'):
# #     print("Creating dummy .npy files for demonstration...")
# #     num_points = 100
# #     np.save('points.npy', np.random.rand(num_points, 3))
# #     # Assuming f_dc is a single file with 3 columns for f_dc_0, f_dc_1, f_dc_2
# #     np.save('f_dc.npy', np.random.rand(num_points, 3))
# #     np.save('log_scales.npy', np.random.rand(num_points, 3))
# #     np.save('logit_opacities.npy', np.random.rand(num_points, 1))
# #     # Rotations as quaternions (w, x, y, z)
# #     np.save('rotations.npy', np.random.rand(num_points, 4))

# # --- Call the function with your file paths ---
# path_files="/home/hamit/Softwares/Dynamic3DGaussians/output/2025-08-06_16-41-05_3412x2500_combin3_1_28sc_test3_contrast11_sharp05/2025-08-06_16-41-05_3412x2500_combin3_1_28sc/npy_files"
# create_ply_from_splat_data(
#     points_file=os.path.join(path_files,'means3D.npy'),
#     colors_dc_file=os.path.join(path_files,'rgb_colors.npy'),
#     log_scales_file=os.path.join(path_files,'log_scales.npy'),
#     logit_opacities_file=os.path.join(path_files,'logit_opacities.npy'),
#     rotations_file=os.path.join(path_files,'unnorm_rotations.npy'),
#     output_ply_file='output.ply'
# )


# import numpy as np
# import struct
# import os

# def create_threejs_compatible_ply(
#     points_file,
#     rgb_colors_file,
#     log_scales_file,
#     logit_opacities_file,
#     rotations_file,
#     output_ply_file
# ):
#     # Load data
#     points = np.load(points_file)
#     rgb_colors_logit = np.load(rgb_colors_file)
#     log_scales = np.load(log_scales_file)
#     logit_opacities = np.load(logit_opacities_file)
#     rotations = np.load(rotations_file)
#     points = points[2]
#     rgb_colors_logit = rgb_colors_logit[2]
#     log_scales = log_scales[2]
#     logit_opacities = logit_opacities[2]
#     rotations = rotations[2]

#     assert not np.any(np.isnan(points)), "Points contain NaN"  
#     assert not np.any(np.isnan(rgb_colors_logit)), "Colors contain NaN"  
#     assert not np.any(np.isnan(log_scales)), "Scales contain NaN"  
#     assert not np.any(np.isnan(logit_opacities)), "Opacities contain NaN"  
#     assert not np.any(np.isnan(rotations)), "Rotations contain NaN"  
#     num_points = points.shape[0]
    
#     # Convert RGB to SH DC coefficients
#     rgb_colors =  1.0 / (1.0 + np.exp(-rgb_colors_logit))
#     if rgb_colors.max() > 1.0:
#         colors_final = rgb_colors / 255.0
#     else:  
#         colors_final = rgb_colors.copy()  
    
#     colors_final = np.clip(colors_final, 0.0, 1.0)  
#     # if rgb_colors.max() > 1.0:
#     #     rgb_colors = rgb_colors / 255.0
    
    
#     SH_C0 = 0.28209479177387814
#     f_dc = (colors_final - 0.5) / SH_C0
#     # f_dc = colors_final * 2.0 - 1.0  # Convert [0,1] to [-1,1]
#     # Process other data
#     scales = np.exp(log_scales)
#     opacities = 1.0 / (1.0 + np.exp(-logit_opacities))

#     if opacities.ndim > 1:
#         opacities = opacities.squeeze()
#     rotations = rotations / np.linalg.norm(rotations, axis=1, keepdims=True)
    
#     # Write PLY file
#     with open(output_ply_file, 'wb') as f:
#         # Header
#         f.write(b'ply\n')
#         f.write(b'format binary_little_endian 1.0\n')
#         f.write(f'element vertex {num_points}\n'.encode())
        
#         # Properties in exact order
#         f.write(b'property float x\n')
#         f.write(b'property float y\n')
#         f.write(b'property float z\n')
#         f.write(b'property float nx\n')
#         f.write(b'property float ny\n')
#         f.write(b'property float nz\n')
#         f.write(b'property float f_dc_0\n')
#         f.write(b'property float f_dc_1\n')
#         f.write(b'property float f_dc_2\n')
        
#         for i in range(45):
#             f.write(f'property float f_rest_{i}\n'.encode())
        
#         f.write(b'property float opacity\n')
#         f.write(b'property float scale_0\n')
#         f.write(b'property float scale_1\n')
#         f.write(b'property float scale_2\n')
#         f.write(b'property float rot_0\n')
#         f.write(b'property float rot_1\n')
#         f.write(b'property float rot_2\n')
#         f.write(b'property float rot_3\n')
#         f.write(b'end_header\n')
        
#         # Binary data
#         for i in range(num_points):
#             # Position
#             f.write(struct.pack('<f', points[i, 0]))
#             f.write(struct.pack('<f', points[i, 1]))
#             f.write(struct.pack('<f', points[i, 2]))
            
#             # Normals (zeros)
#             f.write(struct.pack('<f', 0.0))
#             f.write(struct.pack('<f', 0.0))
#             f.write(struct.pack('<f', 0.0))
            
#             # SH DC coefficients
#             f.write(struct.pack('<f', f_dc[i, 0]))
#             f.write(struct.pack('<f', f_dc[i, 1]))
#             f.write(struct.pack('<f', f_dc[i, 2]))
            
#             # 45 f_rest coefficients (zeros)
#             for j in range(45):
#                 f.write(struct.pack('<f', 0.0))
            
#             # Opacity, scales, rotations
#             f.write(struct.pack('<f', opacities[i]))
#             f.write(struct.pack('<f', scales[i, 0]))
#             f.write(struct.pack('<f', scales[i, 1]))
#             f.write(struct.pack('<f', scales[i, 2]))
#             f.write(struct.pack('<f', rotations[i, 0]))
#             f.write(struct.pack('<f', rotations[i, 1]))
#             f.write(struct.pack('<f', rotations[i, 2]))
#             f.write(struct.pack('<f', rotations[i, 3]))
# path_files="/home/hamit/Softwares/Dynamic3DGaussians/output/2025-08-06_16-41-05_3412x2500_combin3_1_28sc_test3_contrast11_sharp05/2025-08-06_16-41-05_3412x2500_combin3_1_28sc/npy_files"
# create_threejs_compatible_ply(
#     points_file=os.path.join(path_files,'means3D.npy'),
#     rgb_colors_file=os.path.join(path_files,'rgb_colors.npy'),
#     log_scales_file=os.path.join(path_files,'log_scales.npy'),
#     logit_opacities_file=os.path.join(path_files,'logit_opacities.npy'),
#     rotations_file=os.path.join(path_files,'unnorm_rotations.npy'),
#     output_ply_file='output.ply'
# )



# import os
# import numpy as np
# import struct
# import sys
# from pathlib import Path

# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# def load_npy(path):
#     arr = np.load(path)
#     if hasattr(arr, 'astype'):  # ensure it's a NumPy array
#         return arr
#     return np.array(arr)

# def to_uint8_rgb(colors):
#     # Expect colors in [0,1]; clamp and scale
#     colors = np.clip(colors, 0.0, 1.0)
#     colors = (colors * 255.0).round().astype(np.uint8)
#     return colors

# def write_ply_ascii(points, colors, out_path, alphas=None, scales=None, extra_header_comments=None):
#     N = points.shape[0]
#     has_alpha = alphas is not None
#     has_scales = scales is not None

#     header_lines = [
#         "ply",
#         "format ascii 1.0",
#         f"element vertex {N}",
#         "property float x",
#         "property float y",
#         "property float z",
#         "property uchar red",
#         "property uchar green",
#         "property uchar blue",
#     ]
#     if has_alpha:
#         header_lines.append("property uchar alpha")
#     if has_scales:
#         header_lines += ["property float scale_x", "property float scale_y", "property float scale_z"]
#     if extra_header_comments:
#         for c in extra_header_comments:
#             header_lines.append(f"comment {c}")
#     header_lines.append("end_header")

#     with open(out_path, "w") as f:
#         f.write("\n".join(header_lines) + "\n")
#         if has_alpha and has_scales:
#             for (x, y, z), (r, g, b), a, (sx, sy, sz) in zip(points, colors, alphas, scales):
#                 f.write(f"{x:.7f} {y:.7f} {z:.7f} {int(r)} {int(g)} {int(b)} {int(a)} {sx:.7f} {sy:.7f} {sz:.7f}\n")
#         elif has_alpha:
#             for (x, y, z), (r, g, b), a in zip(points, colors, alphas):
#                 f.write(f"{x:.7f} {y:.7f} {z:.7f} {int(r)} {int(g)} {int(b)} {int(a)}\n")
#         elif has_scales:
#             for (x, y, z), (r, g, b), (sx, sy, sz) in zip(points, colors, scales):
#                 f.write(f"{x:.7f} {y:.7f} {z:.7f} {int(r)} {int(g)} {int(b)} {sx:.7f} {sy:.7f} {sz:.7f}\n")
#         else:
#             for (x, y, z), (r, g, b) in zip(points, colors):
#                 f.write(f"{x:.7f} {y:.7f} {z:.7f} {int(r)} {int(g)} {int(b)}\n")

# def write_ply_binary_little_endian(points, colors, out_path, alphas=None, scales=None, extra_header_comments=None):
#     import struct
#     N = points.shape[0]
#     has_alpha = alphas is not None
#     has_scales = scales is not None

#     header_lines = [
#         "ply",
#         "format binary_little_endian 1.0",
#         f"element vertex {N}",
#         "property float x",
#         "property float y",
#         "property float z",
#         "property uchar red",
#         "property uchar green",
#         "property uchar blue",
#     ]
#     if has_alpha:
#         header_lines.append("property uchar alpha")
#     if has_scales:
#         header_lines += ["property float scale_x", "property float scale_y", "property float scale_z"]
#     if extra_header_comments:
#         for c in extra_header_comments:
#             header_lines.append(f"comment {c}")
#     header_lines.append("end_header")
#     header = ("\n".join(header_lines) + "\n").encode("ascii")

#     with open(out_path, "wb") as f:
#         f.write(header)
#         if has_alpha and has_scales:
#             packer = struct.Struct("<fffBBBBfff")
#             for i in range(N):
#                 f.write(packer.pack(
#                     float(points[i,0]), float(points[i,1]), float(points[i,2]),
#                     int(colors[i,0]), int(colors[i,1]), int(colors[i,2]),
#                     int(alphas[i]),
#                     float(scales[i,0]), float(scales[i,1]), float(scales[i,2]),
#                 ))
#         elif has_alpha:
#             packer = struct.Struct("<fffBBBB")
#             for i in range(N):
#                 f.write(packer.pack(
#                     float(points[i,0]), float(points[i,1]), float(points[i,2]),
#                     int(colors[i,0]), int(colors[i,1]), int(colors[i,2]),
#                     int(alphas[i]),
#                 ))
#         elif has_scales:
#             packer = struct.Struct("<fffBBBfff")
#             for i in range(N):
#                 f.write(packer.pack(
#                     float(points[i,0]), float(points[i,1]), float(points[i,2]),
#                     int(colors[i,0]), int(colors[i,1]), int(colors[i,2]),
#                     float(scales[i,0]), float(scales[i,1]), float(scales[i,2]),
#                 ))
#         else:
#             packer = struct.Struct("<fffBBB")
#             for i in range(N):
#                 f.write(packer.pack(
#                     float(points[i,0]), float(points[i,1]), float(points[i,2]),
#                     int(colors[i,0]), int(colors[i,1]), int(colors[i,2]),
#                 ))

# def main(
#     points_path="points.npy",
#     colors_path="colors.npy",
#     log_scales_path="log_scales.npy",
#     logit_opacities_path="logit_opacities.npy",
#     rotations_path="rotations.npy",
#     out_path="output.ply",
#     ascii_ply=False,
#     include_alpha=True,
#     assume_quaternion_order="wxyz",  # or "xyzw"
# ):
#     # Load arrays
#     points = load_npy(points_path).astype(np.float32)  # (N,3)
#     colors = load_npy(colors_path).astype(np.float32)  # (N,3) in [0,1]
#     log_scales = load_npy(log_scales_path).astype(np.float32)  # (N,3)
#     logit_opacities = load_npy(logit_opacities_path).astype(np.float32)  # (N,) or (N,1)
#     rotations = load_npy(rotations_path).astype(np.float32)  # (N,4) quaternions
#     points = points[0]
#     colors =  sigmoid(colors[0])
#     log_scales = log_scales[0]
#     logit_opacities = logit_opacities[0]
#     rotations = rotations[0]


#     # Basic shape checks
#     N = points.shape[0]
#     assert points.shape[1] == 3, "points should be (N,3)"
#     assert colors.shape[0] == N and colors.shape[1] == 3, "colors should be (N,3)"
#     assert log_scales.shape[0] == N and log_scales.shape[1] == 3, "log_scales should be (N,3)"
#     if logit_opacities.ndim == 2 and logit_opacities.shape[1] == 1:
#         logit_opacities = logit_opacities[:, 0]
#     assert logit_opacities.shape[0] == N, "logit_opacities should be (N,) or (N,1)"
#     assert rotations.shape[0] == N and rotations.shape[1] == 4, "rotations should be (N,4)"

#     # Convert attributes
#     scales = np.exp(log_scales)  # (N,3)
#     opacities = sigmoid(logit_opacities)  # (N,)
#     rgb_u8 = to_uint8_rgb(colors)
#     alpha_u8 = None
#     if include_alpha:
#         alpha_u8 = (np.clip(opacities, 0.0, 1.0) * 255.0).round().astype(np.uint8)

#     # Normalize quaternions (if needed)
#     # Determine component order
#     if assume_quaternion_order not in ("wxyz", "xyzw"):
#         raise ValueError("assume_quaternion_order must be 'wxyz' or 'xyzw'")
#     if assume_quaternion_order == "xyzw":
#         # reorder to wxyz
#         rotations = rotations[:, [3, 0, 1, 2]]
#     # Normalize
#     norms = np.linalg.norm(rotations, axis=1, keepdims=True) + 1e-8
#     rotations = rotations / norms

#     # Add useful comments to header for provenance (non-standard but handy)
#     comments = [
#         "Generated from Gaussian Splatting params",
#         "Stored per-vertex: position (x y z), color (r g b), optional alpha",
#         "Original params available at generation time: scales (log->exp), opacities (logit->sigmoid), rotations (quats)",
#         f"Num points: {N}",
#     ]

#     # Write PLY
#     out_path = str(out_path)
#     if ascii_ply:
#         write_ply_ascii(points, rgb_u8, out_path, alphas=alpha_u8, scales=scales, extra_header_comments=comments)
#     else:
#         write_ply_binary_little_endian(points, rgb_u8, out_path, alphas=alpha_u8, scales=scales, extra_header_comments=comments)

#     print(f"Wrote {out_path} with {N} vertices.")

# if __name__ == "__main__":
#     # Example usage:
#     # python script.py (if you convert to a standalone script)
#     # Here we just call main with defaults assuming files exist in CWD.
#     path_files="/home/hamit/Softwares/Dynamic3DGaussians/output/2025-08-06_16-41-05_3412x2500_combin3_1_28sc_test3_contrast11_sharp05/2025-08-06_16-41-05_3412x2500_combin3_1_28sc/npy_files"

#     main(
#         points_path=os.path.join(path_files,'means3D.npy'),
#         colors_path=os.path.join(path_files,'rgb_colors.npy'),
#         log_scales_path=os.path.join(path_files,'log_scales.npy'),
#         logit_opacities_path=os.path.join(path_files,'logit_opacities.npy'),
#         rotations_path=os.path.join(path_files,'unnorm_rotations.npy'),
#         out_path='output.ply',
#         ascii_ply=True,          # set True for ASCII
#         include_alpha=True,       # include alpha from opacities
#         assume_quaternion_order="wxyz",  # change to "xyzw" if needed
#     )

#     # points_path=os.path.join(path_files,'means3D.npy'),
#     # colors_path=os.path.join(path_files,'rgb_colors.npy'),
#     # log_scales_path=os.path.join(path_files,'log_scales.npy'),
#     # logit_opacities_path=os.path.join(path_files,'logit_opacities.npy'),
#     # rotations_path=os.path.join(path_files,'unnorm_rotations.npy'),
#     # out_pathoutput_ply_file='output.ply'


import numpy as np
import os
import struct
from pathlib import Path

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# def load(path):
#     return np.asarray(np.load(path))

# def ensure_quat_wxyz(q, order="wxyz"):
#     if order == "xyzw":
#         q = q[:, [3,0,1,2]]
#     elif order != "wxyz":
#         raise ValueError("order must be 'wxyz' or 'xyzw'")
#     q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
#     return q

# def write_gsplat_ply_from_rgb(
#     points_path,
#     colors_path,           # colors_precomp in [0,1], shape (N,3)
#     log_scales_path,
#     logit_opacities_path,
#     rotations_path,
#     out_path="gaussians.ply",
#     quat_order="wxyz",
#     binary=True,
#     sh_degree=0,           # 0 => no f_rest; 3 => 24 f_rest coeffs (all zeros)
# ):
#     xyz = load(points_path).astype(np.float32)          # (N,3)
#     rgb = load(colors_path).astype(np.float32) 
#              # (N,3) assumed [0,1]
#     log_scales = load(log_scales_path).astype(np.float32)    # (N,3)
#     logit_opacity = load(logit_opacities_path).astype(np.float32)  # (N,) or (N,1)
#     rot = load(rotations_path).astype(np.float32)    
    
#     xyz = xyz[0]         # (N,3)
#     rgb =  sigmoid(rgb[0])
#     log_scales =  log_scales[0]    # (N,3)
#     logit_opacity = logit_opacity[0]
#     rot =   rot[0]
    
#        # (N,4)

#     N = xyz.shape[0]
#     assert xyz.shape == (N,3)
#     assert rgb.shape == (N,3)
#     assert log_scales.shape == (N,3)
#     assert rot.shape == (N,4)
#     if logit_opacity.ndim == 2 and logit_opacity.shape[1] == 1:
#         logit_opacity = logit_opacity[:,0]
#     assert logit_opacity.shape[0] == N

#     # Convert parameters
#     f_dc = np.clip(rgb, 0.0, 1.0).astype(np.float32)   # use RGB as DC features
#     # f_rest size for RGB SH of degree D: 3*((D+1)^2 - 1)
#     Crest = 3 * ((sh_degree + 1)**2 - 1)
#     if Crest > 0:
#         f_rest = np.zeros((N, Crest), dtype=np.float32)
#     else:
#         f_rest = np.zeros((N, 0), dtype=np.float32)

#     scale = np.exp(log_scales).astype(np.float32)
#     opacity = sigmoid(logit_opacity).astype(np.float32)
#     rot = ensure_quat_wxyz(rot, quat_order)

#     # Build header
#     header_lines = [
#         "ply",
#         "format binary_little_endian 1.0" if binary else "format ascii 1.0",
#         f"element vertex {N}",
#         "property float x",
#         "property float y",
#         "property float z",
#         "property float f_dc_0",
#         "property float f_dc_1",
#         "property float f_dc_2",
#     ]
#     for i in range(Crest):
#         header_lines.append(f"property float f_rest_{i}")
#     header_lines += [
#         "property float opacity",
#         "property float scale_0",
#         "property float scale_1",
#         "property float scale_2",
#         "property float rot_0",  # w
#         "property float rot_1",  # x
#         "property float rot_2",  # y
#         "property float rot_3",  # z
#         f"comment features_dc_dim 3",
#         f"comment features_rest_dim {Crest}",
#         "end_header",
#     ]

#     if binary:
#         header = ("\n".join(header_lines) + "\n").encode("ascii")
#         with open(out_path, "wb") as f:
#             f.write(header)
#             # 3 + 3 + Crest + 1 + 3 + 4 floats
#             fmt = "<" + "f" * (3 + 3 + Crest + 1 + 3 + 4)
#             packer = struct.Struct(fmt)
#             for i in range(N):
#                 row = (
#                     xyz[i,0], xyz[i,1], xyz[i,2],
#                     f_dc[i,0], f_dc[i,1], f_dc[i,2],
#                     *f_rest[i].tolist(),
#                     float(opacity[i]),
#                     scale[i,0], scale[i,1], scale[i,2],
#                     rot[i,0], rot[i,1], rot[i,2], rot[i,3],
#                 )
#                 f.write(packer.pack(*row))
#     else:
#         with open(out_path, "w") as f:
#             f.write("\n".join(header_lines) + "\n")
#             for i in range(N):
#                 vals = [
#                     f"{xyz[i,0]:.7f}", f"{xyz[i,1]:.7f}", f"{xyz[i,2]:.7f}",
#                     f"{f_dc[i,0]:.7f}", f"{f_dc[i,1]:.7f}", f"{f_dc[i,2]:.7f}",
#                     *[f"{v:.7f}" for v in f_rest[i]],
#                     f"{opacity[i]:.7f}",
#                     f"{scale[i,0]:.7f}", f"{scale[i,1]:.7f}", f"{scale[i,2]:.7f}",
#                     f"{rot[i,0]:.7f}", f"{rot[i,1]:.7f}", f"{rot[i,2]:.7f}", f"{rot[i,3]:.7f}",
#                 ]
#                 f.write(" ".join(vals) + "\n")

#     print(f"Wrote {out_path} with {N} gaussians. f_rest_dim={Crest} (zeros).")

# if __name__ == "__main__":
#     # Example usage: adjust paths to your .npy files
#     path_files="/home/hamit/Softwares/Dynamic3DGaussians/output/2025-08-06_16-41-05_3412x2500_combin3_1_28sc_test3_contrast11_sharp05/2025-08-06_16-41-05_3412x2500_combin3_1_28sc/npy_files"

#     write_gsplat_ply_from_rgb(
#         points_path=os.path.join(path_files,'means3D.npy'),
#         colors_path=os.path.join(path_files,'rgb_colors.npy'),
#         log_scales_path=os.path.join(path_files,'log_scales.npy'),
#         logit_opacities_path=os.path.join(path_files,'logit_opacities.npy'),
#         rotations_path=os.path.join(path_files,'unnorm_rotations.npy'),
#         out_path='output.ply',
#         quat_order="wxyz",    # change to "xyzw" if needed
#         binary=False,
#         sh_degree=0,          # set to 3 to emit 24 zero f_rest
#     )



import numpy as np
from plyfile import PlyData, PlyElement
path_files="/home/hamit/Softwares/Dynamic3DGaussians/output/2025-08-06_16-41-05_3412x2500_combin3_1_28sc_test3_contrast11_sharp05/2025-08-06_16-41-05_3412x2500_combin3_1_28sc/npy_files"

 
# Load your npy files (update paths if needed)
points = np.load(os.path.join(path_files,'means3D.npy'))[0]  # (N, 3)
colors = sigmoid(np.load(os.path.join(path_files,'rgb_colors.npy'))[0])  # (N, 3), assuming RGB in [0,1]
scales_log = np.load(os.path.join(path_files,'log_scales.npy'))[0]  # (N, 3)
opacities_logit = np.load(os.path.join(path_files,'logit_opacities.npy'))[0].reshape(-1, 1)  # (N, 1)
rotations = np.load(os.path.join(path_files,'unnorm_rotations.npy'))[0]  # (N, 4)
scales_log = np.clip(scales_log, -10, -0.5)  # 
opacities = np.clip(1 / (1 + np.exp(-opacities_logit)), 0.01, 0.99) 
print(f"Points: shape={points.shape}, min={points.min()}, max={points.max()}")
print(f"Colors: shape={colors.shape}, min={colors.min()}, max={colors.max()}")  # Should be [0,1] for RGB
print(f"Scales_log: shape={scales_log.shape}, min={scales_log.min()}, max={scales_log.max()}")
print(f"Opacities_logit: shape={opacities_logit.shape}, min={opacities_logit.min()}, max={opacities_logit.max()}")
print(f"Rotations: shape={rotations.shape}, min={rotations.min()}, max={rotations.max()}")

# Quick checks
if colors.min() < 0 or colors.max() > 1:
    print("Warning: Colors not in [0,1]—normalize if needed (e.g., divide by 255 if [0,255]).")
if np.any(np.isinf(np.exp(scales_log))):
    print("Warning: Infinite scales—clamp logs before exp.")
# Compute derived values
N = points.shape[0]
scales = np.exp(scales_log)  # (N, 3)
# opacities = 1 / (1 + np.exp(-opacities_logit))  # sigmoid, (N, 1)

# Normalize rotations to unit quaternions
rot_norms = np.linalg.norm(rotations, axis=1, keepdims=True)
rotations = rotations / np.maximum(rot_norms, 1e-6)  # Avoid div by zero

# Convert RGB to SH (DC only, rest zeros) for Gaussian Splatting format
C0 = 0.28209479177387814
sh_dc = (colors - 0.5) / C0  # (N, 3)
sh_dc = np.clip(sh_dc, -10, 10)  
sh_rest = np.zeros((N, 45))  # 15 per channel * 3 = 45, all zero for constant color

# Dummy normals (often set to 0 in GS PLY)
normals = np.zeros((N, 3))

# Combine all data into a structured array
vertex_data = np.hstack([
    points.astype(np.float32),          # x, y, z
    normals.astype(np.float32),         # nx, ny, nz (dummy)
    sh_dc.astype(np.float32),           # f_dc_0,1,2
    sh_rest.astype(np.float32),         # f_rest_0 to _44
    opacities.astype(np.float32),       # opacity
    scales.astype(np.float32),          # scale_0,1,2
    rotations.astype(np.float32)        # rot_0,1,2,3
])

# Define dtype for PLY (62 floats total per vertex)
properties = ['x', 'y', 'z', 'nx', 'ny', 'nz'] + \
             [f'f_dc_{i}' for i in range(3)] + \
             [f'f_rest_{i}' for i in range(45)] + \
             ['opacity'] + \
             [f'scale_{i}' for i in range(3)] + \
             [f'rot_{i}' for i in range(4)]

dtype = [(prop, 'f4') for prop in properties]
vertex_array = np.empty(N, dtype=dtype)
for i, prop in enumerate(properties):
    vertex_array[prop] = vertex_data[:, i]

# Create PLY element and write to file
vertex_element = PlyElement.describe(vertex_array, 'vertex')
PlyData([vertex_element], text=False).write('output.ply')

N_small = 10000
small_array = vertex_array[:N_small]
small_element = PlyElement.describe(small_array, 'vertex')
PlyData([small_element], text=False, byte_order='<').write('output_small.ply')
print(f"Created output_small.ply with {N_small} Gaussians for quick testing!")

print(f"Created output.ply with {N} Gaussians!")
   