import open3d as o3d  
import numpy as np 
from plyfile import PlyData, PlyElement  
# /home/hamit/Softwares/gaussian-splatting/output/5d0fc536-e/point_cloud/iteration_100000/point_cloud.ply
# pcd = o3d.io.read_point_cloud("filtered_point_cloud_new.ply")
# ply_path ="/home/hamit/Softwares/gaussian-splatting/output/114f3143-b/point_cloud/iteration_50000/point_cloud.ply"
ply_path ="/home/hamit/Downloads/point_cloud.ply"
pcd = o3d.io.read_point_cloud(ply_path)
points = np.asarray(pcd.points)
#(x < -0.2  && x > -3.2 && y < 2.3 && z < 1.85 && z >-4.5)
# Define coordinate intervals
# x0_min, x0_max = 0., 2.25
# y0_min, y0_max = 0.12, 2.55
# z0_min, z0_max = -4.5, 0.1

# # (x < 2.1  && x > -3.2 && y > 0.1 && y < 2.3 && z < 0 && z >-4.5)

# x_min, x_max = -3.3, -0.
# y_min, y_max = 0.12, 2.55
# z_min, z_max = -4.2, 2

# rotated and scaled 1.44
x_min, x_max = -3, 4.5
y_min, y_max = 2.05, 2.9
z_min, z_max = -6.1, -0.4








# Create boolean mask for vertices within intervals
mask = (
    (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
    (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
    (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
)





# Get indices of selected vertices
selected_indices = np.where(mask)[0]
#keep_indices = np.where(~mask)[0]
#
print(selected_indices)
# Create new point cloud with selected vertices
selected_pcd = pcd.select_by_index(selected_indices)
#pcd.paint_uniform_color([1, 0, 0])        # Red for original
#selected_pcd.paint_uniform_color([0, 1, 0])  # Green for filtered


axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
o3d.visualization.draw_geometries([selected_pcd, axes])

# exit()


# Load the PLY file
# plydata = PlyData.read('filtered_point_cloud_new.ply')
plydata =PlyData.read(ply_path)

vertex_data = plydata['vertex']

# Convert to structured array for easier manipulation
vertices = np.array(vertex_data.data)
# Create a boolean mask (True for vertices to keep)
keep_mask = np.zeros(len(vertices), dtype=bool)
keep_mask[selected_indices] = True  

filtered_vertices = vertices[keep_mask]

new_vertex_element = PlyElement.describe(
    filtered_vertices,
    'vertex'
)

# Create new PlyData object
new_plydata = PlyData([new_vertex_element])
new_plydata.write('filtered_point_cloud_rotated_scaled144_model_yplane.ply')  



