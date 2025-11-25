import numpy as np
import open3d as o3d

# The transformation matrix from cam1 to cam2
# T_cam1_to_cam2 = np.array([[-0.61850017,  0.06039895, -0.78345995,  0.0778297 ],
#                            [ 0.02175688,  0.99797561,  0.05976061,  0.03480662],
#                            [ 0.7854834 ,  0.0199163 , -0.61856218,  0.05587208],
#                            [ 0.        ,  0.        ,  0.        ,  1.        ]])

T_cam1_to_cam2 = np.array([[ 0.99237707,  0.10753324,  0.06020258,  0.11220422],
                           [-0.10621229,  0.99403572, -0.02473713,  0.02746519],
                           [-0.06250357,  0.0181543 ,  0.99787961, -0.0608798 ],
                           [ 0.        ,  0.        ,  0.        ,  1.        ]])

# Load your point cloud - replace with your actual file path
pcd = o3d.io.read_point_cloud("/media/hamit/HamitsKingston/processed_data/2025-05-20_18-18-09_hamit_1/colmap_input/sparse/points3D.ply")

# Transform the point cloud
pcd_transformed = pcd.transform(T_cam1_to_cam2)

# Save the transformed point cloud
o3d.io.write_point_cloud("transformed_pointcloud.ply", pcd_transformed)

print(f"Original points: {len(pcd.points)}")
print(f"Transformed points: {len(pcd_transformed.points)}")