

import open3d as o3d
import numpy as np
import copy

# Load point clouds
source = o3d.io.read_point_cloud("/media/hamit/HamitsKingston/processed_data/2025-05-20_18-24-20_hamit_burak_1/colmap_input/points3d_cleaned.ply")
target = o3d.io.read_point_cloud("/media/hamit/HamitsKingston/processed_data/2025-05-20_18-24-20_hamit_burak_1/colmap_input/result_adjust_lumi_masked/points3d_cleaned_masked.ply")
# 2. Preprocessing
voxel_size = 0.05  # Adjust based on your data scale

# Downsample point clouds
source_down = source.voxel_down_sample(voxel_size)
target_down = target.voxel_down_sample(voxel_size)

# Estimate normals (required for FPFH features)
radius_normal = voxel_size * 2
# source_down.estimate_normals(
#     o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
# target_down.estimate_normals(
#     o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

# 3. Compute FPFH features
radius_feature = voxel_size * 5
source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
    source_down,
    o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
    target_down,
    o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

# 4. Global registration (RANSAC)
distance_threshold = voxel_size * 1.5
print("Running global registration...")
result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    source_down, target_down, source_fpfh, target_fpfh, True,
    distance_threshold,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
    3, [
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
    ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
print("Global registration result:", result_ransac)

# 5. Local refinement (ICP)
max_correspondence_distance = voxel_size * 2
result_icp = o3d.pipelines.registration.registration_icp(
    source_down, target_down, max_correspondence_distance, result_ransac.transformation,
    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
print("ICP registration result:", result_icp)

print("ICP refinement result:")
print(result_icp)
print("Transformation:")
print(repr(result_icp.transformation))

# 6. Apply final transformation
source_transformed = copy.deepcopy(source)
source_transformed.transform(result_icp.transformation)
ply_source = o3d.io.read_point_cloud("/media/hamit/HamitsKingston/processed_data/2025-05-20_18-24-20_hamit_burak_1/colmap_input/points3d.ply")

ply_transformed = ply_source.transform(result_icp.transformation)

# 7. Visualization
source_transformed.paint_uniform_color([1, 0, 0])  # Red
target.paint_uniform_color([0, 1, 0])             # Green
o3d.visualization.draw_geometries([source_transformed, target])

# (Optional) Save aligned point cloud
o3d.io.write_point_cloud("points3d_aligned.ply", ply_transformed)