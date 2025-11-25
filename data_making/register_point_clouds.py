import open3d as o3d
import numpy as np

# Load source and target point clouds
source = o3d.io.read_point_cloud("source.pcd")
target = o3d.io.read_point_cloud("target.pcd")

# Downsample for faster processing (optional)
source_down = source.voxel_down_sample(voxel_size=0.05)
target_down = target.voxel_down_sample(voxel_size=0.05)

# Estimate normals (required for some registration methods)
source_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
target_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# Optional: Compute FPFH features for better initial alignment
source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
    source_down,
    o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=100))
target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
    target_down,
    o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=100))

# Use RANSAC for initial coarse alignment
distance_threshold = 0.1
result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    source_down, target_down, source_fpfh, target_fpfh, True,
    distance_threshold,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
    4, [
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
    ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))

print("RANSAC result:")
print(result_ransac)
print("Transformation:")
print(result_ransac.transformation)

# Refine registration with ICP
result_icp = o3d.pipelines.registration.registration_icp(
    source, target, 0.05, result_ransac.transformation,
    o3d.pipelines.registration.TransformationEstimationPointToPlane())

print("ICP refinement result:")
print(result_icp)
print("Transformation:")
print(result_icp.transformation)

# Apply transformation to source point cloud
source.transform(result_icp.transformation)

# Visualize the aligned point clouds
o3d.visualization.draw_geometries([source.paint_uniform_color([1, 0, 0]), target.paint_uniform_color([0, 1, 0])])