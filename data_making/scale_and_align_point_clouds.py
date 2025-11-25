import open3d as o3d
import numpy as np
import copy

def estimate_scale_factor(source, target, method='bbox'):
    """
    Estimate scale factor between two point clouds
    """
    if method == 'bbox':
        # Method 1: Using bounding box diagonal
        source_bbox = source.get_axis_aligned_bounding_box()
        target_bbox = target.get_axis_aligned_bounding_box()
        
        source_diagonal = np.linalg.norm(source_bbox.get_extent())
        target_diagonal = np.linalg.norm(target_bbox.get_extent())
        
        scale_factor = target_diagonal / source_diagonal
        
    elif method == 'centroid_distance':
        # Method 2: Using average distance from centroid
        source_points = np.asarray(source.points)
        target_points = np.asarray(target.points)
        
        source_centroid = np.mean(source_points, axis=0)
        target_centroid = np.mean(target_points, axis=0)
        
        source_avg_dist = np.mean(np.linalg.norm(source_points - source_centroid, axis=1))
        target_avg_dist = np.mean(np.linalg.norm(target_points - target_centroid, axis=1))
        
        scale_factor = target_avg_dist / source_avg_dist
    
    return scale_factor

def apply_scale_transform(pcd, scale_factor):
    """
    Apply uniform scaling to point cloud
    """
    scaled_pcd = copy.deepcopy(pcd)
    scaled_pcd.scale(scale_factor, center=scaled_pcd.get_center())
    return scaled_pcd

def coarse_registration_with_features(source, target, voxel_size):
    """
    Improved coarse registration with better parameters
    """
    # Downsample
    source_down = source.voxel_down_sample(voxel_size)
    target_down = target.voxel_down_sample(voxel_size)
    
    # Estimate normals with larger radius
    radius_normal = voxel_size * 3  # Increased from 2
    source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=50))
    target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=50))
    
    # Compute FPFH features with larger radius
    radius_feature = voxel_size * 8  # Increased from 5
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=150))
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=150))
    
    # RANSAC with relaxed parameters
    distance_threshold = voxel_size * 2.0  # Increased threshold
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4, [  # Increased from 3
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.8),  # Relaxed from 0.9
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 0.999))  # More iterations
    
    return result
# def coarse_registration_with_features(source, target, voxel_size):
#     """
#     Perform coarse registration using FPFH features
#     """
#     # Downsample
#     source_down = source.voxel_down_sample(voxel_size)
#     target_down = target.voxel_down_sample(voxel_size)
    
#     # Estimate normals
#     radius_normal = voxel_size * 2
#     source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
#     target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    
#     # Compute FPFH features
#     radius_feature = voxel_size * 5
#     source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
#         source_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
#     target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
#         target_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    
#     # RANSAC registration
#     distance_threshold = voxel_size * 1.5
#     result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
#         source_down, target_down, source_fpfh, target_fpfh, True,
#         distance_threshold,
#         o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
#         3, [
#             o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
#             o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
#         ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 0.999))
    
#     return result

def fine_registration_icp(source, target, initial_transform, voxel_size):
    """
    Perform fine registration using ICP
    """
    distance_threshold = voxel_size * 0.4
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    
    return result

def align_point_clouds_with_scale(source_path, target_path, voxel_size=0.05):
    """
    Complete pipeline to align two point clouds with different scales
    
    Args:
        source_path: Path to source point cloud
        target_path: Path to target point cloud  
        voxel_size: Voxel size for downsampling and registration
    
    Returns:
        aligned_source: Transformed source point cloud
        transformation: Final transformation matrix
        scale_factor: Estimated scale factor
    """
    
    # Load point clouds
    print("Loading point clouds...")
    source = o3d.io.read_point_cloud(source_path)
    target = o3d.io.read_point_cloud(target_path)
    
    print(f"Source points: {len(source.points)}")
    print(f"Target points: {len(target.points)}")
    
    # Step 1: Estimate scale factor
    print("\nEstimating scale factor...")
    scale_factor = estimate_scale_factor(source, target, method='bbox')
    print(f"Estimated scale factor: {scale_factor:.4f}")
    
    # Step 2: Apply scaling
    print("Applying scale transformation...")
    scaled_source = apply_scale_transform(source, scale_factor)
    
    # Step 3: Coarse registration
    print("Performing coarse registration...")
    coarse_result = coarse_registration_with_features(scaled_source, target, voxel_size)
    print(f"Coarse registration fitness: {coarse_result.fitness:.4f}")
    print(f"Coarse registration RMSE: {coarse_result.inlier_rmse:.4f}")
    
    # Step 4: Fine registration (ICP)
    print("Performing fine registration...")
    fine_result = fine_registration_icp(scaled_source, target, 
                                       coarse_result.transformation, voxel_size)
    print(f"Fine registration fitness: {fine_result.fitness:.4f}")
    print(f"Fine registration RMSE: {fine_result.inlier_rmse:.4f}")
    
    # Step 5: Apply final transformation
    aligned_source = copy.deepcopy(scaled_source)
    aligned_source.transform(fine_result.transformation)
    
    # Combine scale and rigid transformation
    scale_matrix = np.eye(4)
    scale_matrix[:3, :3] *= scale_factor
    final_transformation = fine_result.transformation @ scale_matrix
    
    return aligned_source, final_transformation, scale_factor
def global_registration_alternative(source, target, voxel_size):
    """
    Alternative using Fast Global Registration (FGR)
    """
    # Downsample and compute features (same as above)
    source_down = source.voxel_down_sample(voxel_size)
    target_down = target.voxel_down_sample(voxel_size)
    
    radius_normal = voxel_size * 3
    source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=50))
    target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=50))
    
    radius_feature = voxel_size * 8
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=150))
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=150))
    
    # Use Fast Global Registration instead
    distance_threshold = voxel_size * 0.5
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    
    return result

def fine_registration_icp_refined(source, target, initial_transform, voxel_size):
    """
    Perform fine registration using ICP with refined parameters
    """
    # The distance threshold for finding correspondences.
    # It should be slightly larger than the expected maximum distance between corresponding points.
    # If it's too small, ICP might not find enough correspondences.
    # If it's too large, it might find incorrect correspondences.
    distance_threshold = voxel_size * 1.5 # Increased from 0.4 to 1.5 times voxel_size
    
    # Number of iterations for ICP. More iterations can lead to better convergence
    # but also takes longer.
    max_iterations = 200 # Increased from default (often 30) to 200
    
    # Point-to-point ICP is usually good for initial fine alignment.
    # For very fine alignment, point-to-plane can be better if normals are accurate.
    estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    
    # Convergence criteria
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=1e-6, # Stop if fitness improvement is small
        relative_rmse=1e-6,    # Stop if RMSE improvement is small
        max_iteration=max_iterations
    )

    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, initial_transform,
        estimation_method,
        criteria
    )
    
    return result
def align_point_clouds_with_scale_fgr(source_path, target_path, voxel_size=0.05):
    """
    Complete pipeline using Fast Global Registration instead of RANSAC
    """
    
    # Load point clouds
    print("Loading point clouds...")
    source = o3d.io.read_point_cloud(str(source_path))
    target = o3d.io.read_point_cloud(str(target_path))
    
    print(f"Source points: {len(source.points)}")
    print(f"Target points: {len(target.points)}")
    
    # Step 1: Estimate scale factor
    print("\nEstimating scale factor...")
    scale_factor = estimate_scale_factor(source, target, method='centroid_distance')
    print(f"Estimated scale factor: {scale_factor:.4f}")
    
    # Step 2: Apply scaling
    print("Applying scale transformation...")
    # scaled_source = apply_scale_transform(source, scale_factor)
    
    # Step 3: Global registration using FGR (instead of RANSAC)
    print("Performing global registration with FGR...")
    global_result = global_registration_alternative(source, target, voxel_size)
    print(f"Global registration fitness: {global_result.fitness:.4f}")
    print(f"Global registration RMSE: {global_result.inlier_rmse:.4f}")
    
    # Step 4: Fine registration (ICP)
    print("Performing fine registration...")
    fine_result = fine_registration_icp(source, target, 
                                       global_result.transformation, voxel_size)
    print(f"Fine registration fitness: {fine_result.fitness:.4f}")
    print(f"Fine registration RMSE: {fine_result.inlier_rmse:.4f}")
    
    # Step 5: Apply final transformation
    aligned_source = copy.deepcopy(source)
    aligned_source.transform(fine_result.transformation)
    
    # Combine scale and rigid transformation
    scale_matrix = np.eye(4)
    scale_matrix[:3, :3] *= scale_factor
    final_transformation = fine_result.transformation @ scale_matrix
    
    return aligned_source, final_transformation, scale_factor
def align_point_clouds_robust(source_path, target_path, voxel_size=0.05):
    """
    Try both RANSAC and FGR, pick the better result
    """
    
    # Load and scale (same as before)
    source = o3d.io.read_point_cloud(str(source_path))
    target = o3d.io.read_point_cloud(str(target_path))
    
    scale_factor = estimate_scale_factor(source, target, method='bbox')
    scaled_source = apply_scale_transform(source, scale_factor)
    
    print("Trying RANSAC registration...")
    try:
        ransac_result = coarse_registration_with_features(scaled_source, target, voxel_size)
        ransac_fitness = ransac_result.fitness
        print(f"RANSAC fitness: {ransac_fitness:.4f}")
    except:
        ransac_fitness = 0
        ransac_result = None
    
    print("Trying FGR registration...")
    try:
        fgr_result = global_registration_alternative(scaled_source, target, voxel_size)
        fgr_fitness = fgr_result.fitness
        print(f"FGR fitness: {fgr_fitness:.4f}")
    except:
        fgr_fitness = 0
        fgr_result = None
    
    # Pick the better result
    if fgr_fitness > ransac_fitness:
        print("Using FGR result")
        coarse_result = fgr_result
    else:
        print("Using RANSAC result")
        coarse_result = ransac_result
    
    # Continue with ICP refinement
    fine_result = fine_registration_icp(scaled_source, target, 
                                       coarse_result.transformation, voxel_size)
    
    aligned_source = copy.deepcopy(scaled_source)
    aligned_source.transform(fine_result.transformation)
    
    return aligned_source, fine_result.transformation, scale_factor



def align_with_centering(source_path, target_path, voxel_size=0.05):
    """
    Center both clouds at origin before alignment
    """
    # Load point clouds
    source = o3d.io.read_point_cloud(str(source_path))
    target = o3d.io.read_point_cloud(str(target_path))
    
    # Center both clouds
    source_center = source.get_center()
    target_center = target.get_center()
    
    source.translate(-source_center)
    target.translate(-target_center)
    
    # Apply scaling
    scale_factor = estimate_scale_factor(source, target, method='centroid_distance')
    scaled_source = apply_scale_transform(source, scale_factor)
    
    # Try alignment
    global_result = global_registration_alternative(scaled_source, target, voxel_size)
    fine_result = fine_registration_icp(scaled_source, target, 
                                       global_result.transformation, voxel_size)
    
    # Apply transformation
    aligned_source = copy.deepcopy(scaled_source)
    aligned_source.transform(fine_result.transformation)
    
    # Translate back to target's original position
    aligned_source.translate(target_center)
    target.translate(target_center)
    
    return aligned_source, fine_result.transformation, scale_factor
def manual_initial_alignment(source, target):
    """
    Manually pick corresponding points for initial alignment
    """
    print("Pick corresponding points:")
    print("1. In source cloud (first window)")
    print("2. In target cloud (second window)")
    print("Pick at least 3-4 point pairs")
    
    # Pick points in source
    print("Pick points in SOURCE cloud...")
    vis_source = o3d.visualization.VisualizerWithEditing()
    vis_source.create_window("Source - Pick Points", width=800, height=600)
    vis_source.add_geometry(source)
    vis_source.run()
    source_points = vis_source.get_picked_points()
    vis_source.destroy_window()
    
    # Pick points in target  
    print("Pick corresponding points in TARGET cloud...")
    vis_target = o3d.visualization.VisualizerWithEditing()
    vis_target.create_window("Target - Pick Points", width=800, height=600)
    vis_target.add_geometry(target)
    vis_target.run()
    target_points = vis_target.get_picked_points()
    vis_target.destroy_window()
    
    if len(source_points) != len(target_points) or len(source_points) < 3:
        print("Need at least 3 corresponding point pairs!")
        return np.eye(4)
    
    # Compute transformation from point correspondences
    source_pts = np.asarray(source.points)[source_points]
    target_pts = np.asarray(target.points)[target_points]
    
    # Use Kabsch algorithm (SVD-based alignment)
    transformation = compute_transformation_from_correspondences(source_pts, target_pts)
    
    return transformation

def compute_transformation_from_correspondences(source_pts, target_pts):
    """
    Compute rigid transformation from point correspondences using SVD
    """
    # Center the points
    source_center = np.mean(source_pts, axis=0)
    target_center = np.mean(target_pts, axis=0)
    
    source_centered = source_pts - source_center
    target_centered = target_pts - target_center
    
    # Compute rotation using SVD
    H = source_centered.T @ target_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Ensure proper rotation (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Compute translation
    t = target_center - R @ source_center
    
    # Build transformation matrix
    transformation = np.eye(4)
    transformation[:3, :3] = R
    transformation[:3, 3] = t
    
    return transformation
def align_with_manual_initial(source_path, target_path, voxel_size=0.05):
    """
    Complete alignment pipeline with manual initial alignment
    """
    
    # Load point clouds
    print("Loading point clouds...")
    source = o3d.io.read_point_cloud(str(source_path))
    target = o3d.io.read_point_cloud(str(target_path))
    
    # Apply scaling first
    print("Applying scaling...")
    scale_factor = estimate_scale_factor(source, target, method='bbox')
    scaled_source = apply_scale_transform(source, scale_factor)
    print(f"Scale factor: {scale_factor:.4f}")
    
    # Manual initial alignment
    print("Starting manual point picking...")
    initial_transform = manual_initial_alignment(scaled_source, target)
    
    # Apply initial transformation
    scaled_source.transform(initial_transform)
    
    # Fine-tune with ICP
    print("Fine-tuning with ICP...")
    fine_result = fine_registration_icp(scaled_source, target, np.eye(4), voxel_size)
    
    # Apply final transformation
    aligned_source = copy.deepcopy(scaled_source)
    aligned_source.transform(fine_result.transformation)
    
    # Combine all transformations
    final_transform = fine_result.transformation @ initial_transform
    
    return aligned_source, final_transform, scale_factor

# Usage
def try_multiple_voxel_sizes(source_path, target_path):
    """
    Try registration with different voxel sizes
    """
    voxel_sizes = [0.1, 0.05, 0.02, 0.01, 0.005,0.001 ]  # From coarse to fine
    voxel_sizes =[0.02]
    best_fitness = 0
    best_result = None
    
    for voxel_size in voxel_sizes:
        print(f"\nTrying voxel size: {voxel_size}")
        try:
            aligned_source, transformation, scale_factor = align_point_clouds_with_scale_fgr(
                source_path, target_path, voxel_size=voxel_size)
            
            # Evaluate fitness
            source = o3d.io.read_point_cloud(str(source_path))
            target = o3d.io.read_point_cloud(str(target_path))
            
            # Quick fitness check
            distances = aligned_source.compute_point_cloud_distance(target)
            fitness = np.sum(np.asarray(distances) < voxel_size) / len(distances)
            
            print(f"Fitness: {fitness:.4f}")
            # target.paint_uniform_color([1.0, 0.0, 0.0])  # Red

            # Paint aligned source cloud blue
            # aligned_source.paint_uniform_color([0.0, 0.0, 1.0])  # Blue
            o3d.visualization.draw_geometries([aligned_source, target]) 

            o3d.io.write_point_cloud("points3D_trans_new.ply", aligned_source)

            if fitness > best_fitness:
                best_fitness = fitness
                best_result = (aligned_source, transformation, scale_factor)
                
        except Exception as e:
            print(f"Failed with voxel size {voxel_size}: {e}")
    
    return best_result

# Example usage function
def demo_alignment():
    """
    Demo function showing how to use the alignment pipeline
    """
    print("Point Cloud Alignment with Scale Demo")
    print("=" * 50)
    
    # Create sample point clouds for demonstration
    print("Creating sample point clouds...")
    
    # Create a bunny-like shape
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    source_pcd = mesh.sample_points_uniformly(number_of_points=1000)
    
    # Create target by scaling, rotating, and translating
    target_pcd = copy.deepcopy(source_pcd)
    
    # Apply scale (make it 2x larger)
    target_pcd.scale(2.0, center=target_pcd.get_center())
    
    # Apply rotation
    R = target_pcd.get_rotation_matrix_from_xyz((np.pi/4, np.pi/6, np.pi/8))
    target_pcd.rotate(R, center=target_pcd.get_center())
    
    # Apply translation
    target_pcd.translate([1.0, 0.5, -0.3])
    
    # Add some noise
    points = np.asarray(target_pcd.points)
    points += np.random.normal(0, 0.01, points.shape)
    target_pcd.points = o3d.utility.Vector3dVector(points)
    
    print(f"Source points: {len(source_pcd.points)}")
    print(f"Target points: {len(target_pcd.points)}")
    
    # Estimate scale
    scale_factor = estimate_scale_factor(source_pcd, target_pcd)
    print(f"Estimated scale factor: {scale_factor:.4f} (true: 2.0)")
    
    return source_pcd, target_pcd

# # Run demo
# source_demo, target_demo = demo_alignment()

# print("\nAlignment pipeline components ready!")
# print("\nTo use with your own point clouds:")
# print("aligned_source, transformation, scale = align_point_clouds_with_scale('source.ply', 'target.ply')")
target_path = '/home/hamit/Softwares/Dynamic3DGaussians/data/2025-08-06_16-38-56_3412x2500_combin3_2_48sc/points3D_simplified_trans.ply'  # Your second point cloud  '
source_path = '/home/hamit/Softwares/Dynamic3DGaussians/data_making/points3D_trans.ply' # Your second point cloud  '
# source_path =  '/home/hamit/Documents/point_cloud_combin3_simplified.ply'
target = o3d.io.read_point_cloud(target_path)  

# aligned_source, transformation, scale_factor = align_point_clouds_with_scale(
#      source_path, # Your first point cloud
#     target_path,
#     voxel_size=0.1       # Adjust based on your data density
# )

aligned_source, transformation, scale_factor = align_point_clouds_with_scale_fgr(
    source_path,
    target_path,
    voxel_size=0.02
)
# aligned_source, transformation, scale_factor = align_with_centering(  
#         source_path,
#     target_path,  
#     voxel_size=0.02  # Start with larger voxel size  
# ) 
# best_results = try_multiple_voxel_sizes(
#        source_path,
#     target_path,
# )
# print(best_results)

# aligned_source, transformation, scale_factor = align_with_manual_initial(
#     source_path,
#     target_path,  
#     voxel_size=0.05
# )

# Method 2: Try both, pick better
# aligned_source, transformation, scale_factor = align_point_clouds_robust(
#     source_path,
#     target_path,
#     voxel_size=0.05
# )
# target.paint_uniform_color([1.0, 0.0, 0.0])  # Red
# aligned_source.paint_uniform_color([0.0, 1.0, 0.0])  # Red

# Paint aligned source cloud blue
# aligned_source.paint_uniform_color([0.0, 0.0, 1.0])  # Blue
o3d.visualization.draw_geometries([aligned_source, target])  

o3d.io.write_point_cloud("points3D_trans_new.ply", aligned_source)
