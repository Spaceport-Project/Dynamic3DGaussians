# 1. Feature extraction
colmap feature_extractor \
    --database_path database.db \
    --image_path images \
    --ImageReader.camera_model OPENCV  # or PINHOLE

# 2. Feature matching
colmap exhaustive_matcher \
    --database_path database.db

# 3. Create sparse/0 with your text files
mkdir -p sparse/0
# Put cameras.txt, images.txt, points3D.txt (can be empty) in sparse/0/

# 4. Run triangulation directly (reads text format)
colmap point_triangulator \
    --database_path database.db \
    --image_path images \
    --input_path sparse/0 \
    --output_path sparse/0

# 5. Generate dense point cloud
colmap image_undistorter \
    --image_path images \
    --input_path sparse/0 \
    --output_path dense

colmap patch_match_stereo \
    --workspace_path dense

colmap stereo_fusion \
    --workspace_path dense \
    --output_path dense/fused.ply