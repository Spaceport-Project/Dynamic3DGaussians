#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import logging
from argparse import ArgumentParser
import shutil
import cv2
import numpy as np

DIM = (2048,1500)
# This Python script is based on the shell converter script provided in the MipNerF 360 repository.
parser = ArgumentParser("Colmap converter")
parser.add_argument("--no_gpu", action='store_true')
parser.add_argument("--skip_matching", action='store_true')
parser.add_argument("--mask_path", "-m", default="", type=str)
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--camera", default="OPENCV", type=str)
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--resize", action="store_true")
parser.add_argument("--magick_executable", default="", type=str)
args = parser.parse_args()
colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "magick"
use_gpu = 1 if not args.no_gpu else 0
input_folder = "/input_enhanced "
def reduce_purple_tint(image, blue_factor=0.7, red_factor=0.7, green_boost=1.1):
    result = image.copy()
    # Adjust channels individually
    result[:,:,0] = np.clip(result[:,:,0] * red_factor, 0, 255).astype(np.uint8)  # Red channel
    result[:,:,1] = np.clip(result[:,:,1] * green_boost, 0, 255).astype(np.uint8)  # Green channel
    result[:,:,2] = np.clip(result[:,:,2] * blue_factor, 0, 255).astype(np.uint8)  # Blue channel
    return result


def reduce_purple_tint_for_folder(folder_name):
    for file in os.listdir(os.path.join(folder_name, "input")):
        img = cv2.imread(os.path.join(folder_name, "input", file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = reduce_purple_tint(img)
        cv2.imwrite(os.path.join(folder_name, "input", file), cv2.cvtColor(res, cv2.COLOR_RGB2BGR))


if not args.skip_matching:

    # reduce_purple_tint_for_folder(args.source_path)
    os.makedirs(args.source_path + "/distorted/sparse", exist_ok=True)
   
    #   --SiftExtraction.max_image_size 100000  \
    #     --SiftExtraction.max_num_features 200000  \
   #--SiftExtraction.domain_size_pooling true \
    #  --SiftExtraction.peak_threshold 0.002 \
    #     --SiftExtraction.edge_threshold 15 \
    # --SiftExtraction.estimate_affine_shape true \
#  --ImageReader.mask_path " + args.source_path +  "/masks \
    # Feature extraction
    feat_extracton_cmd = colmap_command + " feature_extractor \
        --database_path " + args.source_path + "/distorted/database.db \
        --image_path " + args.source_path + input_folder  + " \
        --ImageReader.mask_path " + args.source_path +  "/masks \
        --ImageReader.single_camera 1 \
        --ImageReader.camera_model " + args.camera + " \
        --SiftExtraction.num_threads 32 \
        --SiftExtraction.peak_threshold 0.002 \
        --SiftExtraction.edge_threshold 15 \
        --SiftExtraction.domain_size_pooling true \
        --SiftExtraction.max_image_size 100000  \
        --SiftExtraction.max_num_features 200000  \
        --SiftExtraction.use_gpu " + str(use_gpu)  
    
       
        
    exit_code = os.system(feat_extracton_cmd)
    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)

    
    # --SiftMatching.guided_matching true \
    # --TwoViewGeometry.min_num_inliers 50 \
    # --TwoViewGeometry.min_inlier_ratio 0.5  \
    # --TwoViewGeometry.confidence 0.99999 \
    #--SiftMatching.max_num_matches 200000 \
        # --SiftMatching.max_distance 0.3 \
        # --TwoViewGeometry.max_error 1.5 \

    ## Feature matching
    feat_matching_cmd = colmap_command + " exhaustive_matcher \
        --database_path " + args.source_path + "/distorted/database.db \
        --SiftMatching.cross_check 1  \
        --SiftMatching.max_num_matches 200000 \
        --SiftMatching.max_distance 0.3 \
        --TwoViewGeometry.max_error 1.5 \
        --SiftMatching.guided_matching true \
        --SiftMatching.use_gpu " + str(use_gpu)
    exit_code = os.system(feat_matching_cmd)
    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ## Bundle adjustment
    # The default Mapper tolerance is unnecessarily large,
    # decreasing it speeds up bundle adjustment steps.
    mapper_cmd = (colmap_command + " mapper \
        --database_path " + args.source_path + "/distorted/database.db \
        --image_path "  + args.source_path + input_folder + " \
        --output_path "  + args.source_path + "/distorted/sparse \
        --Mapper.ba_refine_principal_point 1 \
        --Mapper.ba_refine_extra_params 1 \
        --Mapper.filter_max_reproj_error 2.0 \
        --Mapper.init_max_error 2.0 \
        --Mapper.ba_global_function_tolerance=0.000001")
    exit_code = os.system(mapper_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)
    os.makedirs(os.path.join(args.source_path, "distorted_sparse_aligned"), exist_ok=True)
    aligner_cmd = ( colmap_command + " model_orientation_aligner \
                    --method  MANHATTAN-WORLD \
                    --image_path " + args.source_path + input_folder + " \
                    --input_path " + args.source_path + "/distorted/sparse/0 \
                    --output_path " + args.source_path + "/distorted_sparse_aligned")
    exit_code = os.system(aligner_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)




#Image undistortion
img_undist_cmd = (colmap_command + " image_undistorter \
    --image_path " + args.source_path + input_folder + " \
    --input_path " + args.source_path + "/distorted_sparse_aligned \
    --output_path " + args.source_path + "\
    --output_type COLMAP")
exit_code = os.system(img_undist_cmd)
if exit_code != 0:
    logging.error(f"Mapper failed with code {exit_code}. Exiting.")
    exit(exit_code)

files = os.listdir(args.source_path + "/sparse")
os.makedirs(args.source_path + "/sparse/0", exist_ok=True)
# Copy each file from the source directory to the destination directory
for file in files:
    if file == '0':
        continue
    source_file = os.path.join(args.source_path, "sparse", file)
    destination_file = os.path.join(args.source_path, "sparse", "0", file)
    shutil.copy2(source_file, destination_file)


exit()

path_match_cmd = (colmap_command + " patch_match_stereo   \
                  --workspace_format COLMAP \
                  --PatchMatchStereo.gpu_index=0 \
            --workspace_path "  + os.path.join(args.source_path) )
exit_code = os.system(path_match_cmd)
print("path matching done!")

# --StereoFusion.mask_path " + args.source_path +  "/masks \

stereo_fusion_command = (colmap_command + " stereo_fusion \
                         --workspace_format COLMAP \
                          --input_type geometric \
            --workspace_path " + os.path.join(args.source_path) + " \
            --StereoFusion.mask_path " + args.source_path +  "/masks \
            --output_path " + os.path.join(args.source_path, 'points3d.ply') )
exit_code = os.system(stereo_fusion_command)
print("stereo fusion  done!")

# exit_code = os.system("nohup bash  -c '(" + path_match_cmd + "; " + stereo_fusion_command + ")' > output.log 2>&1 &")
# print("path matching and stereo fusion done!")

if(args.resize):
    print("Copying and resizing...")

    # Resize images.
    os.makedirs(args.source_path + "/images_2", exist_ok=True)
    os.makedirs(args.source_path + "/images_4", exist_ok=True)
    os.makedirs(args.source_path + "/images_8", exist_ok=True)
    # Get the list of files in the source directory
    files = os.listdir(args.source_path + "/images")
    # Copy each file from the source directory to the destination directory
    for file in files:
        source_file = os.path.join(args.source_path, "images", file)

        destination_file = os.path.join(args.source_path, "images_2", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 50% " + destination_file)
        if exit_code != 0:
            logging.error(f"50% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

        destination_file = os.path.join(args.source_path, "images_4", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 25% " + destination_file)
        if exit_code != 0:
            logging.error(f"25% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

        destination_file = os.path.join(args.source_path, "images_8", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 12.5% " + destination_file)
        if exit_code != 0:
            logging.error(f"12.5% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

print("Done.")
