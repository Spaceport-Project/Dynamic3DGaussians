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
import sqlite3

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

  
if not args.skip_matching:
    os.makedirs(args.source_path + "/distorted/sparse", exist_ok=True)
    #  --SiftExtraction.max_image_size 3500 \
    #     --SiftExtraction.max_num_features 25000 \
    # --SiftExtraction.max_image_size 3200  \
    #     --SiftExtraction.max_num_features 12000 "
    # Feature extraction
    feat_extracton_cmd = colmap_command + " feature_extractor \
        --database_path " + args.source_path + "/distorted/database.db \
        --image_path " + args.source_path + "/input  \
        --ImageReader.single_camera 1 \
        --ImageReader.camera_model " + args.camera + " \
        --ImageReader.mask_path " + args.mask_path + " \
        --SiftExtraction.estimate_affine_shape true \
        --SiftExtraction.domain_size_pooling true \
        --SiftExtraction.num_threads 16 \
        --SiftExtraction.use_gpu " + str(use_gpu) + " \
        --SiftExtraction.max_image_size 30000  \
        --SiftExtraction.max_num_features 120000 "
        
    exit_code = os.system(feat_extracton_cmd)
    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)

    

    ## Feature matching
    feat_matching_cmd = colmap_command + " exhaustive_matcher \
        --database_path " + args.source_path + "/distorted/database.db \
        --SiftMatching.max_num_matches 100000 \
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
        --image_path "  + args.source_path + "/input \
        --output_path "  + args.source_path + "/distorted/sparse \
        --Mapper.ba_global_function_tolerance=0.000001")
    exit_code = os.system(mapper_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)
    os.makedirs(os.path.join(args.source_path, "distorted_sparse_aligned"), exist_ok=True)
    aligner_cmd = ( colmap_command + " model_orientation_aligner \
                   --image_path " + args.source_path + "/input \
                    --input_path " + args.source_path + "/distorted/sparse/0 \
                    --output_path " + args.source_path + "/distorted_sparse_aligned")
    exit_code = os.system(aligner_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

# Image undistortion
img_undist_cmd = (colmap_command + " image_undistorter \
    --image_path " + args.source_path + "/input \
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
path_match_cmd = (colmap_command + " patch_match_stereo   \
                  --workspace_format COLMAP \
            --workspace_path "  + os.path.join(args.source_path) )
# exit_code = os.system(path_match_cmd)
# print("path matching done!")
stereo_fusion_command = (colmap_command + " stereo_fusion \
                         --workspace_format COLMAP \
                          --input_type geometric \
            --workspace_path " + os.path.join(args.source_path) + " \
            --output_path " + os.path.join(args.source_path, 'points3d.ply') )
exit_code = os.system("nohup bash  -c '(" + path_match_cmd + "; " + stereo_fusion_command + ")' > output.log 2>&1 &")
print("path matching and stereo fusion done!")

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
