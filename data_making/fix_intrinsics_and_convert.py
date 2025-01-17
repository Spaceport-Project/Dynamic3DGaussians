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
import pprint
import shutil
import sqlite3
import cv2
import numpy as np
import yaml

DIM = (4096, 3000)

def resize_to_org_size(folder_path):

    for file in os.listdir(folder_path):
        if file.endswith(".png") or  file.endswith(".jpg"):
            image = cv2.imread(os.path.join(folder_path, file))
            resized = cv2.resize(image, DIM, interpolation = cv2.INTER_AREA)
            cv2.imwrite(os.path.join(folder_path, file), resized)

def create_intrinsics(folder_path):
    params_dict = {}
    for yml_file in os.listdir(folder_path):
        root, _ = os.path.splitext(yml_file)
        with open(os.path.join(folder_path, yml_file), 'r') as file:
            data = yaml.safe_load(file)
            # print(data['camera_matrix']['data'])
            params = [ pr for pr in data['camera_matrix']['data'] if pr !=0 and pr != 1 ]
            params[1], params[2] = params[2], params[1]
            params = params + data['distortion_coefficients']['data'][:-4]
            params_dict[int(root)] = params
            
    # print(params_dict) 
    pprint.pprint(params_dict)
    return params_dict
def print_all_cameras(file_db):
    try:
        conn = sqlite3.connect(file_db)
        cursor = conn.cursor()
        
        # Get all rows from cameras table
        cursor.execute("SELECT * FROM cameras ORDER BY camera_id")
        rows = cursor.fetchall()
        
        if not rows:
            print("No cameras found in the database")
            return
        
        print("\nAll Cameras in Database:")
        print("-" * 110)
        print(f"{'ID':^4} | {'Model':^8} | {'Width':^6} | {'Height':^6} | {'Parameters':^100} | {'Prior Focal':^8}")
        print("-" * 110)
        
        for row in rows:
            camera_id, model, width, height, params_blob, prior_focal = row
            # Truncate params if too long for display
            #   params_display = str(params) if len(str(params)) < 10000 else str(params)[:37] + "..."
            #   print(f"{camera_id:^4} | {model:^8} | {width:^6} | {height:^6} | {params_display:<100} | {prior_focal:^8}")
            params = decode_params_blob(params_blob)
            params_str = f"[{', '.join(f'{x:.6f}' for x in params)}]" if params else "Invalid params"
            
            # Truncate params if too long for display
            params_display = params_str if len(params_str) < 100 else params_str[:75] + "..."
            print(f"{camera_id:^4} | {model:^8} | {width:^6} | {height:^6} | {params_display:<100} | {prior_focal:^8}")
        
        
        print("-" * 110)
        print(f"Total cameras: {len(rows)}")
          
    except sqlite3.Error as error:
        print(f"Error accessing database: {error}")
    finally:
        if conn:
            conn.close()    
def create_images_cam_id_dict(file_db):
    images_cam_id_dict = {}
    try:
        conn = sqlite3.connect(file_db)
        cursor = conn.cursor()
        
        # Get all rows from cameras table
        cursor.execute("SELECT * FROM images ORDER BY image_id")
        rows = cursor.fetchall()
        
        if not rows:
            print("No images found in the database")
            return
        
        print("\nAll images in Database:")
        print("-" * 50)
        print(f"  {'image_id':^8} | {'name':^6} | {'camera_id':^6} |")
        print("-" * 50)
        
        for row in rows:
            image_id, name, camera_id  = row[:3]
            name_id, _= os.path.splitext(name)
            images_cam_id_dict[int(name_id)] = camera_id
            
            # params_display = params_str if len(params_str) < 100 else params_str[:75] + "..."
            print(f"{image_id:^8} | {name:^6} | {camera_id:^6} |")
        
        
        print("-" * 550)
        print(f"Total images: {len(rows)}")
        
          
    except sqlite3.Error as error:
        print(f"Error accessing database: {error}")
    finally:
        
        if conn:
            conn.close()  
        return images_cam_id_dict
    
def decode_params_blob(blob):
  """Convert binary blob to list of floats"""
  try:
      # Convert blob to numpy array of float64
      return np.frombuffer(blob, dtype=np.float64).tolist()
  except:
      return None
  
def encode_params_to_blob(params):
  """Convert list of floats to binary blob"""
  try:
    #   if isinstance(params, str):
    #       # Convert string of comma-separated values to list of floats
    #       params = [float(x.strip()) for x in params.split(',')]
      return np.array(params, dtype=np.float64).tobytes()
  except:
      return None
def update_camera_params(database_path, params):
    # check_table_structure()
    img_cam_id = create_images_cam_id_dict(database_path)
    try:
        db = sqlite3.connect(database_path)
        update_query = '''
                UPDATE cameras 
                SET model = ?,
                    width = ?,
                    height = ?,
                    params = ?,
                    prior_focal_length = ?
                WHERE camera_id = ?
                '''
        insert_query = '''
        INSERT INTO cameras (camera_id, model, width, height, params, prior_focal_length)
        VALUES (?, ?, ?, ?, ?, ?)
        '''
        model, width, height = 4, DIM[0], DIM[1]
        cursor = db.cursor()
        for key, val in params.items():
           
            # params = ' '.join(params)

            camera_id = img_cam_id[key]
            params_blob = encode_params_to_blob(val)
            # model_id = get_model_param_count(model)
            # if model_id is None:
            #   raise ValueError(f"Unknown camera model: {model_id}")
            
            
       
      
            cursor.execute(update_query, (model, width, height, params_blob, 1, camera_id))
        
            if cursor.rowcount > 0:
                print(f"Successfully updated camera with ID: {camera_id}")
                # Show the updated row
                cursor.execute("SELECT * FROM cameras WHERE camera_id = ?", (camera_id,))
                print("Updated row:", cursor.fetchone())
            else:
                print(f"No camera found with ID: {camera_id}")
        db.commit()
        print_all_cameras(database_path)
        # print_all_images(database_path)

        

                
    except sqlite3.Error as error:
        print(f"Error updating camera info: {error}")
    except ValueError as error:
        print(f"Error with params format: {error}")
    finally:
        # Close the connection
        if db:
            db.close()

# This Python script is based on the shell converter script provided in the MipNerF 360 repository.
parser = ArgumentParser("Colmap converter")
parser.add_argument("--no_gpu", action='store_true')
parser.add_argument("--skip_matching", action='store_true')
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--intrinsics_folder_path", "-i", required=True, type=str)
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

    # Feature extraction
    feat_extracton_cmd = colmap_command + " feature_extractor \
        --database_path " + args.source_path + "/distorted/database.db \
        --image_path " + args.source_path + "/input  \
        --ImageReader.single_camera 0 \
        --ImageReader.camera_model " + args.camera + " \
        --SiftExtraction.estimate_affine_shape true \
        --SiftExtraction.domain_size_pooling true \
        --SiftExtraction.num_threads 32 \
        --SiftExtraction.max_image_size 3200 \
        --SiftExtraction.max_num_features 15000 \
        --SiftExtraction.use_gpu " + str(use_gpu)
    exit_code = os.system(feat_extracton_cmd)
    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)
    
    params_dict = create_intrinsics(args.intrinsics_folder_path)
    update_camera_params(args.source_path + "/distorted/database.db", params_dict)
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
        --Mapper.ba_global_function_tolerance=0.000001 \
        --Mapper.ba_refine_focal_length 0 \
        --Mapper.ba_refine_principal_point 0  \
        --Mapper.ba_refine_extra_params 0"
        )
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
    --output_path " + args.source_path + " \
    --output_type COLMAP \
    --max_image_size 4096 3100 \
    --roi_max_x 1 \
    --roi_max_y 1 \
    ")
exit_code = os.system(img_undist_cmd)
if exit_code != 0:
    logging.error(f"Mapper failed with code {exit_code}. Exiting.")
    exit(exit_code)

# resize_to_org_size(args.source_path + "/images/")

files = os.listdir(args.source_path + "/sparse")
os.makedirs(args.source_path + "/sparse/0", exist_ok=True)
# Copy each file from the source directory to the destination directory
for file in files:
    if file == '0':
        continue
    source_file = os.path.join(args.source_path, "sparse", file)
    destination_file = os.path.join(args.source_path, "sparse", "0", file)
    shutil.copy2(source_file, destination_file)

# Dense reconstruction
path_match_cmd = (colmap_command + " patch_match_stereo   \
                  --workspace_format COLMAP \
            --workspace_path "  + os.path.join(args.source_path) )
exit_code = os.system(path_match_cmd)
print("path matching done!")
stereo_fusion_command = (colmap_command + " stereo_fusion \
                         --workspace_format COLMAP \
                          --input_type geometric \
            --workspace_path " + os.path.join(args.source_path) + " \
            --output_path " + os.path.join(args.source_path, 'points3d.ply') )
exit_code = os.system(stereo_fusion_command)

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



