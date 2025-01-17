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
def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def load_cameras_params(calib_cameras_data_path):
    """Load camera parameters from a JSON file."""
    fs = cv2.FileStorage(str(calib_cameras_data_path), cv2.FILE_STORAGE_READ)
    num_cameras =  int(fs.getNode("nb_camera").real())
    params_dict = {}
    quatvec_and_trans_dict = {}
    for cam_idx in range(num_cameras):
        cam_name = f"camera_{cam_idx}"
        camera_matrix = fs.getNode(cam_name).getNode("camera_matrix").mat()
        # with np.nditer(camera_matrix, op_flags=['readwrite']) as it:
        #     for x in it:
        #         if x != 0 and x !=1 :
        #             x *= 2 
        # camera_matrix_ = [it if (it == 1 or it == 0) else it/2  for it in camera_matrix ]
        camera_matrix = camera_matrix.flatten().tolist()
        params = [ pr for pr in camera_matrix if pr !=0 and pr != 1 ]
        params[1], params[2] = params[2], params[1]

        distortion_coefficients = fs.getNode(cam_name).getNode("distortion_vector").mat()

        params = params + distortion_coefficients.flatten().tolist()[:4]
        params_dict[cam_idx] = params

        camera_pose_matrix = fs.getNode(cam_name).getNode("camera_pose_matrix").mat()
        w2c = np.linalg.inv(camera_pose_matrix)
        quatvec = rotmat2qvec(w2c[:3,:3])
        trans = w2c[:3,3]/100
        print("quatvec and trans:", quatvec.tolist() + trans.tolist())
        quatvec_and_trans_dict[cam_idx] =  quatvec.tolist() + trans.tolist()
        # camera_pose_matrix[:3,3] *= scale/1000
        # camera_pose_matrix = c2w_0 @ camera_pose_matrix
        # w2c[:3,3] *= 15.86/1000
    return params_dict, quatvec_and_trans_dict

# def create_intrinsics(folder_path):
#     params_dict = {}
#     for yml_file in os.listdir(folder_path):
#         root, _ = os.path.splitext(yml_file)
#         with open(os.path.join(folder_path, yml_file), 'r') as file:
#             data = yaml.safe_load(file)
#             # print(data['camera_matrix']['data'])
#             params = [ pr for pr in data['camera_matrix']['data'] if pr !=0 and pr != 1 ]
#             params[1], params[2] = params[2], params[1]
#             params = params + data['distortion_coefficients']['data'][:-4]
#             params_dict[int(root)] = params
            
#     # print(params_dict) 
#     pprint.pprint(params_dict)
#     return params_dict
def print_all_cameras(file_db, extr):
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
        file = open(os.path.join(os.path.dirname(file_db), "cameras.txt"),"w")
        file.write("# Camera list with one line of data per camera\n"
                    "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
                   )
        file.write(f"# Number of cameras: {len(rows)}\n")
       
        for row in rows:
            camera_id, model, width, height, params_blob, prior_focal = row
            # Truncate params if too long for display
            #   params_display = str(params) if len(str(params)) < 10000 else str(params)[:37] + "..."
            #   print(f"{camera_id:^4} | {model:^8} | {width:^6} | {height:^6} | {params_display:<100} | {prior_focal:^8}")
            params = decode_params_blob(params_blob)
            params_str = f"[{', '.join(f'{x:.6f}' for x in params)}]" if params else "Invalid params"
            params_str
            # Truncate params if too long for display
            params_display = params_str if len(params_str) < 100 else params_str[:75] + "..."
            print(f"{camera_id:^4} | {model:^8} | {width:^6} | {height:^6} | {params_display:<100} | {prior_focal:^8}")
            params = ' '.join([str(pr) for pr in params])
            file.write(f'{camera_id} OPENCV {width} {height} {params} \n' )
        file.close()
        print("-" * 110)
        print(f"Total cameras: {len(rows)}")


        cursor.execute("SELECT * FROM images ORDER BY image_id")
        rows = cursor.fetchall()
        file2 = open(os.path.join(os.path.dirname(file_db), "images.txt"),"w")
        file2.write("# image list with one line of data per image\n"
                    "# images id, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
                   )
        file2.write(f"# Number of images: {len(rows)}\n")
        if not rows:

            print("No images found in the database")
            return
        for row in rows:
            image_id, name, CAMERA_ID, _, _, _, _, _, _, _ = row
            # print(extr[int(CAMERA_ID)])
            name_id = os.path.splitext(name)[0]
            QW, QX, QY, QZ, TX, TY, TZ = tuple(extr[int(name_id)])
            file2.write(f'{image_id} {QW} {QX} {QY} {QZ} {TX} {TY} {TZ} {CAMERA_ID} {name}\n\n' )

        
        file2.close()
          
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
def update_camera_params(database_path, params_and_extr):
    # check_table_structure()
    params, extr = params_and_extr
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
        print_all_cameras(database_path, extr)
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
parser.add_argument("--intrinsics_yml_file_path", "-i", required=True, type=str)
parser.add_argument("--database_path", required=True, type=str)

parser.add_argument("--camera", default="OPENCV", type=str)
parser.add_argument("--colmap_executable", default="", type=str)
args = parser.parse_args()
colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"


    
params_and_ext_dict = load_cameras_params(args.intrinsics_yml_file_path)
print(params_and_ext_dict)
update_camera_params(args.database_path, params_and_ext_dict)
## Feature matching

   