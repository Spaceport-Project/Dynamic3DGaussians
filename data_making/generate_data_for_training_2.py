from copy import deepcopy
import errno
import glob
import shutil
import subprocess
import sys
import threading
import time
from typing import List
import cv2
import os
import json
import ffmpeg
import numpy as np
import argparse
import multiprocessing
from pathlib import Path
from argparse import Namespace  

import generate_masked_data
import xmltodict


# def extract_frames_with_pts(cam_file_path, output_path, cam_serial):
#     # First, get frame information
#     cmd = ['ffprobe',
#            '-v', 'quiet',
#            '-select_streams', 'v:0',
#            '-show_frames',
#            '-of', 'json',
#            cam_file_path]

#     result = subprocess.run(cmd, stdout=subprocess.PIPE)
    
#     frames_data = json.loads(result.stdout)

#     # Now extract frames and rename them with their PTS
#     # output_file_path = f'{output_path}/Cam_{cam_index:03}'
    
#     os.makedirs(output_path, exist_ok=True)
#     subprocess.run(['ffmpeg', '-i', cam_file_path, '-vf', "select='between(n,0,350)'", 
#                    '-vsync', '0', '-start_number', '0',
#                    f'{output_path}/frame_%3d_{cam_serial}.png'])

#     # Rename files with PTS
#     # print(frames_data['frames'])
#     k = 0
#     for i, frame in enumerate(frames_data['frames'], 0):
#         # pts_time = frame.get('pts', str(i))
#         pts_time = frame["pkt_pts"]
#         # if pts_time >= 1637998 and pts_time <= 1703331:

#         # if pts_time >= 1749998 and pts_time <= 1823664:
#             # print(frame)
#         old_name = f'{output_path}/frame_{i:03}_{cam_serial}.png'
#         new_name = f'{output_path}/frame_{pts_time}_{cam_serial}.png'
#         print(old_name, new_name)
#         subprocess.run(['mv', old_name, new_name])

def calculate_number_valid_timesteps(dataset_path: str, cam_number: int, number_timesteps:int = 300) -> List[str]:
    # Initialize empty list for valid timestamps
    list_valid_ts = []

    # Get all PNG files and extract timestamps
    png_files = glob.glob(os.path.join(dataset_path, "*.png"))
    png_files.sort(key=lambda x: int(x.split('_')[-2]))
    png_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[-2]))

    # Extract timestamps from filenames
    def get_timestamp(filename):
        return int(filename.split('_')[-2]) # Assuming timestamp is second-to-last part

    # Get all unique timestamps
    all_timestamps = sorted(set(get_timestamp(f) for f in png_files))

    # Find first timestamp that has correct number of cameras
    first_timestamp = None
    for timestamp in all_timestamps:
        matching_files = glob.glob(os.path.join(dataset_path, f"*{timestamp}*.png"))
        len_matching_files = len(matching_files)
        if len_matching_files == cam_number:
            first_timestamp = timestamp
            list_serial_numbers = [int(f.split('_')[-1].split('.')[-2])  for f in matching_files ]
            list_serial_numbers.sort()
            break

    if not first_timestamp:
        return []

    # Process timestamps
    id = 0
    for timestamp in all_timestamps:
        # Skip timestamps before first_timestamp
        if int(timestamp) < int(first_timestamp):
            continue

        # Count number of images for current timestamp
        current_cam_number = len(glob.glob(os.path.join(dataset_path, f"*{timestamp}*.png")))
        cam_number_minus_one = cam_number - 1

        # Check if we have enough images for this timestamp
        if current_cam_number < cam_number_minus_one:
            print(f"There must be at least {cam_number_minus_one} images out of {cam_number} "
                  f"for a timestamp, but there exist {current_cam_number} images for the "
                  f"following time stamp '{timestamp}'.\n"
                  f"Exiting from the function of calculating number of valid timesteps!",
                  file=sys.stderr)
            break

        # Break if we've processed 300 timestamps
        if id >= number_timesteps:
            break

        list_valid_ts.append(timestamp)
        id += 1

    return list_valid_ts, list_serial_numbers

def organize_images_folder_by_cam(dataset_path: str, output_folder_path: str, camera_file_path, cam_number: int, number_timesteps:int):
    # Find first timestamp with correct number of cameras
    list_valid_ts=calculate_number_valid_timesteps(dataset_path, cam_number, number_timesteps)
    first_timestamp=list_valid_ts[0]
    png_files = glob.glob(os.path.join(dataset_path, "*.png"))
    png_files.sort()

    # camera_matrix_list, distortion_coefficients_list = load_cameras_params(camera_file_path)


    def get_serial_number(filename):
        return filename.split('_')[-1].split('.')[0]

    # Get list of serial numbers from first timestamp
    first_ts_files = glob.glob(os.path.join(dataset_path, f"*_{first_timestamp}_*.png"))
    list_serial_numbers = sorted([get_serial_number(f) for f in first_ts_files])

    exit_flag = False
    id = 0

    for idx in range(len(list_valid_ts)):
        # Get next 6 timestamps for potential replacements
        iters = list_valid_ts[idx+1:idx+7]
        files_to_undistort = {}
        for a, sn in enumerate(list_serial_numbers):
            # Create output directory if it doesn't exist
            cam_output_dir = os.path.join(output_folder_path, 'ims', str(a))
            os.makedirs(cam_output_dir, exist_ok=True)

            # Try to copy current timestamp image
            current_file = os.path.join(dataset_path, f"frame_{list_valid_ts[idx]}_{sn}.png")
            formatted_number = f"{id:06d}"
            output_file = os.path.join(cam_output_dir, f"{formatted_number}.png")

            if os.path.exists(current_file):
                os.symlink(current_file, output_file)
                # shutil.copy2(current_file, output_file)
            else:
                # Try next 6 timestamps
                found_replacement = False
                for next_ts in iters:
                    next_file = os.path.join(dataset_path, f"frame_{next_ts}_{sn}.png")
                    if os.path.exists(next_file):
                        print(f"Copying {next_file} to {output_file}", file=sys.stderr)
                        shutil.copy2(next_file, output_file)
                        found_replacement = True
                        break

                if not found_replacement:
                    print("Cannot find a image within next 6 time stamps. "
                          "It is because either you have reached the end of the list "
                          "or something wrong with the recordings. "
                          "Make sure your recording does not have that many missing images. "
                          "Exiting!", file=sys.stderr)
                    exit_flag = True
                    break

            if exit_flag:
                break

            files_to_undistort.add(output_file)    
        if exit_flag:
            break
        
        print(f"Copying images from {a+1} cameras for {list_valid_ts[idx]} timestamp has finished!", 
              file=sys.stderr)
        id += 1

def all_items_same(lst):
  if not lst:
      return True  # An empty list is considered to have all items the same

  first_item = lst[0]
  return all(item == first_item for item in lst)

def create_list_cameras_params(ims_folder, k, w2c):
    """Get image path as <cam_id>/<img_file> and cam ids as list"""
    imgs_len = len(k)
    images = []
    cam_ids = []
    k_all = []
    w2c_all = []
    
    # cam_ims_folders_len = [ len([ file for file in os.listdir(os.path.join(ims_folder,folder)) if file.endswith('.png')]) for folder in os.listdir(ims_folder) if os.path.isdir(os.path.join(ims_folder, folder)) ]
    cam_ims_folders = [(folder, len([ file for file in os.listdir(os.path.join(ims_folder,folder)) if file.endswith('.png')])) for folder in os.listdir(ims_folder) if os.path.isdir(os.path.join(ims_folder, folder)) ]
    cam_ims_folders = sorted(cam_ims_folders, key=lambda x: x[1], reverse=True)

    last_file = "000180.png" #sorted(os.listdir(os.path.join(ims_folder, cam_ims_folders[0][0])))[-1]
    number_timestamp = int(os.path.splitext(last_file)[0])

    reorganize_folder_for_missing(ims_folder, number_timestamp + 1)

    seg_folder = os.path.join(os.path.dirname(ims_folder),"seg")

    cam_ims_folders = [(folder, len([ file for file in os.listdir(os.path.join(ims_folder,folder)) if file.endswith('.png')])) for folder in os.listdir(ims_folder) if os.path.isdir(os.path.join(ims_folder, folder)) ]
    cam_seg_folders = [(folder, len([ file for file in os.listdir(os.path.join(seg_folder,folder)) if file.endswith('.png')])) for folder in os.listdir(seg_folder) if os.path.isdir(os.path.join(seg_folder, folder)) ]

    lens_cam_ims_folders = [it[1] for it in cam_ims_folders]
    lens_cam_seg_folders  = [it[1] for it in cam_seg_folders]
    assert all_items_same(lens_cam_ims_folders), "Some folders in image folder have different number of image files"
    assert all_items_same(lens_cam_seg_folders), "Some folders in segmentation folder  have different number of image files"

    indices_to_remove = []
    k = [x for i, x in enumerate(k) if i not in indices_to_remove]
    w2c = [x for i, x in enumerate(w2c) if i not in indices_to_remove]


    for file in sorted(os.listdir(os.path.join(ims_folder,cam_ims_folders[0][0]))):
        image_list = []
        cam_list = []
        for idx in range(imgs_len):
            
            if int(idx) in indices_to_remove:
                continue
            image_list.append(f"{idx}/{file}")
            cam_list.append(int(idx))
        images.append(image_list)
        cam_ids.append(cam_list)
        k_all.append(k)
        w2c_all.append(w2c)

   
    return images, cam_ids, k_all, w2c_all


def load_cameras_params(calib_cameras_data_path):
    """Load camera parameters from a JSON file."""
    fs = cv2.FileStorage(str(calib_cameras_data_path), cv2.FILE_STORAGE_READ)
    num_cameras =  int(fs.getNode("nb_camera").real())
    camera_matrix_list = []
    distortion_coefficients_list = []

    w2c_list = []
    # w2c_0 = np.array([   [ 0.25681035,  0.0670684,  -0.96413188, 0.65729515 ],
    #     [-0.38885296,  0.92045067, -0.03954678, -0.9941085],
    #     [ 0.88478349,  0.38506155,  0.262461, 4.50231357 ],
    #     [0,0,0,1]
    # ] )

    # w2c_0 = np.array([[ 0.93674253, -0.03888465,  0.34785257, -0.20542087],
    #    [ 0.16813808,  0.92162764, -0.34976003, -0.70957294],
    #    [-0.30699025,  0.38612236,  0.8698658 ,  3.68601693],
    #    [ 0.        ,  0.        ,  0.        ,  1.        ]]) # colmap
    
   
 
    c2w_0 = np.array([-0.61258866740844176, -0.26255009921296429, 0.74552167638909927, -1.9142765403606425,
                     -0.26478619283153432, -0.82054891064668067, -0.50654492132714823, 1.3277324287616383,
                       0.7447304187748286, -0.50770752470276426, 0.43313932251835613, -3.5313985762332871, 
                       0, 0, 0, 1]).reshape(4,4) # agisoft

    # rotation = np.array([
    #                      -0.76286913809379497 ,
    #                      0.25276760300339507 ,
    #                      -0.59509597294549799 ,
    #                      -0.0050123218674521528 ,
    #                      0.91807366734314977 ,
    #                      0.39637812497739672 ,
    #                      0.64653349083672984 ,
    #                      0.30536745111916874 ,
    #                     -0.69910311472876063
    #                 ]).reshape(3,3).astype(float)
    # center = np.array([
    #                     0.58958615371745526,
    #                     -0.93097813306740995,  # meshroom
    #                     0.65726862275186793
    #         ])
    # w2c_0 = np.vstack((np.concatenate((rotation, center[:,None]), axis=1),[0,0,0,1]))
    # c2w_0 = np.linalg.inv(w2c_0)
    for cam_idx in range(num_cameras):
        cam_name = f"camera_{cam_idx}"
        camera_matrix = fs.getNode(cam_name).getNode("camera_matrix").mat()
        # with np.nditer(camera_matrix, op_flags=['readwrite']) as it:
        #     for x in it:
        #         if x != 0 and x !=1 :
        #             x *= 2 
        # camera_matrix_ = [it if (it == 1 or it == 0) else it/2  for it in camera_matrix ]
        camera_matrix_list.append(camera_matrix)
        distortion_coefficients = fs.getNode(cam_name).getNode("distortion_vector").mat()
        distortion_coefficients_list.append(distortion_coefficients)
        
        camera_pose_matrix = fs.getNode(cam_name).getNode("camera_pose_matrix").mat()
        # camera_pose_matrix[:3,3] *= 17.429/1000
        # scale=3.3299749231959184/0.2057002568030526 #colmap
        scale = 1.965672400401114/0.2057002568030526 # agisoft
        # scale=0.8267701819523383/0.2057002568030526 # meshroom
        # scale = 4
        # scale=16.1888482090152167

        camera_pose_matrix[:3,3] *= scale/1000
        camera_pose_matrix = c2w_0 @ camera_pose_matrix
        w2c = np.linalg.inv(camera_pose_matrix)
        # w2c[:3,3] *= 15.86/1000
        w2c_list.append(w2c.tolist())
    return camera_matrix_list, distortion_coefficients_list, w2c_list


# def load_cameras_params(calib_cameras_data_path):
#     """Load camera parameters from a JSON file."""
#     fs = cv2.FileStorage(str(calib_cameras_data_path), cv2.FILE_STORAGE_READ)
#     num_cameras =  int(fs.getNode("nb_camera").real())
#     camera_matrix_list = []
#     distortion_coefficients_list = []
#     for cam_idx in range(num_cameras):
#         cam_name = f"camera_{cam_idx}"
#         camera_matrix = fs.getNode(cam_name).getNode("camera_matrix").mat()
#         # with np.nditer(camera_matrix, op_flags=['readwrite']) as it:
#         #     for x in it:
#         #         if x != 0 and x !=1 :
#         #             x *= 2 
#         # camera_matrix_ = [it if (it == 1 or it == 0) else it/2  for it in camera_matrix ]
#         camera_matrix_list.append(camera_matrix)
#         distortion_coefficients = fs.getNode(cam_name).getNode("distortion_vector").mat()
#         distortion_coefficients_list.append(distortion_coefficients)

   
   
#     return camera_matrix_list, distortion_coefficients_list

# def undistort_images(images_path, time_step, output_directory, camera_matrix_list, distortion_coefficients_list):
#     """Undistort images in the specified directory."""
#     Path(output_directory).mkdir(parents=True, exist_ok=True)
#     # print("Saving undistorted images to " + output_directory)
#     # tmp_dir = f"./tmps/tmp_{time_step}"
#     # os.makedirs(tmp_dir, exist_ok=True)
    
#     for path in images_path:
#         cam_id = int(os.path.basename(os.path.dirname(path)))
        
#         image = cv2.imread(path)
#         if image is None:
#             raise Exception(f"Errro: Unable to read image {path}. Exiting.")
            
#         h, w = image.shape[:2]
#         new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix_list[cam_id], distortion_coefficients_list[cam_id], (w, h), 1, (w, h))
#         undistorted_image = cv2.undistort(image, camera_matrix_list[cam_id], distortion_coefficients_list[cam_id], None, new_camera_matrix)
#         x, y, w_new, h_new = roi
#         undistorted_image = undistorted_image[y:y+h_new, x:x+w_new]
#         resized_undistorted_image = cv2.resize(undistorted_image, (w, h))  
#         basename = os.path.basename(path)
#         cv2.imwrite(os.path.join(output_directory, str(cam_id), basename), resized_undistorted_image)
#     print(f"Saved undistorted images corresponding to {time_step} into {output_directory}")

# def worker_task(args):
#     images_path, time_step, output_directory, camera_matrix, distortion_coefficients = args
#     undistort_images(images_path, time_step, output_directory, camera_matrix, distortion_coefficients)

# def prepare_tuples_for_undistortion(directory_tuples, input_dir, output_dir, camera_file_path, num_timesteps):
#     # Prepare directory pairs for multiprocessing
#     # directory_tuples = []
#     camera_matrix_list, distortion_coefficients_list, _ = load_cameras_params(camera_file_path)
#     for ts in range(int(num_timesteps)):
#         files_to_be_undistorted = {}
#         for cam_dir in os.listdir(input_dir):
#             cam_dir_path = os.path.join(input_dir, cam_dir)
#             if os.path.isdir(cam_dir_path):
#                 os.makedirs(os.path.join(output_dir, cam_dir), exist_ok=True)
#                 if os.path.exists(os.path.join(cam_dir_path, f"{ts:06}.png")):
#                     files_to_be_undistorted.add(os.path.join(cam_dir_path, f"{ts:06}.png"))
#                 else:
#                     for k in range(5):
#                         time.sleep(5)
#                         if os.path.exists(os.path.join(cam_dir_path, f"{ts:06}.png")):
#                             files_to_be_undistorted.add(os.path.join(cam_dir_path, f"{ts:06}.png"))
#                             break
                    
#                     raise Exception(os.path.join(cam_dir_path, f"{ts:06}.png") + " does not exist. Exiting!")
#         # not os.path.join(cam_dir_path, f"{ts:06}.png") in directory_tuples:
#         directory_tuples.add((files_to_be_undistorted, ts, output_dir, camera_matrix_list, distortion_coefficients_list))
#     return directory_tuples
def symlink_force(target, link_name):
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e
def adjust_camera_matrix(K, original_size, new_size):
    """
    Adjust camera matrix for resized image.

    Args:
        K: Original camera matrix (3x3)
        original_size: (width, height) of the original image
        new_size: (width, height) of the resized image
    """
    # Calculate scaling factors
    scale_x = new_size[0] / original_size[0]
    scale_y = new_size[1] / original_size[1]

    # Create new camera matrix
    new_K = K.copy()
    new_K[0, 0] *= scale_x  # fx
    new_K[1, 1] *= scale_y  # fy
    new_K[0, 2] *= scale_x  # cx
    new_K[1, 2] *= scale_y  # cy

    return new_K

def load_xml_camera_params(file_name) :

    with open(file_name) as file:
    # Parse XML to dictionary
        data_dict = xmltodict.parse(file.read())

        # Access data like a regular dictionary
        width = float(data_dict['document']['chunk']['sensors']['sensor']['calibration']['resolution']["@width"])
        height = float(data_dict['document']['chunk']['sensors']['sensor']['calibration']['resolution']["@height"])
        fx = float(data_dict['document']['chunk']['sensors']['sensor']['calibration']['f'])
        fy = float(data_dict['document']['chunk']['sensors']['sensor']['calibration']['f'])
        cx = float(data_dict['document']['chunk']['sensors']['sensor']['calibration']['cx']) + (width - 1)/2
        cy = float(data_dict['document']['chunk']['sensors']['sensor']['calibration']['cy']) + (height - 1)/2
        k1 = float(data_dict['document']['chunk']['sensors']['sensor']['calibration']['k1'])
        k2 = float(data_dict['document']['chunk']['sensors']['sensor']['calibration']['k2'])
        k3 = float(data_dict['document']['chunk']['sensors']['sensor']['calibration']['k3'])
        p1 = float(data_dict['document']['chunk']['sensors']['sensor']['calibration']['p1'])
        p2 = float(data_dict['document']['chunk']['sensors']['sensor']['calibration']['p2'])

    camera_matrix = K = np.array([
                    [fx , 0, cx ],
                    [0, fy, cy],
                    [0, 0, 1]
                ])
    distortion_coffs = np.array([k1, k2, p1, p2, k3])


    len (data_dict['document']['chunk']['cameras']['camera'])
    cam_pose_list = []
    for id, cam in enumerate(data_dict['document']['chunk']['cameras']['camera']):
        # print(cam['@id'])
        transform = [float(t)  for t in cam['transform'].split() ]
        cam_pose_list.append(np.linalg.inv(np.array(transform).reshape(4,4)).tolist())
    
    return camera_matrix, distortion_coffs, cam_pose_list

def organize_and_undistort_by_timestep(dataset_path: str, undistorted_output_folder_path: str, camera_file_path, list_serial_number, valid_ts, iters, ts_id, num_timesteps):
   
    files_to_undistort = []
    dirname = os.path.dirname(undistorted_output_folder_path)

    for a, sn in enumerate(list_serial_number):
        # Create output directory if it doesn't exist
        cam_output_dir = os.path.join(dirname, 'ims', str(a))
        os.makedirs(cam_output_dir, exist_ok=True)

        # Try to copy current timestamp image
        current_file = os.path.join(dataset_path, f"frame_{valid_ts}_{sn}.png")
        output_file = os.path.join(cam_output_dir, f"{ts_id:06d}.png")

        if os.path.exists(current_file):
            symlink_force(current_file, output_file)
            # os.symlink(current_file, output_file)
            # shutil.copy2(current_file, output_file)
        else:
            if ts_id == num_timesteps - 1:
                file_prev =  f"{ts_id-1:06d}.png"
                prev_file_path = os.path.join(cam_output_dir, file_prev)
                os.symlink(prev_file_path, current_file)
                continue
            it = iter(iters)
            while True:
                item = next(it, None)
                next_file_path = os.path.join(dataset_path, f"frame_{item}_{sn}.png")
                if os.path.exists(next_file_path):
                    # shutil.copy2(next_file_path, current_file)
                    os.symlink(next_file_path, output_file)
                    break
                if item is None:
                    print("No succesive image found in 5 next timestamps")
                    return
           

        files_to_undistort.append(output_file)    
   
    print(f"Finished creating links for the images for {valid_ts} timestamp!", 
            file=sys.stderr)

    # camera_matrix_list, distortion_coefficients_list, _ = load_cameras_params(camera_file_path)
    camera_matrix, distortion_coefficients, _ = load_xml_camera_params("/home/hamit/Documents/agisoft_new.xml")

    # undistorted_output_folder_path = os.path.join(output_folder_path, "ims_undistorted")
    final_camera_matrix_dict ={}
    for id, file in enumerate(files_to_undistort):
        cam_id = int(os.path.basename(os.path.dirname(file)))
        os.makedirs(os.path.join(undistorted_output_folder_path, str(cam_id)), exist_ok=True)
        if os.path.getsize(file) > 0:
            image = cv2.imread(file)
            if image is None:
                raise Exception(f"Errro: Unable to read image {file}. Exiting.")
                
            h, w = image.shape[:2]
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coefficients, (w, h), 1, (w, h))

            # new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix_list[cam_id], distortion_coefficients_list[cam_id], (w, h), 1, (w, h))
            x, y, w_new, h_new = roi
            cropped_size = (w_new, h_new) 
            cropped_K = new_camera_matrix.copy()  
            cropped_K[0, 2] -= x  # Adjust principal point x
            cropped_K[1, 2] -= y
            # # print(new_camera_matrix)
            undistorted_image = cv2.undistort(image, camera_matrix, distortion_coefficients, None, new_camera_matrix)

            # undistorted_image = cv2.undistort(image, camera_matrix_list[cam_id], distortion_coefficients_list[cam_id], None, new_camera_matrix)
            undistorted_image = undistorted_image[y:y+h_new, x:x+w_new]
            final_size =(w, h)
            resized_undistorted_image = cv2.resize(undistorted_image, final_size)  
            basename = os.path.basename(file)
            final_K = adjust_camera_matrix(cropped_K, cropped_size, final_size)  
            if int(ts_id) == 0:
                final_camera_matrix_dict[cam_id] = final_K
            cv2.imwrite(os.path.join(undistorted_output_folder_path, str(cam_id), basename), resized_undistorted_image)
    print(f"Finished undistortion for the images for {valid_ts} timestamp")
    if int(ts_id) == 0:
        return final_camera_matrix_dict
    else:
        return None
def prepare_tuples_for_organize_undistort_by_timestep(list_valid_ts, dataset_path, undistorted_output_folder_path, camera_file_path, list_serial_number):
    tuples_for_organize_undistort=[]

    for id, ts in enumerate(list_valid_ts):
        # if id > 0:
        #     break
        iters = [t for t in list_valid_ts[id +1:id+6]]

        tuples_for_organize_undistort.append((dataset_path, undistorted_output_folder_path, camera_file_path, list_serial_number, ts, iters, id, len(list_valid_ts)))

    return tuples_for_organize_undistort   


def extract_frames_with_pts(cam_file_path, output_path, cam_serial, num_timesteps_interv):
    # Create output directory
    os.makedirs(output_path, exist_ok=True)

   # Get frame information using ffprobe
    probe = ffmpeg.probe(
        cam_file_path,
        v='quiet',
        select_streams='v:0',
        show_frames=None,
        of='json'
    )  
    frames_data = probe['frames']  # List of frame information  

    

    try:
        # Extract frames using ffmpeg-python with chained operations
        (
            ffmpeg
            .input(cam_file_path)
            .filter('select', f'between(n,{num_timesteps_interv[0]},{num_timesteps_interv[1] + 50})')
            .output(f'{output_path}/frame_%03d_{cam_serial}.png', 
                   vsync='0',
                   start_number=0)
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        print('stdout:', e.stdout.decode('utf8'))
        print('stderr:', e.stderr.decode('utf8'))
        raise e

    # Rename files with PTS
    for i, frame in enumerate(frames_data, 0):
        pts_time = frame["pkt_pts"]
        old_name = f'{output_path}/frame_{i:03}_{cam_serial}.png'
        new_name = f'{output_path}/frame_{pts_time}_{cam_serial}.png'
       
        if os.path.exists(old_name):
            print(old_name, new_name)
            os.rename(old_name, new_name)

def prepare_tuples_for_extract_frames(mp4s_path, dataset_path, num_timesteps_intver):
    

    inputs=[]
    mp4_files_full = sorted([file for file in os.listdir(mp4s_path) if file.endswith(".mp4") and file.startswith("Cam-Full_")])
    
    for cam_index, cam_file in enumerate(mp4_files_full):
        # if cam_index > 2:
        #     break
        cam_file_base= os.path.splitext(cam_file)[0]
        cam_serial = cam_file_base.split('_')[-1]
        cam_file_path= os.path.join(mp4s_path, cam_file)
        inputs.append((cam_file_path, dataset_path, cam_serial, num_timesteps_intver))
    
    mp4_files =  sorted([file for file in os.listdir(mp4s_path) if file.endswith(".mp4") and file.startswith("Cam_")])
    for cam_index, cam_file in enumerate(mp4_files):
    # if cam_index > 2:
    #     break
        cam_file_base= os.path.splitext(cam_file)[0]
        cam_serial = cam_file_base.split('_')[-1]
        cam_file_path= os.path.join(mp4s_path, cam_file)
        inputs.append((cam_file_path, dataset_path, cam_serial, num_timesteps_intver))
    
    return inputs

def reorganize_folder_for_missing(ims_folder, number_timestamps):
    
    folders = [folder for folder in sorted(os.listdir(ims_folder)) if os.path.isdir(os.path.join(ims_folder, folder))]
    for folder in folders:
        idx=0
        for ts in range(number_timestamps):
            iters = [id for id in range(ts+1, ts+6)]
            file = f"{ts:06d}.png"
            if not os.path.exists(os.path.join(ims_folder,folder,file)):
                print(f"{os.path.join(ims_folder,folder, file)} is missing, trying to find the possible image from next timestamps")
                if ts == number_timestamps - 1:
                    file_prev =  f"{ts-1:06d}.png"
                    im_file_prev_path = os.path.join(ims_folder,folder, file_prev)
                    im_file_path = os.path.join(ims_folder,folder, file)
                    shutil.copy2(im_file_prev_path, im_file_path)
                   
                    two_up_folder=os.path.basename(os.path.dirname(os.path.dirname(im_file_prev_path)))
                    seg_file_prev_path = im_file_prev_path.replace(f"/{two_up_folder}/","/seg/")
                    seg_file_path = im_file_path.replace(f"/{two_up_folder}/","/seg/")
                    shutil.copy2(seg_file_prev_path, seg_file_path)
                    continue

                it = iter(iters)
                while True:
                    item = next(it, None)
                    next_file = f"{item:06d}.png"
                    next_im_file_path = os.path.join(ims_folder,folder,next_file)
                    
                    if os.path.exists(next_im_file_path):
                       
                        two_up_folder=os.path.basename(os.path.dirname(os.path.dirname(next_im_file_path)))
                        next_seg_file_path = next_im_file_path.replace(f"/{two_up_folder}/","/seg/")
                        im_file_path =  os.path.join(ims_folder,folder,file)
                        seg_file_path = im_file_path.replace(f"/{two_up_folder}/","/seg/")
                        shutil.copy2(next_im_file_path, im_file_path)
                        shutil.copy2(next_seg_file_path, seg_file_path)

                        break
                    if item is None:
                        print("No succesive image found in 5 next timestamps")
                        return

def create_namespace_for_masking():
   return Namespace(add_background=False, adjust_color=False, background_imgs_path=None, camid=0, 
    ckpt='/home/hamit/Softwares/YOLOX/model_weights/yolox_x.pth', conf=0.25, 
    demo='images', device='gpu', exp_file=None, experiment_name='yolox_x', 
    fp16=False, fuse=False, legacy=False, mask_video_output=None, name='yolox-x', 
    nms=0.45, path="", 
    sam_checkpoint='/home/hamit/Softwares/segment-anything/model_weights/sam_vit_h_4b8939.pth', 
    sam_model_type='vit_h', save_result=True, scale_factor=1.0, trt=False, tsize=640)

def prepare_tuples_for_masking(num_timesteps, output_path):
    
    tuples_for_masking=[]
   

    for id in range(num_timesteps):

        # temp_folder=f"./tmps/masking_tmp_{id}/"
        # if os.path.exists(temp_folder):
        #     shutil.rmtree(temp_folder) 
        # os.makedirs(temp_folder)

        # files = glob.glob(os.path.join(output_path, f"**/{id:06}.png"), recursive=True)
        # files = sorted(files, key=lambda x: int(os.path.basename(os.path.dirname(x))))
        
        # for idx, file in enumerate(files):
        #     os.symlink(file, f"{temp_folder}/{idx}.png")

        tuples_for_masking.append((id, output_path))

    return tuples_for_masking

def generate_masked_images(ts_id, output_path):
    
    temp_folder=f"/tmp/masking_tmp_{ts_id}/"
    if os.path.exists(temp_folder):
        shutil.rmtree(temp_folder) 
    os.makedirs(temp_folder)

    files = glob.glob(os.path.join(output_path, f"**/{ts_id:06}.png"), recursive=True)
    files = sorted(files, key=lambda x: int(os.path.basename(os.path.dirname(x))))
    for idx, file in enumerate(files):
        os.symlink(file, f"{temp_folder}/{idx}.png")
    
    output_path = os.path.dirname(output_path)

    args_for_masking = create_namespace_for_masking()  
    args_for_masking.path = temp_folder
   
    # args_namespace = Namespace(**dict(args_for_masking))
    exp = generate_masked_data.get_exp(args_for_masking.exp_file, args_for_masking.name)
    generate_masked_data.main(exp, args_for_masking)
    

    files = glob.glob(os.path.join(temp_folder, "masked_undistorted_images" , "*_black.png"))
    files = sorted(files, key=lambda x: int(os.path.basename(x).split('_')[0]))

    for id, file in enumerate(files):
        os.makedirs(os.path.join(output_path, "ims_black", str(id)), exist_ok=True)
        shutil.move(file, os.path.join(output_path, "ims_black", str(id), f"{ts_id:06}.png"))
    
    files = glob.glob(os.path.join(temp_folder, "masked_undistorted_images" , "*_black_white.png"))
    files = sorted(files, key=lambda x: int(os.path.basename(x).split('_')[0]))

    for id, file in enumerate(files):
        os.makedirs(os.path.join(output_path, "seg", str(id)), exist_ok=True)
        shutil.move(file, os.path.join(output_path, "seg", str(id), f"{ts_id:06}.png"))

    shutil.rmtree(temp_folder)    
    # os.makedirs(os.path.join(os.path.dirname(output_path), "seg", str(id)), exist_ok=True)

    #     if "_black.png" in file:
    #         shutil.move(file, os.path.join(os.path.dirname(output_path), "ims_black", str(id), f"{ts_id:06}.png"))
    #     elif "_black_white.png" in file:
    #         shutil.move(file, os.path.join(os.path.dirname(output_path), "seg", str(id), f"{ts_id:06}.png"))
        
        
def prepare_final_step(folder_path, camera_file_path, dataset_name, k_matrix):
    import open3d as o3d
    # global final_camera_matrix_dict
    folder_path = os.path.dirname(folder_path)
    ims_folder = os.path.join(folder_path, 'ims_black')
    pt_cld_path = os.path.join(folder_path,  'points3D.ply')
    pt_cld = o3d.io.read_point_cloud(pt_cld_path)
    # scale=3.3299749231959184/0.2057002568030526
    # pt_cld.scale(1/scale, center=(0,0,0))
    xyzs, rgbs = np.asarray(pt_cld.points), np.asarray(pt_cld.colors)
    seg = np.ones_like(xyzs[:, 0])[:, None]   # Always static for now, segmentation always 1
    pt_cld = dict()
    pt_cld = np.concatenate((xyzs, rgbs, seg), axis=1).tolist()
    output_path = './data'
    if not os.path.exists(os.path.join(output_path, dataset_name)):
        os.makedirs(os.path.join(output_path, dataset_name))
    np.savez(os.path.join(output_path, dataset_name, 'init_pt_cld.npz'), data=pt_cld)
    print('Point cloud saved')
    data = dict()

    data['w'] = 4096 # sorted_intr[1].width
    data['h'] = 3000 # sorted_intr[1].height
    
    # _, _, w2c = load_cameras_params(camera_file_path) 
    _, _, w2c = load_xml_camera_params('/home/hamit/Documents/agisoft_new.xml')
    
    # kk = list(k_matrix.values())
    kk = k_matrix
    k = [ it.tolist() for it in kk]
    # Get images
    fn_all, cam_id_all, k_all, w2c_all = create_list_cameras_params( ims_folder,  k, w2c)
    data['k'] = k_all
    data['w2c'] = w2c_all
    data['fn'] = fn_all # Add dimension as I only have 1 timestamp for now 
    data['cam_id'] = cam_id_all  

    with open(os.path.join(output_path, dataset_name, 'train_meta.json'), 'w') as f:
        json.dump(data, f)    
    


def main():
    parser = argparse.ArgumentParser(description="Undistort images using camera parameters")
    parser.add_argument("--mp4s_path", default=None, type=str, help="Path to the folder where mp4s exist")
    parser.add_argument("--dataset_path", required=True, help="Base directory to save all images")
    parser.add_argument("--undistorted_output_folder_path",  required=True, help="Base directory to save processed images")
    parser.add_argument("--camera_yml_file_path",  required=True , help="Path to the yml file containing camera parameters")
    parser.add_argument("--cam_number", required=True, type=int, help="Number of cameras in the camera rig setup")
    parser.add_argument("--num_timesteps_interv", nargs=2, default=(300, 600), type=int, help="Number of timesteps to be processed")
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name.')

    args = parser.parse_args()
    num_timesteps = args.num_timesteps_interv[1] - args.num_timesteps_interv[0]
    
    # if  args.mp4s_path:
    #     tuples_for_extract_frames = prepare_tuples_for_extract_frames(args.mp4s_path, args.dataset_path, args.num_timesteps_interv)
    

    #     with multiprocessing.Pool(processes=8) as pool_extract_frames:
    #         results = []  
    #         for tuple in tuples_for_extract_frames:
    #             result = pool_extract_frames.apply_async(extract_frames_with_pts, tuple)  
    #             results.append(result)  
    #         results = [r.get() for r in results] 
            
    #     print("Extract Frames Results:", results)  



    # list_valid_ts, list_serial_number = calculate_number_valid_timesteps(args.dataset_path, args.cam_number, num_timesteps)
    # tuples_for_organize_undistort = prepare_tuples_for_organize_undistort_by_timestep(list_valid_ts, args.dataset_path, args.undistorted_output_folder_path, args.camera_yml_file_path, list_serial_number)
    
    # with multiprocessing.Pool(processes=8) as pool_organize_undistort:
    #     results = []  
    #     for tuple in tuples_for_organize_undistort:
    #         result = pool_organize_undistort.apply_async(organize_and_undistort_by_timestep, tuple)  
    #         results.append(result)  
    #     results = [r.get() for r in results] 
    # print("Organize and Undistort Results:", results)  

    # k_matrix = deepcopy(results[0])
    

   

    # tuples_for_masking = prepare_tuples_for_masking(num_timesteps, args.undistorted_output_folder_path)
    
    # with multiprocessing.Pool(processes=3) as pool_masking:
    #     # pool_masking.starmap(generate_masked_images, tuples_for_masking)
    #     results = []  
    #     for tuple in tuples_for_masking:
    #         result = pool_masking.apply_async(generate_masked_images, tuple)  
    #         results.append(result)  
        
    #     results = [r.get() for r in results] 
    # print("Masking Results:", results)  
    
    k_matrix = [np.array([[2.95417160e+03, 0.00000000e+00, 2.02975805e+03],
       [0.00000000e+00, 2.98711764e+03, 1.46593560e+03],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])]*24
    prepare_final_step(args.undistorted_output_folder_path, args.camera_yml_file_path, args.dataset_name, k_matrix)
    
    
    
    # parser_masking = argparse.ArgumentParser(description="Masking parameters for yolox and segment-anything")


    


    # python data_making/undistort_opencv.py --dataset_path /mnt/Elements2/19-12-2024_Data/2024-12-19_19-12-14_4096/all_frames_10-20 --output_folder_path /home/hamit/DATA/19-12-2024_Data/2024-12-19_19-12-14_4096/processed_data_3010-20_calib --camera_yml_file_path /home/hamit/DATA/19-12-2024_Data/2024-12-19_19-07-50_4096_calib/cameras_calibs_data_51mm_4096.yml --cam_number 24  --num_timesteps_interv 10 20 --mp4s_path /media/hamit/HamitsKingston/19-12-2024_Data/2024-12-19_19-12-14_4096/

        # pool_organize_undistort.starmap(organize_and_undistort_by_timestep, tuples_for_multiprocess)
    
    # thread_organize = threading.Thread(target=organize_images_folder_by_cam, args=(args.dataset_path, args.output_folder_path, args.cam_number, args.num_timesteps))  
    # thread_organize.start()
    
    # time.sleep(15)
    # directory_tuples= {}
    # results=[]
    # with multiprocessing.Pool(processes=2) as pool_undistort:
        
    #     for k in range(args.num_timesteps):
    #         directory_tuples = prepare_tuples_for_undistortion(directory_tuples, args.output_folder_path, os.path.join(args.output_folder_path, "undistorted"), args.camera_yml_file_path, args.num_timesteps)
        
    #         results.append(pool_undistort.starmap(undistort_images, directory_tuples))


    
    # directory_tuples = prepare_tuples_for_undistortion(args.output_folder_path, os.path.join(args.output_folder_path, "undistorted"), args.camera_yml_file_path, args.num_timesteps)
    # # Load camera parameters
    # camera_matrix_list, distortion_coefficients_list = load_cameras_params(args.camera_file_path)

    # # Prepare directory pairs for multiprocessing
    # input_directory = args.input_directory
    # directory_pairs = []
    # for ts in range(int(args.num_timesteps)):
    #     files_to_be_undistored = []
    #     for cam_dir in os.listdir(input_directory):
    #         cam_dir_path = os.path.join(input_directory, cam_dir)
    #         if os.path.isdir(cam_dir_path):
    #             os.makedirs(os.path.join(args.output_directory, cam_dir), exist_ok=True)
    #             if os.path.exists(os.path.join(cam_dir_path, f"{ts:06}.png")):
    #             #    new_file_link = os.path.join(cam_dir_path, f"{cam_dir}.png")
    #             #    os.link(os.path.join(cam_dir_path, f"{ts:06}.png"), new_file_link)
    #                files_to_be_undistored.append(os.path.join(cam_dir_path, f"{ts:06}.png"))
    #             else:
    #                 raise Exception(os.path.join(cam_dir_path, f"{ts:06}.png") + " does not exist. Exiting!")
    #             output_directory = args.output_directory
    #     directory_pairs.append((files_to_be_undistored, ts, output_directory, camera_matrix_list, distortion_coefficients_list))

 

if __name__ == "__main__":
    main()
