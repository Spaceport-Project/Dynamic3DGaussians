import io
import argparse
import glob
import multiprocessing
import os
import re
import sys
from pathlib import Path
from collections import Counter
import cv2
import ffmpeg
from PIL import Image

import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
from rembg import new_session, remove
import numpy as np
import xml.etree.ElementTree as ET
import imagesize  




def ims_index(path):
    # Get folder name just before the filename
    # .../ims/<number>/000001.png
    folder = os.path.basename(os.path.dirname(path))
    # Fallback: extract last integer in the path
    m = re.search(r'(\d+)', folder)
    return int(m.group(1)) if m else 0

def extract_frames_from_mp4s(dataset_path, mp4s_path, cam_prefix, num_timesteps_interv ):
    if  args.mp4s_path:
        tuples_for_extract_frames = prepare_tuples_for_extract_frames(prefix=cam_prefix, mp4s_path=mp4s_path, dataset_path=dataset_path, num_timesteps_intver=num_timesteps_interv)
        

        with multiprocessing.Pool(processes=18) as pool_extract_frames:
            results = []  
            for tuple in tuples_for_extract_frames:
                result = pool_extract_frames.apply_async(extract_frames_with_pts, tuple)  
                results.append(result)  
            results = [r.get() for r in results] 
            
        print("Extract Frames Results:", results)  

def prepare_tuples_for_extract_frames(prefix, mp4s_path, dataset_path, num_timesteps_intver):
    

    inputs=[]
   
    
    mp4_files =  sorted([file for file in os.listdir(mp4s_path) if file.endswith(".mp4") and file.startswith(prefix)])
    for cam_index, cam_file in enumerate(mp4_files):
    # if cam_index > 2:
    #     break
        cam_file_base= os.path.splitext(cam_file)[0]
        cam_prefix = cam_file_base.split('_')[0]
        cam_serial = cam_file_base.split('_')[-1]
        cam_file_path= os.path.join(mp4s_path, cam_file)
        inputs.append([cam_file_path, cam_prefix, dataset_path, cam_serial, num_timesteps_intver])
    
    return inputs
def extract_frames_with_pts(cam_file_path, cam_prefix, dataset_path, cam_serial, num_timesteps_interv, frames_data, valid_last_pts_time, ext="png"):
    # Create output directory
    os.makedirs(dataset_path, exist_ok=True)



    

    try:
        # Extract frames using ffmpeg-python with chained operations
        (
            ffmpeg
            .input(cam_file_path)
            # .filter('select', f'between(n,{num_timesteps_interv[0]},{num_timesteps_interv[1]})')
            # .filter('select', f'between(n,{num_timesteps_interv[0]},{num_timesteps_interv[1]})* \
            #         not(mod(n-{num_timesteps_interv[0]},5))')
            .filter('select', f'between(n,{num_timesteps_interv[0]},{num_timesteps_interv[1]-1})') 
            # .filter('select', 'not(mod(n,2))')  
            .output(f'{dataset_path}/{cam_prefix}_%06d_{cam_serial}.{ext}', **{'q:v': 1},
                   vsync='0',
                   start_number=num_timesteps_interv[0]
                   )
            .run(capture_stdout=True, capture_stderr=True)
        )

        
        # (
        #     ffmpeg
        #     .input(cam_file_path, hwaccel='cuda')
        #     .filter('select', f'between(n,{num_timesteps_interv[0]},{num_timesteps_interv[1]})')
        #     .output(f'{output_path}/{cam_prefix}_%03d_{cam_serial}.{ext}',
        #             **{'q:v': 1},
        #             vsync='0',
        #             start_number=0)
        #     .run(capture_stdout=True, capture_stderr=True)
        # )

       
        
    except ffmpeg.Error as e:
        print('stdout:', e.stdout.decode('utf8'))
        print('stderr:', e.stderr.decode('utf8'))
        raise e

    # Rename files with PTS
    valid_pts_time = None
    for i, frame in enumerate(frames_data[num_timesteps_interv[0]:num_timesteps_interv[1]], num_timesteps_interv[0]):
       
        pts_time = frame["pkt_pts"]
        old_name = f'{dataset_path}/{cam_prefix}_{i:06}_{cam_serial}.{ext}'

        if pts_time > valid_last_pts_time :
            new_name = f'{dataset_path}/{cam_prefix}_{pts_time}_{cam_serial}.{ext}'
        
            if os.path.exists(old_name):
                # print(old_name, new_name)
                os.rename(old_name, new_name)
                valid_pts_time = pts_time
        else:
            os.remove(old_name)
    print(f"{cam_file_path} has been extracted!")

    return valid_pts_time
    

def calculate_number_valid_timesteps(dataset_path, cam_prefix, ext, cam_number):
    """
    Calculate valid timestamps based on camera image availability.
    
    Args:
        dataset_path: Path to the dataset directory
        cam_prefix: Camera filename prefix
        ext: File extension (e.g., 'jpg', 'png')
        cam_number: Expected number of cameras
    
    Returns:
        List of valid timestamps
    """
    temp_list = []
    
    # Find all matching files and extract timestamps
    pattern = f"{cam_prefix}*.{ext}"
    files = sorted(Path(dataset_path).glob(pattern))
    
    # Extract timestamps from filenames (second to last part when split by '_')
    timestamps = []
    for file in files:
        parts = file.stem.split('_')
        if len(parts) >= 2:
            timestamps.append(parts[-2])
    
    # Count occurrences of each timestamp
    timestamp_counts = Counter(timestamps)
    
    # Find first timestamp that appears cam_number times
    sorted_unique_timestamps = sorted(set(timestamps), key=lambda x: int(x) if x.isdigit() else x)
    first_timestamp = None
    for ts in sorted_unique_timestamps:
        if timestamp_counts[ts] == cam_number:
            first_timestamp = ts
            break
    
    if first_timestamp is None:
        print(f"Warning: No timestamp found with exactly {cam_number} images", file=sys.stderr)
        return temp_list
    
    # Process timestamps starting from first_timestamp
    for idx, timestamp in enumerate(sorted_unique_timestamps):
        # Skip timestamps before first_timestamp
        if int(timestamp) < int(first_timestamp):
            continue
        
        
        
        # Count current images for this timestamp
        current_cam_number = sum(1 for f in Path(dataset_path).glob(f"*{timestamp}*.{ext}") 
                                  if f.is_file())
        
        cam_number_minus_one = cam_number - 1
        

        # if current_cam_number < cam_number:
        #     print(f"There exist {current_cam_number} which is less than {cam_number} "
        #           f"Continuing to next timesteps!", 
        #           file=sys.stderr)
        #     continue
        
        # Check if we have enough images
        if current_cam_number < cam_number_minus_one:
            print(f"There must be at least {cam_number_minus_one} images out of {cam_number} "
                  f"for a timestamp, but there exist {current_cam_number} images for the "
                  f"following time stamp '{timestamp}'.\n"
                  f"Exiting from the function of calculating number of valid timesteps!", 
                  file=sys.stderr)
            break
        elif current_cam_number == cam_number_minus_one:
            print(f'There exist {cam_number_minus_one} images for the time stamp {timestamp}')
        
        
        temp_list.append(timestamp)
    
    return temp_list

def organize_images_folder_by_cam(list_valid_ts, dataset_path, output_folder_path, id_counter, cam_folder,
                                   cam_prefix, ext, cam_number):
    """
    Organizes images from a dataset folder into structured output folders by camera.
    
    Args:
        list_valid_ts: List of valid timestamps
        dataset_path: Path to the dataset directory
        output_folder_path: Path to the output directory
        cam_prefix: Camera filename prefix
        ext: File extension (e.g., 'jpg', 'png')
        cam_number: Number of cameras
        cam_folder: Camera folder name
    """
    
    # Find first timestamp
    pattern = os.path.join(dataset_path, f"{cam_prefix}*.{ext}")
    files = sorted(glob.glob(pattern))
    
    # Extract timestamps and count occurrences
    timestamp_counts = {}
    for file in files:
        parts = os.path.basename(file).split('_')
        timestamp = parts[-2]
        timestamp_counts[timestamp] = timestamp_counts.get(timestamp, 0) + 1
    
    # Find first timestamp with cam_number occurrences
    first_timestamp = None
    for ts in sorted(timestamp_counts.keys()):
        if timestamp_counts[ts] == cam_number:
            first_timestamp = ts
            break
    
    if first_timestamp is None:
        print("Could not find valid first timestamp", file=sys.stderr)
        return
    
    # Get list of serial numbers from first timestamp
    pattern = os.path.join(dataset_path, f"{cam_prefix}{first_timestamp}_*.{ext}")
    files = sorted(glob.glob(pattern))
    list_serial_numbers = []
    for file in files:
        parts = os.path.basename(file).split('_')
        serial = parts[-1].split('.')[0]
        list_serial_numbers.append(serial)
    
    exit_flag = False
    # id_counter = 0
    output_dir = os.path.join(output_folder_path, cam_folder)
    # if os.path.isdir(output_dir):
    #     shutil.rmtree(str(output_dir)) 
    for idx in range(len(list_valid_ts)):
        # Get next 6 timestamps for fallback
        iters = []
        for offset in range(1, 7):
            if idx + offset < len(list_valid_ts):
                iters.append(list_valid_ts[idx + offset])
        
        a = 0
        for sn in list_serial_numbers:
            # Create output directory if it doesn't exist
            output_dir = os.path.join(output_folder_path, cam_folder, str(a))
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Check if primary file exists
            file_path = os.path.join(dataset_path, f"{cam_prefix}{list_valid_ts[idx]}_{sn}.{ext}")
            formatted_number = f"{id_counter:06d}"
            output_file = os.path.join(output_dir, f"{formatted_number}.{ext}")
            
            if os.path.exists(file_path):
                # Create symbolic link
                # os.symlink(file_path, output_file)
                os.rename(file_path, output_file)
            else:
                # Try to find file in next 6 timestamps
                kk = 0
                found = False
                
                for it in iters:
                    next_file = os.path.join(dataset_path, f"{cam_prefix}{it}_{sn}.{ext}")
                    if os.path.exists(next_file):
                        print(f"Copying {next_file} to {output_file}", file=sys.stderr)
                        # os.symlink(next_file, output_file)
                        shutil.copy2(next_file, output_file)
                        found = True
                        break
                    kk += 1
                
                if not found:
                    print("Cannot find a image within next 6 time stamps. It is because either "
                          "you have reached the end of the list or something wrong with the recordings. "
                          "Make sure your recording does not have that many missing images. Exiting!", 
                          file=sys.stderr)
                    exit_flag = True
                    break
            
            if exit_flag:
                break
            
            a += 1
        
        if exit_flag:
            break
        
        print(f"Copying images from {a} cameras for {list_valid_ts[idx]} timestamp has finished!", 
              file=sys.stderr)
        
        id_counter += 1

# # Example usage:
# dataset_path="/home/hamit/DATA/processed_data/12-11-2005_Data/2025-11-12_14-06-52_aliyenur_4/all_frames"
# valid_timestamps = calculate_number_valid_timesteps(
#     dataset_path=dataset_path,
#     cam_prefix="Cam_",
#     ext="png",
#     cam_number=44
# )
# output_folder = '/home/hamit/DATA/processed_data/12-11-2005_Data/2025-11-12_14-06-52_aliyenur_4/ims_deneme'
# organize_images_folder_by_cam(list_valid_ts=valid_timestamps, dataset_path= dataset_path, output_folder_path=output_folder, cam_prefix= "Cam_", cam_number=44, ext="png")

# print(valid_timestamps)
# print(len(valid_timestamps))

def run_colmap(output_folder_path, cam_folder, ext):
    """
    Python version of the run_colmap bash function.
    
    Args:
        output_folder_path: Path to the output folder
        cam_folder: Camera folder name
        ext: File extension (e.g., 'jpg', 'png')
    """
    # Create colmap_input directory structure
    colmap_input = os.path.join(output_folder_path, "colmap_input")
    input_dir = os.path.join(colmap_input, "input")
    
    if not os.path.exists(input_dir):
        os.makedirs(input_dir, exist_ok=True)
    
    # Find and copy files
    ims_path = os.path.join(output_folder_path, cam_folder)
    
    # Find all files matching the pattern
    # pattern = os.path.join(ims_path, "**", f"000001.{ext}")
    # files = sorted(glob.glob(pattern, recursive=True))
    # def natural_sort_key(path):
    #     """Extract numbers from path for natural sorting"""
    #     parts = re.split(r'(\d+)', path)
    #     return [int(part) if part.isdigit() else part for part in parts]
    

    pattern = os.path.join(ims_path, "**", f"000020.{ext}")  
    files = glob.glob(pattern, recursive=True)
    files = sorted(files, key=ims_index)
    # pattern = os.path.join(ims_path, "**", f"000000.{ext}")
    # files = sorted(glob.glob(pattern, recursive=True), key=natural_sort_key)
    # Copy files with sequential numbering
    for idx, file in enumerate(files):
        dest_file = os.path.join(input_dir, f"{idx}.{ext}")
        shutil.copy(file, dest_file)
    
    # cmd = [
    #         'python', 'convert_mask.py',
    #         '-s', colmap_input,
    #         '--no_gpu'
           
    #     ]
    # subprocess.run(cmd, check=True)
def run_colmap_undistort_frames( output_folder_path, number_timesteps, cam_folder, ext):
    """
    Python version of run_colmap_undistort_frames bash function
    
    Args:
        output_folder_path: Base output folder path
        number_timesteps: Number of timesteps to process
        cam_folder: Camera folder name
        ext: File extension (e.g., 'jpg', 'png')
    """
    
    max_jobs = 30
    
    def colmap_undistort(file_name, folder):
        """Run COLMAP undistortion on a folder"""
        # Run COLMAP image undistorter
        cmd = [
            'colmap', 'image_undistorter',
            '--image_path', folder,
            '--input_path', f'{output_folder_path}/colmap_input/distorted_sparse_aligned/',
            '--output_path', folder,
            '--output_type', 'COLMAP'
        ]
        subprocess.run(cmd, check=True)
        
        print(f"Moving {file_name} files!")
        
        # Find and process all images
        images_path = Path(folder) / 'images'
        if images_path.exists():
            image_files = sorted(images_path.glob(f'*.{ext}'))
            
            for file in image_files:
                fol = file.stem  # filename without extension
                output_dir = Path(output_folder_path) / cam_folder / fol
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Move file to destination
                dest_file = output_dir / file_name
                if dest_file.is_symlink():
                    # Remove the symlink *itself*; target stays untouched
                    dest_file.unlink()

                    # Now move src into that path (now just a normal filename)
                    shutil.move(str(file), str(dest_file))
                else:
                    shutil.move(str(file), str(dest_file))
    def process_timestep(id_num, temp_base):
        """Process a single timestep"""
        temp_folder = f"/tmp/tmp_{temp_base}"
        temp_path = Path(temp_folder)
        
        # Clean and create temp folder
        if temp_path.exists():
            shutil.rmtree(temp_folder)
        temp_path.mkdir(parents=True, exist_ok=True)
        
        formatted_number = f"{id_num:06d}"
        file_name = f"{formatted_number}.{ext}"
        
        
        # Create symbolic links
        search_path = Path(output_folder_path) / cam_folder
        matching_files = (search_path.rglob(file_name))
        matching_files = sorted(matching_files, key=ims_index)  

        
        for a, file in enumerate(matching_files):
            link_path = temp_path / f"{a}.{ext}"
            # link_path.symlink_to(file.resolve())
            os.symlink( str(file) , str(link_path))
        
        # Run COLMAP undistortion
        try:
            colmap_undistort(file_name, temp_folder)
        finally:
            # Clean up temp folder
            if temp_path.exists():
                shutil.rmtree(temp_folder)
        
        return id_num
    
    # Process all timesteps with thread pool
    with ThreadPoolExecutor(max_workers=max_jobs) as executor:
        futures = {}
        
        for id_num in range(number_timesteps):
            future = executor.submit(process_timestep, id_num, id_num)
            futures[future] = id_num
        
        # Wait for all tasks to complete
        for future in as_completed(futures):
            id_num = futures[future]
            try:
                result = future.result()
                print(f"Completed timestep {result}")
            except Exception as e:
                print(f"Error processing timestep {id_num}: {e}")
    
def create_cam_matrix_and_dist_coeff(xml_path):
    # ---- 1. Read Metashape XML from file ----
#    / xml_path = "camposes_metashape.xml"   # <-- change this to your filename

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Assuming one sensor (id=0) as in your snippet:
    calibration = root.find('.//sensor[@id="0"]/calibration')
    if calibration is None:
        raise RuntimeError("Could not find <calibration> node under sensor id=0")

    resolution = calibration.find('resolution')
    width = int(resolution.get('width'))
    height = int(resolution.get('height'))

    # ---- 2. Extract calibration parameters ----
    f  = float(calibration.find('f').text)
    cx = float(calibration.find('cx').text)
    cy = float(calibration.find('cy').text)
    k1 = float(calibration.find('k1').text)
    k2 = float(calibration.find('k2').text)
    k3 = float(calibration.find('k3').text)
    p1 = float(calibration.find('p1').text)
    p2 = float(calibration.find('p2').text)

    # Metashape cx, cy are offsets from image center → convert to absolute principal point
    pp_x = width  / 2.0 + cx
    pp_y = height / 2.0 + cy

    # ---- 3. Create camera matrix (NumPy) ----
    K = np.array([
        [f, 0,   pp_x],
        [0, f,   pp_y],
        [0, 0,   1.0 ]
    ], dtype=np.float64)

# ---- 4. Create distortion coefficients (NumPy, OpenCV convention) ----
# OpenCV: [k1, k2, p1, p2, k3]
    dist = np.array([k1, k2, p1, p2, k3], dtype=np.float64)
    return K, dist
def undistort_images( output_dir, camera_matrix, distortion_coefficients, img_ext="png"):
    """Undistort images in the specified directory."""
    
    for file in os.listdir(output_dir):
        if not file.endswith(img_ext):
            continue
        image_path = os.path.join(output_dir, file)
        print('Processing image:', image_path)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Unable to read image {image_path}. Skipping.")
            continue

        h, w = image.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coefficients, (w, h), 1, (w, h))
        undistorted_image = cv2.undistort(image, camera_matrix, distortion_coefficients, None, new_camera_matrix)

        # mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, distortion_coefficients, None, new_camera_matrix (w,h), 5)
        # undistorted_image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
        x, y, w, h = roi
        undistorted_image = undistorted_image[y:y+h, x:x+w]

        basename = os.path.basename(image_path)
        os.makedirs(os.path.join(output_dir, "images"),exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, "images", basename), undistorted_image)

    # return undistorted_image

def colmap_undistort(input_folder_path, folder):
        """Run COLMAP undistortion on a folder"""
        # Run COLMAP image undistorter
        cmd = [
            'colmap', 'image_undistorter',
            '--image_path', folder,
            '--input_path', f'{input_folder_path}/colmap_input/distorted_sparse_aligned/',
            # '--input_path', f'{input_folder_path}/3DF/distorted/',
            '--output_path', folder,
            '--output_type', 'COLMAP'
        ]
        subprocess.run(cmd, check=True)

def process_timestamp( id_num, input_folder_path, cam_folder,  output_fol_name, seg_output_fol_name,  model_name1, model_name2, gpu_id, ext="png"):
        
        """Process a single timestep"""

        
        temp_folder = f"/tmp/tmp_{id_num}"
        temp_root = Path(temp_folder)
        
        input_root = Path(input_folder_path)

        # Clean and create temp folder
        if temp_root.exists():
            shutil.rmtree(temp_folder)
        temp_root.mkdir(parents=True, exist_ok=True)
        
        formatted_number = f"{id_num:06d}"
        file_name = f"{formatted_number}.{ext}"
        
        
        # Create symbolic links
        parent =  input_root.parent
        search_path = input_root / cam_folder 
        matching_files = (search_path.rglob(file_name))
        matching_files = sorted(matching_files, key=ims_index)  
        width, height = imagesize.get(matching_files[0])
        
        for a, file in enumerate(matching_files):
            link_path = temp_root / f"{a}.{ext}"
            # link_path.symlink_to(file.resolve())
            os.symlink( str(file) , str(link_path))
        
        # Run COLMAP undistortion
        try:
            colmap_undistort(str(input_root), temp_folder)
            # undistort_images( temp_folder, camera_matrix, dist_coeff)
        
            # Set the GPU for this process
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            
            
            session1 = new_session(model_name1)
            session2 = new_session(model_name2)
           

            for file in os.listdir(os.path.join(temp_folder, "images")):
                input_path = Path(os.path.join(temp_folder,"images", file))
               
                input_data = Image.open(str(input_path))
                # input_data =cv2.imread(input_path)
                # input_data = cv2.resize(input_data, (width, height), interpolation=cv2.INTER_AREA)  

                # with open(str(input_path), 'rb') as i:
                    # input_data = Image.open(str(input_path))
                    # input_data = i.read()
                output = remove(input_data, session=session1, post_process_mask=True)
            
                
                if has_leftover_background(output):
                    print(f"[GPU {gpu_id}] Leftover background detected in {file.name}, applying birefnet-general again...")
                    output = remove(output, session=session1, post_process_mask=True) # alpha_matting=True, alpha_matting_foreground_threshold=270,alpha_matting_background_threshold=20, alpha_matting_erode_size=11)
                    
                    # Step 3: Check again and apply u2net_human_seg if still needed
                    if has_leftover_background(output, threshold=0.01):
                        print(f"[GPU {gpu_id}] Still needs refinement for {file.name}, applying u2net_human_seg...")
                        output = remove(output, session=session2,  post_process_mask=True) # alpha_matting=True, alpha_matting_foreground_threshold=270,alpha_matting_background_threshold=20, alpha_matting_erode_size=11)
                
                
                
                img_array = np.array(output)
                
                

                 # Create result: white where alpha > 0, black where alpha == 0
                result = np.where(img_array[:, :, 3:4] == 0, 
                                    [0, 0, 0],           # Black
                                    [255, 255, 255])   # White
                
                white_seg = Image.fromarray(result.astype('uint8'))  
                if output.mode == 'RGBA':
                    output = output.convert('RGB')
                
                # if len(output.shape) == 3 and output.shape[2] == 4:
                #     # Image is BGRA → convert to BGR (drop alpha)
                #     output = cv2.cvtColor(output, cv2.COLOR_BGRA2BGR)
               

                ouput_parent = input_root / output_fol_name / input_path.stem 
                ouput_parent.mkdir(parents=True, exist_ok=True)
                out_file = ouput_parent / file_name
                output.save(out_file)
                # cv2.imwrite( out_file, output)
                seg_output_parent = input_root / seg_output_fol_name/ input_path.stem
                seg_output_parent.mkdir(parents=True, exist_ok=True)
                seg_output_file =  seg_output_parent / file_name
                white_seg.save(seg_output_file)
                # cv2.imwrite( seg_output_file, white_seg)
            print(f"[GPU {gpu_id}] {temp_folder} has been created!")
            
            
        except Exception as e:
            return f"Error processing {temp_folder}: {str(e)}"
        
        finally:
            # Clean up temp folder
            if temp_root.exists():
                shutil.rmtree(temp_folder)
            for file in matching_files:
                if os.path.islink(file): 
                    target_path = os.readlink(file) 
                    os.remove(target_path) 

                os.remove(file)


 
def has_leftover_background(image, threshold=0.02):
    """
    Check if image has significant semi-transparent edges or leftover background.
    Returns True if refinement is needed.
    
    Args:
        image: PIL Image with alpha channel
        threshold: fraction of pixels that are semi-transparent (0-1 range)
    """
    # Convert byte array to PIL Image if needed
    if isinstance(image, bytes):
        image = Image.open(io.BytesIO(image))
    
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image  # Already numpy array
    
    # Get alpha channel
    if img_array.shape[2] == 4:
        alpha = img_array[:, :, 3]
    else:
        return False  # No alpha channel, assume clean
    
    # Count semi-transparent pixels (not fully opaque, not fully transparent)
    semi_transparent = np.sum((alpha > 10) & (alpha < 245))
    total_pixels = alpha.size
    
    ratio = semi_transparent / total_pixels
    
    return ratio > threshold

    
def extract_organize_run_colmap_undistort_mask_frames( 
        dataset_path, mp4s_path, cam_prefix,
        num_timesteps_interv, input_folder_path, 
         cam_folder, output_fol_name, 
        seg_output_fol_name, cam_number, ext ="png"):
    """
    Python version of run_colmap_undistort_frames bash function
    
    Args:
        output_folder_path: Base output folder path
        number_timesteps: Number of timesteps to process
        cam_folder: Camera folder name
        ext: File extension (e.g., 'jpg', 'png')
    """
    id_counter = 0
    total_number_timesteps = 0
    all_time_stamps = []
    # if os.path.isfile(os.path.join(input_folder_path, "cameras.xml")):
    #     camera_matrix, distort_coeff = create_cam_matrix_and_dist_coeff(os.path.join(input_folder_path, "cameras.xml"))



    tuples_for_extract_frames = prepare_tuples_for_extract_frames(prefix=cam_prefix, mp4s_path=mp4s_path, dataset_path=dataset_path, num_timesteps_intver=num_timesteps_interv)

    # diff = num_timesteps_interv[1]- num_timesteps_interv[0]
    num_timestepsintervals = [(intv-5, intv + 55) for intv in range(num_timesteps_interv[0], num_timesteps_interv[1], 50)  ]
    
    def probe_video(cam_file_path):
        probe = ffmpeg.probe(
            cam_file_path,
            v='quiet',
            select_streams='v:0',
            show_frames=None,
            of='json'
        )
        frames_data = probe['frames']
        return frames_data


    # Multiple files
    mp4_files = [row[0] for row in tuples_for_extract_frames]

    with ThreadPoolExecutor(max_workers=44) as executor:
        futures = [executor.submit(probe_video, file) for file in mp4_files]
        frame_results = [future.result() for future in futures]
    
    for idx, it in enumerate(tuples_for_extract_frames):
        it.append(frame_results[idx])
        it.append(0)
    
    for interval in num_timestepsintervals:
    
        if  args.mp4s_path:
            # tuples_for_extract_frames = prepare_tuples_for_extract_frames(prefix=cam_prefix, mp4s_path=mp4s_path, dataset_path=dataset_path, num_timesteps_intver=interval)
            for idx, it in enumerate(tuples_for_extract_frames):
                it[-3] = interval

            # with multiprocessing.Pool(processes=1) as pool_extract_frames:
            #     results = []  
            #     for tuple in tuples_for_extract_frames:
            #         result = pool_extract_frames.apply_async(extract_frames_with_pts, tuple)  
            #         results.append(result)  
            #     results = [r.get() for r in results] 
            with ThreadPoolExecutor(max_workers=18) as executor:
                # for tuple in tuples_for_extract_frames:
                # futures = [executor.submit(probe_video, file) for file in mp4_files]
                futures = [executor.submit(extract_frames_with_pts, *args) for args in tuples_for_extract_frames]  
                results = [future.result() for future in futures]
                    



            valid_timestamps = calculate_number_valid_timesteps(
                    dataset_path=dataset_path,
                    cam_prefix=cam_prefix,
                    ext=ext,
                    cam_number=cam_number
            )
        
            
            new_time_stamps = []
            id_cnt = 0
            for ts in valid_timestamps:
                if ts not in all_time_stamps:
                    all_time_stamps.append(ts)
                    new_time_stamps.append(ts)
                    id_cnt +=1
            print(f"{len(new_time_stamps)} time stamps will processsed")        
            # # Organize images by camera
            organize_images_folder_by_cam(
                list_valid_ts=new_time_stamps,
                dataset_path=dataset_path,
                output_folder_path=input_folder_path,
                id_counter=id_counter,
                cam_folder=cam_folder,
                cam_prefix=cam_prefix,
                cam_number=cam_number,
                ext=ext
            )
            id_counter += id_cnt
            shutil.rmtree(dataset_path)
            for idx, it in enumerate(tuples_for_extract_frames):
                it[-1] = int(new_time_stamps[-1])
                    
        
        # if  total_number_timesteps  == 0:   
        #     run_colmap(input_folder_path, cam_folder, ext="png")
    
        # print("Extract Frames Results:", results) 

    

        number_timesteps = len(new_time_stamps)
            
        # Calculate processes per GPU (you can adjust this)
        processes_per_gpu = 2 # 6 processes per GPU = 18 total processes
        num_processes =  NUM_GPUS * processes_per_gpu
        print(f"Using {num_processes} processes across {NUM_GPUS} GPUs ({processes_per_gpu} processes per GPU)")
        model_name1 = "birefnet-general"
        model_name2 = "u2net_human_seg"
        # Process images in parallel using apply_async
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = []
            for id_num in range(total_number_timesteps, total_number_timesteps + number_timesteps):
                # if idx >= 20:
                #     break
                # Round-robin GPU assignment
                gpu_id = id_num % NUM_GPUS + 1
                

                # gpu_id+=1
                
                result = pool.apply_async(
                    process_timestamp,
                    (id_num, input_folder_path, cam_folder, output_fol_name, seg_output_fol_name, model_name1, model_name2, gpu_id, "png")
                )
                results.append(result)

            total_number_timesteps += number_timesteps

            # Wait for all results to complete
            results = [r.get() for r in results]
                
            
        # Print summary

        print("\n" + "="*50)
        # print("Processing complete!")
        print(f"Total images processed: {total_number_timesteps}")
        print("="*50)
        
    # Print any errors
    # errors = [r for r in results if r.startswith("Error")]
    # if errors:
    #     print("\nErrors encountered:")
    #     for error in errors:
    #         print(error)

def run_masking_command(output_folder_path, cam_folder, folder_seg, ext, number_timesteps):
    """
    Python version of run_masking_command bash function
    
    Args:
        output_folder_path: Base output folder path
        cam_folder: Camera folder name
        folder_seg: Segmentation folder name
        ext: File extension (e.g., 'jpg', 'png')
        number_timesteps: Number of timesteps to process
    """
    
    # Number of concurrent jobs for masking
    max_jobs = 9
    num_gpus = 3          # number of GPUs you have: 0,1,2  

    
    def masking_command(folder, ts, gpu_id):
        """Execute masking command for a specific folder and timestep"""
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  

        # Run YOLO/SAM2 command
        cmd = [
            "python", "data_making/generate_masked_data_yolo11_sam2.py", "images",
            "-n", "yolox-x",
            "-c", "/home/hamit/Softwares/YOLOX/model_weights/yolox_x.pth",
            "--sam_checkpoint", "/home/hamit/Softwares/YOLOX/model_weights/segment-anything/sam_vit_l_0b3195.pth",
            "--sam_model_type", "vit_l",
            "--path", f"{folder}/",
            "--conf", "0.25",
            "--nms", "0.5",
            "--tsize", "640",
            "--save_result",
            "--device", "gpu"
        ]
        
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        subprocess.run(cmd, env=env)
        
        # Optional: Uncomment if needed
        # subprocess.run(["python", "adjust_lumis.py", "--input_folder", f"{folder}/masked_undistorted_images"])
        
        print(f"Moving masked {ts}. time step files!")
        
        # Find and process masked images
        masked_images_path = Path(folder) / "masked_undistorted_images"
        if masked_images_path.exists():
            # files = sorted(masked_images_path.glob(f"*.{ext}"))
            files = sorted(masked_images_path.glob(f"*.{ext}"), key=lambda x: int(x.stem.split('_')[0]))
            
            for file in files:
                temp_fol = file.name
                # Parse folder name from filename
                array = temp_fol.split('_')
                fol = array[0]
                
                # Create output directories if they don't exist
                black_dir = Path(output_folder_path) / f"{cam_folder}_black" / fol
                seg_dir = Path(output_folder_path) / folder_seg / fol
                
                black_dir.mkdir(parents=True, exist_ok=True)
                seg_dir.mkdir(parents=True, exist_ok=True)
                
                # Format new filename
                formatted_number = f"{ts:06d}"
                new_file_name = f"{formatted_number}.{ext}"
                
                # Move files based on naming pattern
                if f"_black.{ext}" in temp_fol:
                    shutil.move(str(file), str(black_dir / new_file_name))
                elif f"_black_white.{ext}" in temp_fol:
                    shutil.move(str(file), str(seg_dir / new_file_name))
        
        return folder
    
    def process_timestep(id_val, b):
        """Process a single timestep"""
        temp_folder = Path(f"/tmp/tmp_{b}")
        gpu_id = b % num_gpus  

        # Clean and create temp folder
        if temp_folder.exists():
            shutil.rmtree(temp_folder)
        temp_folder.mkdir(parents=True, exist_ok=True)
        
        formatted_number = f"{id_val:06d}"
        file_name = f"{formatted_number}.{ext}"
        

       
        # Create symbolic links
        cam_path = Path(output_folder_path) / cam_folder
        files = cam_path.rglob(file_name)
        files = sorted(files, key=ims_index)  
        
        for a, file in enumerate(files):
            link_path = temp_folder / f"{a}.{ext}"
            link_path.symlink_to(file)
        
        time.sleep(5)
        
        # Execute masking command
        masking_command(str(temp_folder), b, gpu_id)
        
        # Cleanup temp folder
        if temp_folder.exists():
            shutil.rmtree(temp_folder)
        
        return b
    
    # Process timesteps with concurrent execution
    number_timesteps_minus_one = number_timesteps - 1
    
    with ThreadPoolExecutor(max_workers=max_jobs) as executor:
        futures = {}
        
        for b, id_val in enumerate(range(number_timesteps)):
            # Optional: Skip first 50 timesteps (uncomment if needed)
            # if id_val < 50:
            #     continue
            
            future = executor.submit(process_timestep, id_val, b)
            futures[future] = b
        
        # Wait for all tasks to complete
        for future in as_completed(futures):
            b = futures[future]
            try:
                result = future.result()
                print(f"Completed timestep {result}")
            except Exception as e:
                print(f"Error processing timestep {b}: {e}")




   

if __name__ == '__main__':
    NUM_GPUS = 1

   

    parser = argparse.ArgumentParser(description='Process and organize camera images by valid timestamps')
    parser.add_argument("--mp4s_path", default=None, type=str, help="Path to the folder where mp4s exist")

    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to the dataset containing all frames')
    parser.add_argument('--output_folder', type=str, required=True,
                        help='Path to the output folder for organized images')
    parser.add_argument('--cam_prefix', type=str, default='Cam_',
                        help='Camera prefix (default: Cam_)')
    parser.add_argument("--num_timesteps_interv", nargs=2, default=(0, 10), type=int, help="Number of timesteps to be processed")

    parser.add_argument('--cam_folder', type=str, required=True ,
                        help='Camera folder')
    parser.add_argument('--seg_folder', type=str, required=True ,
                        help='Seg. folder')
    parser.add_argument('--rembg_folder', type=str, required=True ,
                        help='Rembg. folder')
    parser.add_argument('--cam_number', type=int, required=True,
                        help='Number of cameras')
    parser.add_argument('--ext', type=str, default='png',
                        help='Image file extension (default: png)')
    
    args = parser.parse_args()

    # extract_frames_from_mp4s(
    #     dataset_path=args.dataset_path,
    #     mp4s_path=args.mp4s_path,
    #     cam_prefix=args.cam_prefix,
    #     num_timesteps_interv=args.num_timesteps_interv,
        
    # )
    
    # # Calculate valid timestamps
    # valid_timestamps = calculate_number_valid_timesteps(
    #     dataset_path=args.dataset_path,
    #     cam_prefix=args.cam_prefix,
    #     ext=args.ext,
    #     cam_number=args.cam_number
    # )
    
    # # Organize images by camera
    # organize_images_folder_by_cam(
    #     list_valid_ts=valid_timestamps,
    #     dataset_path=args.dataset_path,
    #     output_folder_path=args.output_folder,
    #     cam_folder=args.cam_folder,
    #     cam_prefix=args.cam_prefix,
    #     cam_number=args.cam_number,
    #     ext=args.ext
    # )

    # run_colmap(
    #     output_folder_path=args.output_folder, 
    #     cam_folder=args.cam_folder, 
    #     ext=args.ext
    # )

    
    number_timesteps= 20# len(valid_timestamps)
    
    # run_colmap_undistort_frames(
    #     output_folder_path=args.output_folder,
    #     number_timesteps=number_timesteps,
    #     cam_folder=args.cam_folder,
    #     ext=args.ext,

    # )

    extract_organize_run_colmap_undistort_mask_frames(
        dataset_path=args.dataset_path,
        mp4s_path=args.mp4s_path,
        cam_prefix=args.cam_prefix,
        num_timesteps_interv=args.num_timesteps_interv,
        input_folder_path=args.output_folder,
        cam_folder=args.cam_folder,
        output_fol_name=args.rembg_folder,
        seg_output_fol_name=args.seg_folder,
        cam_number=args.cam_number,
        ext=args.ext,

    )

    # run_rembg(
    #     input_path=args.
    #     output_path=args.output_folder
    # )

    # run_masking_command(
    #     output_folder_path=args.output_folder,
    #     cam_folder=args.cam_folder,
    #     folder_seg=args.seg_folder,
    #     ext=args.ext,
    #     number_timesteps=number_timesteps
    # )
    
    # print(f"Total valid timestamps: {number_timesteps}")
