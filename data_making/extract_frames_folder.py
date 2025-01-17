import argparse
from multiprocessing import Pool
import os
import subprocess
import json
import time

def extract_frames_with_pts(cam_file_path, output_path, cam_serial):
    # First, get frame information
    cmd = ['ffprobe',
           '-v', 'quiet',
           '-select_streams', 'v:0',
           '-show_frames',
           '-of', 'json',
           cam_file_path]

    result = subprocess.run(cmd, stdout=subprocess.PIPE)
    
    frames_data = json.loads(result.stdout)

    # Now extract frames and rename them with their PTS
    # output_file_path = f'{output_path}/Cam_{cam_index:03}'
    
    os.makedirs(output_path, exist_ok=True)
    subprocess.run(['ffmpeg', '-i', cam_file_path, '-vf', "select='between(n,0,350)'", 
                   '-vsync', '0', '-start_number', '0',
                   f'{output_path}/frame_%3d_{cam_serial}.png'])

    # Rename files with PTS
    # print(frames_data['frames'])
    k = 0
    for i, frame in enumerate(frames_data['frames'], 0):
        # pts_time = frame.get('pts', str(i))
        pts_time = frame["pkt_pts"]
        # if pts_time >= 1637998 and pts_time <= 1703331:

        # if pts_time >= 1749998 and pts_time <= 1823664:
            # print(frame)
        old_name = f'{output_path}/frame_{i:03}_{cam_serial}.png'
        new_name = f'{output_path}/frame_{pts_time}_{cam_serial}.png'
        print(old_name, new_name)
        subprocess.run(['mv', old_name, new_name])

# Run the function
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--input_path', type=str, required=True, help='Path to the input data.')
    args.add_argument('--output_path', type=str,  required=True, help='Path to the output data.')

    args = args.parse_args()
    input_path = args.input_path
    output_path = args.output_path
    inputs=[]
    mp4_files_full = sorted([file for file in os.listdir(input_path) if file.endswith(".mp4") and file.startswith("Cam-Full_")])
    # print(mp4_files_full)
    
    for cam_index, cam_file in enumerate(mp4_files_full):
        # if cam_index > 2:
        #     break
        cam_file_base= os.path.splitext(cam_file)[0]
        cam_serial = cam_file_base.split('_')[-1]
        cam_file_path= os.path.join(input_path, cam_file)
        inputs.append((cam_file_path, output_path, cam_serial))
    
    # print(inputs)
    if len(inputs) > 0 :
        with Pool(processes=8) as pool:  
            pool.starmap(extract_frames_with_pts, inputs)
    
    
    inputs=[]

    mp4_files =  sorted([file for file in os.listdir(input_path) if file.endswith(".mp4") and file.startswith("Cam_")])

    for cam_index, cam_file in enumerate(mp4_files):
        # if cam_index > 2:
        #     break
        cam_file_base= os.path.splitext(cam_file)[0]
        cam_serial = cam_file_base.split('_')[-1]
        cam_file_path= os.path.join(input_path, cam_file)
        inputs.append((cam_file_path, output_path, cam_serial))
    
    # print(inputs)
    if len(inputs) > 0 :
        with Pool(processes=8) as pool:  
            pool.starmap(extract_frames_with_pts, inputs)
        
    