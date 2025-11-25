"""This script converts data extracted from COLMAP to the required data for Dynamic 3D Gaussians."""
import argparse
import os,sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from glob import glob
import shutil
import numpy as np
from utils import utils_colmap
import json
from PIL import Image as PIL_Image
# from utils import utils_data_making
from collections import OrderedDict

DIM=(4220,3060)

def get_intrinsics_from_txt(path):
    """Convert the file `cameras.txt` extracted from colmap (SIMPLE_PINHOLE) to camera intrinsics."""
    ks = []
    with open(path, 'r') as f:
        for idx, line in enumerate(f.readlines()):
            # Skip first lines
            if idx < 3:
                continue
            line = line.strip().split(' ')

            # Convert x,y,z,r,g,b values from str to float
            for i in range(2, 7):
                line[i] = float(line[i])

            w = line[2]
            h = line[3]
            fx = line[4]
            fy = line[4]
            cx = line[5]
            cy = line[6]

            k = [ [fx, 0, cx], [0, fy, cy], [0, 0, 1] ]
            ks.append(k)
    
    return ks




def main(args):

   
    extrinsics_path = os.path.join(args.colmap_path,  'images.bin')
    intrinsics_path = os.path.join(args.colmap_path,  'cameras.bin')
    # intrinsics_path = os.path.join(args.colmap_path, 'colmap_input', 'sparse', '0', 'cameras.txt')

    extr = utils_colmap.read_extrinsics_binary(extrinsics_path)  # w2c
    sorted_extr = dict(sorted(extr.items(), key=lambda x:  int(x[1].name.split(".")[0])))
    keys_sorted_extr = list(sorted_extr.keys())
    intr = utils_colmap.read_intrinsics_binary(intrinsics_path)
    # intr = utils_colmap.read_intrinsics_text(intrinsics_path)
    # sorted_intr = dict(sorted(intr.items(), key=lambda x:  int(x[1].name.split(".")[0])))
    sorted_intr = OrderedDict((key, intr[key]) for key in keys_sorted_extr if key in intr)
    
    


    # # Generate intrinsics (N, 3, 3) where N is the number of unique cameras
    # k = utils_colmap.get_intrinsics_matrix(sorted_extr, sorted_intr) 
    # # data['k'] = [k] # Add dimension as I only have 1 timestamp for now 
    # print('Intrinsics matrix calculated')

    # Generate extrinsics (N, 4, 4) where N is the number of unique cameras
    
    w2c = utils_colmap.get_extrinsics_matrix( sorted_extr, sorted_intr) 
    # data['w2c'] = [w2c] # Add dimension as I only have 1 timestamp for now   
    # print('Extrinsics matrix calculated')

    # # Get images
    # fn_all, cam_id_all, k_all, w2c_all = utils_colmap.get_cam_images(sorted_extr, ims_folder,  k, w2c)
    # data['k'] = k_all
    # data['w2c'] = w2c_all
    # data['fn'] = fn_all # Add dimension as I only have 1 timestamp for now 
    # data['cam_id'] = cam_id_all
    # print(repr(np.array(data['w2c'][0][0])))                   
    # print(np.linalg.norm(np.linalg.inv(np.array(data['w2c'][0][0]))[:3, 3] - np.linalg.inv(np.array(data['w2c'][0][1]))[:3, 3]))
    # # Save data as a json file
    # with open(os.path.join(args.output_path, args.dataset_name, 'train_meta.json'), 'w') as f:
    #     json.dump(data, f)

if __name__=='__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--colmap_path', type=str, default='', help='Path to the COLMAP data.')
    # args.add_argument('--output_path', type=str, default='data/', help='Path to the output data.')
    # args.add_argument('--dataset_name', type=str, default='', help='Dataset name.')

    args = args.parse_args()

    main(args)