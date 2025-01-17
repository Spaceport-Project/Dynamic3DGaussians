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
from utils import utils_data_making
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


def get_pt_cld_from_text(path):
    """Generate init_pt_cld.npz: shape (N, 7) where N is the number of points"""

    pt_cld = []

    with open(path, 'r') as f:

        for idx, line in enumerate(f.readlines()):
            # Skip first lines
            if idx < 3:
                continue
            line = line.strip().split(' ')
            
            # Convert x,y,z,r,g,b values from str to float
            for i in range(1, 7):
                line[i] = float(line[i])
            
            
            xyz = [line[1], line[2], line[3]]
            rgb = [line[4]/255.0, line[5]/255.0, line[6]/255.0]
            seg = [0.0]   # Always static for now 
            
            pt_cld.extend([xyz + rgb + seg])
            
    pt_cld = np.array(pt_cld)

    return pt_cld


def main(args):
    print('Converting COLMAP data to Dynamic 3D Gaussians data...')

    # Copy images from COLMAP folder to data. Images need to be .JPEG.
    ims_folder = os.path.join(args.colmap_path, 'ims_black')

    
    # Generate init_pt_cld.npz: shape (N, 7) where N is the number of points
    pt_cld_path = os.path.join(args.colmap_path, 'sparse', '0', 'points3D.bin')
    xyzs, rgbs, _ = utils_colmap.read_points3D_binary(pt_cld_path)
    # seg = np.ones_like(xyzs[:, 0])[:, None]   # Always static for now, segmentation always 1
    seg = np.ones_like(xyzs[:, 0])[:, None]   # Always static for now, segmentation always 1

    # for k,pos in enumerate(xyzs):
    # # #    (x < 0.05 && x> -0.77 && y >-1 && y<2.68 && z > -0.5 && z < 0.7)
    # #     if pos[0] < 0.05 and pos[0] > -0.77 and pos[1] > -1 and pos[1] < 2.68 and pos[2] < 0.7 and pos[2] > -0.5:
    # #   #   (x < 1 && x> -0.5 && y >-1.2 && y<2.7 && z > -1 && z < 0.7) for calib data
    # # (x < 0.5 && x> -0.77 && y >-1 && y<2.68 && z > -0.5 && z < 0.9) for 2024-12-04_16-58-12 data
    #     if pos[0] < 0.5 and pos[0] > -0.77 and pos[1] > -1 and pos[1] < 2.68 and pos[2] < 0.9 and pos[2] > -0.5:
    # #     # if pos[0] < 2.4 and pos[0] > -1.5 and  pos[1] < 3.5 and pos[1] > -1 and  pos[2] > 2.2 and  pos[2] < 3.5: # colmap yoga pose_1_3
    # #     # # if pos[0] < 0.5 and pos[0] > -0.5 and  pos[1] < 0.3 and pos[1] > -1 and  pos[2] > -2.7 and  pos[2] < -1: # agisoft
    # #     # # (x <1.72 && x >-0.25 && y<1.6 && z <-0.07 && z>-0.84)
    # #     # # if pos[0] < 1.72 and pos[0] > -0.25 and  pos[1] < 1.6  and  pos[2] > -0.84 and  pos[2] < -0.07: # agisoft
   
    # #     #  (x >0.9 && x< 4 && y>-1.3 && y<5 && z>1 && z<3.5) # colmap yoga pose_1_4
    #         seg[k] = 1
    #         # i+=1

    rgbs = rgbs / 255.0
    pt_cld = dict()
    pt_cld = np.concatenate((xyzs, rgbs, seg), axis=1).tolist()
    if not os.path.exists(os.path.join(args.output_path, args.dataset_name)):
        os.makedirs(os.path.join(args.output_path, args.dataset_name))
    np.savez(os.path.join(args.output_path, args.dataset_name, 'init_pt_cld.npz'), data=pt_cld)
    print('Point cloud saved')

    # Get intrinsics and extrinsics values from COLMAP
    data = dict()
    extrinsics_path = os.path.join(args.colmap_path, 'sparse', '0', 'images.bin')
    intrinsics_path = os.path.join(args.colmap_path, 'sparse', '0', 'cameras.bin')
    # intrinsics_path = os.path.join(args.colmap_path, 'sparse', '0', 'cameras.txt')

    extr = utils_colmap.read_extrinsics_binary(extrinsics_path)  # w2c
    sorted_extr = dict(sorted(extr.items(), key=lambda x:  int(x[1].name.split(".")[0])))
    keys_sorted_extr = list(sorted_extr.keys())
    intr = utils_colmap.read_intrinsics_binary(intrinsics_path)
    # intr = utils_colmap.read_intrinsics_text(intrinsics_path)
    # sorted_intr = dict(sorted(intr.items(), key=lambda x:  int(x[1].name.split(".")[0])))
    sorted_intr = OrderedDict((key, intr[key]) for key in keys_sorted_extr if key in intr)
    
    # for key in list(sorted_intr.keys()):
    #     val = sorted_intr[key]

    #     scale_x = val.width/DIM[0]
    #     scale_y = val.height/DIM[1]
    #     params =  (val.params[0]/scale_x, val.params[1]/scale_y, val.params[2]/scale_x, val.params[3]/scale_y)
    #     sorted_intr[key] = sorted_intr[key]._replace(width= DIM[0], height= DIM[1], params = params)


    data['w'] = sorted_intr[1].width
    data['h'] = sorted_intr[1].height
    


    # Generate intrinsics (N, 3, 3) where N is the number of unique cameras
    k = utils_colmap.get_intrinsics_matrix(sorted_extr, sorted_intr) 
    # data['k'] = [k] # Add dimension as I only have 1 timestamp for now 
    print('Intrinsics matrix calculated')

    # Generate extrinsics (N, 4, 4) where N is the number of unique cameras
    
    w2c = utils_colmap.get_extrinsics_matrix( sorted_extr, sorted_intr) 
    # data['w2c'] = [w2c] # Add dimension as I only have 1 timestamp for now   
    print('Extrinsics matrix calculated')

    # Get images
    fn_all, cam_id_all, k_all, w2c_all = utils_colmap.get_cam_images(sorted_extr, ims_folder,  k, w2c)
    data['k'] = k_all
    data['w2c'] = w2c_all
    data['fn'] = fn_all # Add dimension as I only have 1 timestamp for now 
    data['cam_id'] = cam_id_all
    print(repr(np.array(data['w2c'][0][0])))                   
    print(np.linalg.norm(np.linalg.inv(np.array(data['w2c'][0][0]))[:3, 3] - np.linalg.inv(np.array(data['w2c'][0][1]))[:3, 3]))
    # Save data as a json file
    with open(os.path.join(args.output_path, args.dataset_name, 'train_meta.json'), 'w') as f:
        json.dump(data, f)

if __name__=='__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--colmap_path', type=str, default='', help='Path to the COLMAP data.')
    args.add_argument('--output_path', type=str, default='data/', help='Path to the output data.')
    args.add_argument('--dataset_name', type=str, default='', help='Dataset name.')

    args = args.parse_args()

    main(args)