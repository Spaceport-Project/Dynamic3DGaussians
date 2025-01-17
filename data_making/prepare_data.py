import argparse
import os,sys
from typing import List

import cv2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from glob import glob
import shutil
import numpy as np
# from utils import utils_colmap
import json
from PIL import Image as PIL_Image
from utils import utils_data_making
from collections import OrderedDict
import open3d as o3d




def load_cameras_params(calib_cameras_data_path):
    """Load camera parameters from a JSON file."""
    fs = cv2.FileStorage(str(calib_cameras_data_path), cv2.FILE_STORAGE_READ)
    num_cameras =  int(fs.getNode("nb_camera").real())
    camera_matrix_list = []
    w2c_list = []
    # w2c_0 = np.array([   [ 0.25681035,  0.0670684,  -0.96413188, 0.65729515 ],
    #     [-0.38885296,  0.92045067, -0.03954678, -0.9941085],
    #     [ 0.88478349,  0.38506155,  0.262461, 4.50231357 ],
    #     [0,0,0,1]
    # ] )

    w2c_0 = np.array([[ 0.93674253, -0.03888465,  0.34785257, -0.20542087],
       [ 0.16813808,  0.92162764, -0.34976003, -0.70957294],
       [-0.30699025,  0.38612236,  0.8698658 ,  3.68601693],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
    c2w_0 = np.linalg.inv(w2c_0)
    for cam_idx in range(num_cameras):
        cam_name = f"camera_{cam_idx}"
        camera_matrix = fs.getNode(cam_name).getNode("camera_matrix").mat()
        # with np.nditer(camera_matrix, op_flags=['readwrite']) as it:
        #     for x in it:
        #         if x != 0 and x !=1 :
        #             x *= 2 
        # camera_matrix_ = [it if (it == 1 or it == 0) else it/2  for it in camera_matrix ]
        camera_matrix_list.append(camera_matrix.tolist())
        camera_pose_matrix = fs.getNode(cam_name).getNode("camera_pose_matrix").mat()
        # camera_pose_matrix[:3,3] *= 17.429/1000
        scale=3.3299749231959184/0.2057002568030526
        camera_pose_matrix[:3,3] *= scale/1000
        camera_pose_matrix = c2w_0 @ camera_pose_matrix
        w2c = np.linalg.inv(camera_pose_matrix)
        # w2c[:3,3] *= 15.86/1000
        w2c_list.append(w2c.tolist())
    return camera_matrix_list, w2c_list

def all_items_same(lst):
  if not lst:
      return True  # An empty list is considered to have all items the same

  first_item = lst[0]
  return all(item == first_item for item in lst)

def get_cam_images( ims_folder, k, w2c):
    """Get image path as <cam_id>/<img_file> and cam ids as list"""
    imgs_len = len(k)
    images = []
    cam_ids = []
    k_all = []
    w2c_all = []
    
    # cam_ims_folders_len = [ len([ file for file in os.listdir(os.path.join(ims_folder,folder)) if file.endswith('.png')]) for folder in os.listdir(ims_folder) if os.path.isdir(os.path.join(ims_folder, folder)) ]
    cam_ims_folders = [(folder, len([ file for file in os.listdir(os.path.join(ims_folder,folder)) if file.endswith('.png')])) for folder in os.listdir(ims_folder) if os.path.isdir(os.path.join(ims_folder, folder)) ]
    cam_ims_folders = sorted(cam_ims_folders, key=lambda x: x[1], reverse=True)

    last_file = sorted(os.listdir(os.path.join(ims_folder, cam_ims_folders[0][0])))[-1]
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
                    # seg_file_prev_path = im_file_prev_path.replace("/ims/","/seg/")
                    # seg_file_path = im_file_path.replace("/ims/","/seg/")
                    # shutil.copy2(seg_file_prev_path, seg_file_path)

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
                        # next_seg_file_path = next_im_file_path.replace("/ims/","/seg/")
                        # im_file_path =  os.path.join(ims_folder,folder,file)
                        # seg_file_path = im_file_path.replace("/ims/","/seg/")
                        # shutil.copy2(next_im_file_path, im_file_path)
                        # shutil.copy2(next_seg_file_path, seg_file_path)
                        # break

                        two_up_folder=os.path.basename(os.path.dirname(os.path.dirname(next_im_file_path)))
                        next_seg_file_path = next_im_file_path.replace(f"/{two_up_folder}/","/seg/")
                        im_file_path =  os.path.join(ims_folder,folder,file)
                        seg_file_path = im_file_path.replace(f"/{two_up_folder}/","/seg/")
                        shutil.copy2(next_im_file_path, im_file_path)
                        shutil.copy2(next_seg_file_path, seg_file_path)
                        
                    if item is None:
                        print("No succesive image found in 5 next timestamps")
                        return

        




def main(args):
    print('Preparing data for Dynamic 3D Gaussians...')

    # Copy images from COLMAP folder to data. Images need to be .JPEG.
    ims_folder = os.path.join(args.input_path, 'ims_black')

    
    # Generate init_pt_cld.npz: shape (N, 7) where N is the number of points
    # pt_cld_path = os.path.join(args.colmap_path, 'sparse', '0', 'points3D.bin')
    pt_cld_path = os.path.join(args.input_path,  'points3D.ply')
    pt_cld = o3d.io.read_point_cloud(pt_cld_path)
    # pt_cld.scale(1/15.85, center=(0,0,0))
    # xyzs, rgbs, _ = utils_colmap.read_points3D_binary(pt_cld_path)
    xyzs, rgbs = np.asarray(pt_cld.points), np.asarray(pt_cld.colors)
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

    # rgbs = rgbs / 255.0
    pt_cld = dict()
    pt_cld = np.concatenate((xyzs, rgbs, seg), axis=1).tolist()
    if not os.path.exists(os.path.join(args.output_path, args.dataset_name)):
        os.makedirs(os.path.join(args.output_path, args.dataset_name))
    np.savez(os.path.join(args.output_path, args.dataset_name, 'init_pt_cld.npz'), data=pt_cld)
    print('Point cloud saved')

    # Get intrinsics and extrinsics values from COLMAP
    data = dict()
   
    data['w'] = 4096 # sorted_intr[1].width
    data['h'] = 3000 # sorted_intr[1].height
    


    # Generate intrinsics (N, 3, 3) where N is the number of unique cameras
    k, w2c = load_cameras_params(args.camera_file_path) 


    # Get images
    fn_all, cam_id_all, k_all, w2c_all = get_cam_images( ims_folder,  k, w2c)
    data['k'] = k_all
    data['w2c'] = w2c_all
    data['fn'] = fn_all # Add dimension as I only have 1 timestamp for now 
    data['cam_id'] = cam_id_all                   

    # Save data as a json file
    with open(os.path.join(args.output_path, args.dataset_name, 'train_meta.json'), 'w') as f:
        json.dump(data, f)

if __name__=='__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--input_path', type=str, required=True, help='Path to the data.')
    args.add_argument('--camera_file_path',  type=str, required=True, help='Path to the camera params file.')
    args.add_argument('--output_path', type=str, default='data/', help='Path to the output data.')
    args.add_argument('--dataset_name', type=str, default='', help='Dataset name.')

    args = args.parse_args()

    main(args)