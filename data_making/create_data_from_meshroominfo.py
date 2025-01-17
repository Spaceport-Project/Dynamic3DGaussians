from plyfile import PlyData, PlyElement
import argparse
import os
import numpy as np
import json


def all_items_same(lst):
  if not lst:
      return True  # An empty list is considered to have all items the same

  first_item = lst[0]
  return all(item == first_item for item in lst)

def create_npz_from_ply(ply_file):
    with open(ply_file, 'rb') as f:
        plydata = PlyData.read(f)
        vertices = plydata['vertex']

        # Mandatory: positions
        positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
        # Optional: colors
        if all([c in vertices.dtype().names for c in ['red', 'green', 'blue']]):
            colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
        else:
            colors = np.zeros_like(positions)  # Or any default value
        #x<0.7 && x>-0.5 &&  y< -0.25 &&  z > -1 && z<0
        
        
        i=0
        seg = np.ones_like(positions[:, 0])[:, None]   # Always static for now, segmentation always 1
        for k,pos in enumerate(positions):
            if pos[0] <0.7 and pos[0]>-0.3 and  pos[1] < -0.05 and  pos[2] > 0.5 and  pos[2] < 1:
            # if pos[0] < 0.7/2.58 and pos[0] > -0.5/2.58 and pos[1] < -0.0/2.58 and pos[2] > -1/2.58 and pos[2] < 0:
                seg[k] = 0
                i+=1


        pt_cld = dict()
        pt_cld = np.concatenate((positions, colors, seg), axis=1).tolist()
        # if not os.path.exists(os.path.join(args.output_path, args.dataset_name)):
        #     os.makedirs(os.path.join(args.output_path, args.dataset_name))
        np.savez(os.path.join(os.path.dirname(ply_file), 'init_pt_cld.npz'), data=pt_cld)
        print('Point cloud saved')

def get_intr_and_camera_poses(sfm_file):
    cam_pose = dict()
    c2w = []
    ks = []
    with open(sfm_file, 'r') as f:
        data=json.load(f)
        intr = data['intrinsics'][0]
        w = int(intr['width'])
        h = int(intr['height'])
        sensor_width = int(intr['sensorWidth'])
        sensor_height = int(intr['sensorHeight'])
        focal_length = float(intr['focalLength'])
        fx = max(w,h ) * focal_length / max(sensor_width, sensor_height)
        fy = fx
        cx = w/2
        cy = h/2
        k = [ [fx, 0, cx], [0, fy, cy], [0, 0, 1] ]

        for item in data['views']:
            # print (item)
            cam_pose[item['path'].split('/')[-1].split('.')[0]] = item['poseId'] 
        
        file = open("cam_poses.txt","w")
        cam_indices = list(sorted(cam_pose.keys(), key=lambda x:int(x)))
        for ind in cam_indices:
            for pose in data["poses"]:
                if pose['poseId'] ==  cam_pose[ind]:
                    mat = np.eye(4)
                    R = np.array(pose['pose']['transform']['rotation']).reshape(3,3).astype(float)
                    T = np.array( pose['pose']['transform']['center']).astype(float)
                    # T = T.reshape([3,1])
                    # T[1:] *= -1

                    # # T[1:] = -T[1:]
                    # # R = -R
                    # # R[:,0] = -R[:,0]

                    # R[1:, 0] *= -1
                    # R[0, 1:] *= -1
                    # R= R.T
                    # T = - R @ T
                    mat[:3,:3] = R
                    mat[:3, 3] = T
                    inv_mat = np.linalg.inv(mat)
                    c2w.append(inv_mat.tolist())
                    np.savetxt(file, inv_mat)

                    ks.append(k)
        
        file.close()
    return (w, h), c2w, ks 

def get_cam_images(ims_folder, k, w2c):
    """Get image path as <cam_id>/<img_file> and cam ids as list"""
    # img_idxs = list(extr.keys())
    images = []
    cam_ids = []
    k_all = []
    w2c_all = []
    cam_folders = [(folder, len(os.listdir(os.path.join(ims_folder,folder)))) for folder in os.listdir(ims_folder) if os.path.isdir(os.path.join(ims_folder, folder)) ]
    lens_cam_folders = [it[1] for it in cam_folders]
    assert all_items_same(lens_cam_folders), "Some folders  have different number of image files"
    indices_to_remove = []
    k_clone = k.copy()
    w2c_clone = w2c.copy()
    k = [x for i, x in enumerate(k_clone) if i not in indices_to_remove]
    w2c = [x for i, x in enumerate(w2c_clone) if i not in indices_to_remove]

    for file in sorted(os.listdir(os.path.join(ims_folder,cam_folders[0][0]))):
        image_list = []
        cam_list = []
        for id in range(len(k_clone)):
            
            # id =extr[idx].name.split(".")[0]
            if id in indices_to_remove:
                continue
            image_list.append(f"{id}/{file}")
            cam_list.append(int(id))
        images.append(image_list)
        cam_ids.append(cam_list)
        k_all.append(k)
        w2c_all.append(w2c)

    
    return images, cam_ids, k_all, w2c_all
def main(args):
    data = dict()
    ply_file = os.path.join(args.input_path, "meshroom_data", "point_cloud.ply")
    create_npz_from_ply(ply_file)
    return
    sfm_file = os.path.join(args.input_path,"meshroom_data", "sfm.json")
    (data['w'],  data['h']), w2c, k = get_intr_and_camera_poses(sfm_file)
    ims_folder = os.path.join(args.input_path, 'ims')

    fn_all, cam_id_all, k_all, w2c_all = get_cam_images(ims_folder, k, w2c)
    data['k'] = k_all
    data['w2c'] = w2c_all
    data['fn'] = fn_all # Add dimension as I only have 1 timestamp for now 
    data['cam_id'] = cam_id_all    

   # Save data as a json file
    with open(os.path.join(args.output_path, args.dataset_name, 'train_meta.json'), 'w') as f:
        json.dump(data, f)
        
if __name__=='__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--input_path', type=str, default='', help='Path to the input data.')
    args.add_argument('--output_path', type=str, default='data/', help='Path to the output data.')
    args.add_argument('--dataset_name', type=str, default='test', help='Dataset name.')

    args = args.parse_args()

    main(args)