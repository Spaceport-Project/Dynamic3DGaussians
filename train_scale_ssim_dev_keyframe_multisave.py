import cv2
from sklearn.neighbors import NearestNeighbors
import torch
import os
import json
import copy
import numpy as np
from PIL import Image, ImageFile
import torchvision
ImageFile.LOAD_TRUNCATED_IMAGES = True

from random import randint
from tqdm import tqdm
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from helpers_scl import  fetchPly, l1_loss_v1_noblack,  save_params_single,  setup_camera, l1_loss_v1, l1_loss_v2, weighted_l2_loss_v1, weighted_l2_loss_v2, quat_mult, \
    o3d_knn, params2rendervar, params2cpu, save_params, save_params_multi
from external import  calc_ssim, calc_psnr, build_rotation, calc_ssim_with_mask, densify, inverse_sigmoid, np_inverse_sigmoid, update_params_and_optimizer
import shutil
import open3d as o3d
from fused_ssim import fused_ssim
from torch.optim.lr_scheduler import *  
from plyfile import PlyElement, PlyData 



to8b = lambda x : (255*np.clip(x.permute(1,2,0).cpu().detach().numpy(),0,1)).astype(np.uint8)
to8b_noper = lambda x : (255*np.clip(x.cpu().detach().numpy(),0,1)).astype(np.uint8)


def_pix = lambda w, h :  torch.tensor(
    np.stack(np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5, 1), -1).reshape(-1, 3)).cuda().float()
pix_ones = lambda w, h : torch.ones(h * w, 1).cuda().float()

class CustomError(Exception):
  pass
def get_dataset_noseg(t, md, seq, iters, coords, scale=1):
    dataset = []
    for c in range(len(md['fn'][t])):
        w, h, k, w2c = md['w'], md['h'], md['k'][t][c], md['w2c'][t][c]
        cam = setup_camera(w, h, k, w2c, near=1.0, far=100, scale=scale)
        fn = md['fn'][t][c]
        im_file_path = f"./data/{seq}/ims/{fn}"
        # seg_file_path = f"./data/{seq}/seg/{fn}"
        if not os.path.exists(im_file_path):
            print(f"{im_file_path} is missing, trying to find the possible image from next timestamps")
            it = iter(iters)
            while True:
                item = next(it, None)
                next_im_file_path = f"./data/{seq}/ims/{md['fn'][item][c]}"
                # next_seg_file_path = f"./data/{seq}/seg/{md['fn'][item][c]}"
                if os.path.exists(next_im_file_path):
                    shutil.copy2(next_im_file_path, im_file_path)
                    # shutil.copy2(next_seg_file_path, seg_file_path)
                    break
                if item is None:
                    raise CustomError("No succesive image found in 5 next timestamps")

        
        image = copy.deepcopy(Image.open(im_file_path))
        width, height = image.size
        resized_image = image.resize((int(width/scale), int(height/scale)))
        resized_image = np.array(resized_image)
        
        # mask = np.zeros(resized_image.shape[:2], dtype=np.uint8)

        # # Draw white rectangle on mask (region to keep)
        # mask[coords[c][1]:coords[c][3], coords[c][0]:coords[c][2]] = 255

        # # Apply mask to the image
        # masked_image = cv2.bitwise_and(resized_image, resized_image, mask=mask)
        # # cv2.imwrite("image.png", masked_image)

        resized_image = torch.tensor(resized_image).float().cuda().permute(2, 0, 1) / 255 # torch.from_numpy(np.array(resized_image)) / 255.0
        # torchvision.utils.save_image(resized_image, 'output_image.png')
        # if len(resized_image.shape) == 3:
        #     resized_image.cuda().permute(2, 0, 1)
        # else:
        #     resized_image.cuda().unsqueeze(dim=-1).permute(2, 0, 1)
        threshold = 128


        # Define the resolution  

         # Create a new image with white color  
        seg = Image.new("RGB", (width, height), "white")  
        # seg = copy.deepcopy(Image.open(seg_file_path))
     

        resized_seg =  seg.resize((int(width/scale), int(height/scale)))

        resized_seg = np.array(resized_seg)
        # masked_seg = cv2.bitwise_and(resized_seg, resized_seg, mask=mask)

        resized_seg[:, :, :] = np.where(resized_seg < threshold, 0, 255)
  
        new_img_array = np.zeros((resized_seg.shape[0], resized_seg.shape[1]), dtype=np.uint8)

        # Set pixels to 255 where the original image is white, 0 otherwise
        new_img_array[np.all(resized_seg == [255, 255, 255], axis=-1)] = 1
        # new_img_array = np.zeros((resized_image.shape[1], resized_image.shape[2]), dtype=np.uint8)
    
        new_img_array = torch.tensor(new_img_array).float().cuda()
        seg_col = torch.stack((new_img_array, torch.zeros_like(new_img_array), 1 - new_img_array))
        dataset.append({'cam': cam, 'im': resized_image, 'seg': seg_col, 'id': c})
    return dataset
def get_dataset(t, md, seq, iters, scale=1, bg=torch.tensor([0, 0, 0])):
    dataset = []
    for c in range(len(md['fn'][t])):
        w, h, k, w2c = md['w'], md['h'], md['k'][t][c], md['w2c'][t][c]
        cam = setup_camera(w, h, k, w2c, near=1.0, far=100, scale=scale, bg=bg)
        fn = md['fn'][t][c]
        im_file_path = f"./data/{seq}/ims/{fn}"
        seg_file_path = f"./data/{seq}/seg/{fn}"
        if not os.path.exists(im_file_path):
            print(f"{im_file_path} is missing, trying to find the possible image from next timestamps")
            it = iter(iters)
            while True:
                item = next(it, None)
                next_im_file_path = f"./data/{seq}/ims/{md['fn'][item][c]}"
                next_seg_file_path = f"./data/{seq}/seg/{md['fn'][item][c]}"
                if os.path.exists(next_im_file_path):
                    shutil.copy2(next_im_file_path, im_file_path)
                    shutil.copy2(next_seg_file_path, seg_file_path)
                    break
                if item is None:
                    raise CustomError("No succesive image found in 5 next timestamps")

        
        image = copy.deepcopy(Image.open(im_file_path))
        width, height = image.size
        resized_image = image.resize((int(width/scale), int(height/scale)))
        resized_image = np.array(resized_image)
        resized_image = torch.tensor(resized_image).float().cuda().permute(2, 0, 1) / 255 # torch.from_numpy(np.array(resized_image)) / 255.0
        # if len(resized_image.shape) == 3:
        #     resized_image.cuda().permute(2, 0, 1)
        # else:
        #     resized_image.cuda().unsqueeze(dim=-1).permute(2, 0, 1)
        threshold = 128


        # Define the resolution  

       
        seg = copy.deepcopy(Image.open(seg_file_path))
     

        resized_seg =  seg.resize((int(width/scale), int(height/scale)))

        resized_seg = np.array(resized_seg)
        resized_seg[:, :, :] = np.where(resized_seg < threshold, 0, 255)
        # cv2.imwrite("seg1_dataset.png",resized_seg)
        new_img_array = np.zeros((resized_seg.shape[0], resized_seg.shape[1]), dtype=np.uint8)
        # cv2.imwrite("seg1_newimg_1.png",new_img_array)
        # Set pixels to 255 where the original image is white, 0 otherwise
        new_img_array[np.all(resized_seg == [255, 255, 255], axis=-1)] = 1
        # cv2.imwrite("seg1_newimg_2.png",new_img_array*255)
        # new_img_array = np.zeros((resized_image.shape[1], resized_image.shape[2]), dtype=np.uint8)
    
        new_img_array = torch.tensor(new_img_array).float().cuda()
        seg_col = torch.stack((new_img_array, torch.zeros_like(new_img_array),  torch.zeros_like(new_img_array)))
        # torchvision.utils.save_image(seg_col, f'seg_col.png')
        dataset.append({'cam': cam, 'im': resized_image, 'seg': seg_col, 'id': c})
    return dataset


def get_batch(todo_dataset, dataset):
    if not todo_dataset:
        todo_dataset = dataset.copy()
    index_to_remove = randint(0, len(todo_dataset) - 1)
    curr_data = todo_dataset.pop(index_to_remove)
    return curr_data, index_to_remove

def initialize_params_from_prev(init_file,  md, params_prev):
    init_pt_cld = np.load(init_file)["data"]
    num_points_init = len(init_pt_cld)
    points = params_prev["means3D"].cpu().numpy()
    num_points_prev = len(points)
    colors = params_prev["rgb_colors"].cpu().numpy()
    cam_m =  params_prev["cam_m"].cpu().numpy()
    cam_c =  params_prev["cam_c"].cpu().numpy()
    pcd = o3d.geometry.PointCloud()  
    pcd.points = o3d.utility.Vector3dVector(points) 
    pcd.colors = o3d.utility.Vector3dVector(colors)
    ratio = num_points_init/num_points_prev
    if ratio >= 1.0:
        ratio=0.99
    pcd = pcd.random_down_sample(sampling_ratio=ratio) #pcd.uniform_down_sample(every_k_points=n_value)
    points_new = np.asarray(pcd.points) 
    colors_new = np.asarray(pcd.colors)
    seg = np.ones_like(points_new[:, 0]) 

  

    max_cams = 50
    sq_dist, _ = o3d_knn(points_new, 3)
    mean3_sq_dist = sq_dist.mean(-1).clip(min=0.0000001)
    params = {
        'means3D': points_new,
        'rgb_colors': colors_new,
        'seg_colors': np.stack((seg, np.zeros_like(seg), 1 - seg), -1),
        'unnorm_rotations': np.tile([1, 0, 0, 0], (seg.shape[0], 1)),
        'logit_opacities': np.zeros((seg.shape[0], 1)),
        'log_scales': np.tile(np.log(np.sqrt(mean3_sq_dist))[..., None], (1, 3)),
        'cam_m': cam_m, #np.zeros((max_cams, 3)),
        'cam_c': cam_c, #np.zeros((max_cams, 3)),
    }
    params = {k: torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True)) for k, v in
              params.items()}
    cam_centers = np.linalg.inv(md['w2c'][0])[:, :3, 3]  # Get scene radius
    scene_radius = 1.1 * np.max(np.linalg.norm(cam_centers - np.mean(cam_centers, 0)[None], axis=-1))
    variables = {'max_2D_radius': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'scene_radius': scene_radius,
                 'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'denom': torch.zeros(params['means3D'].shape[0]).cuda().float()}
    return params, variables

def initialize_optimizer_per_timestep(variables,  optimizer):
    
    lrs = {
        
        # 'logit_opacities': 0.05, #0.05,
        # 'log_scales': 0.001, #0.001,
        # 'cam_m': 1e-4,
        # 'cam_c': 1e-4,
        # 'means3D': 0.00016 * 0.1*variables['scene_radius']/2,
        # # 'rgb_colors': 0.0025,
        # 'rgb_colors': 0.0025*0.1,
        # 'seg_colors': 0.0,
        'unnorm_rotations': 0.001,
        # 'rgb_colors':0.000
      
    }

    for param_group in optimizer.param_groups:
        if param_group["name"] in lrs.keys():
            param_group['lr'] = lrs[param_group["name"]]
    
    
  
    return optimizer

def initialize_params(init_file, md):
    max_cams = 50
    if init_file.endswith("npz"):
        init_pt_cld = np.load(init_file)["data"]
        points = init_pt_cld[:, :3]
        colors = init_pt_cld[:, 3:6]
        seg = init_pt_cld[:, 6]
        sq_dist, ind = o3d_knn(points, 3)
    
   
    elif init_file.endswith("ply"):
        pcd = fetchPly(init_file)
        points = pcd.points
        colors = pcd.colors
        seg = np.ones((points.shape[0],1), dtype=np.float32)
        seg = np.squeeze(seg)
        sq_dist, ind = o3d_knn(points, 3)
        # sq_dist_2, ind_2, _ = pytorch3d_knn(torch.tensor(init_pt_cld[:, :3]).to(torch.float32), 3)
    else :
        raise CustomError("No recognized file format for initializing params")


    mean3_sq_dist = sq_dist.mean(-1).clip(min=0.0000001)


    params = {
        'means3D': points,
        'rgb_colors': colors,
        # 'rgb_colors': np_inverse_sigmoid(init_pt_cld[:, 3:6]),
        'seg_colors': np.stack((seg, np.zeros_like(seg), 1 - seg), -1),
        'unnorm_rotations': np.tile([1, 0, 0, 0], (seg.shape[0], 1)),
        'logit_opacities': np.zeros((seg.shape[0], 1)),
        'log_scales': np.tile(np.log(np.sqrt(mean3_sq_dist))[..., None], (1, 3)),
        'cam_m': np.zeros((max_cams, 3)),
        'cam_c': np.zeros((max_cams, 3)),
    }
    params = {k: torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True)) for k, v in
              params.items()}
    cam_centers = np.linalg.inv(md['w2c'][0])[:, :3, 3]  # Get scene radius
    scene_radius = 1.1 * np.max(np.linalg.norm(cam_centers - np.mean(cam_centers, 0)[None], axis=-1))
    variables = {'max_2D_radius': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'scene_radius': scene_radius,
                 'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'denom': torch.zeros(params['means3D'].shape[0]).cuda().float()}
    return params, variables


def initialize_optimizer(params, variables):
    lrs = {
        'means3D': 0.00016 * variables['scene_radius']/2,
        # 'rgb_colors': 0.0025,
        'rgb_colors': 0.0025,
        'seg_colors': 0.0,
        'unnorm_rotations': 0.001,
        'logit_opacities': 0.05, #0.05,
        'log_scales': 0.001, #0.001,
        'cam_m': 1e-4,
        'cam_c': 1e-4,
    }
    param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items()]
    return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)

def set_lrs(optimizer, variables):
    lrs = {
        'means3D': 0.0 ,# 0.00016 * variables['scene_radius']/2,
        # 'rgb_colors': 0.0025,
        'rgb_colors': 0.0025,
        'seg_colors': 0.0,
        'unnorm_rotations': 0.0, #0.001,
        'logit_opacities': 0.05, #0.05,
        'log_scales': 0.001, #0.001,
        'cam_m': 0.0 , # 1e-4,
        'cam_c': 0.0 #1e-4,
    }

    for param_group in optimizer.param_groups:
        if param_group["name"] in lrs.keys():
            param_group['lr'] = lrs[param_group["name"]]
    return optimizer

def fix_lrs(optimizer):
    

    params_to_fix = ['logit_opacities', 'log_scales', 'cam_m', 'cam_c', 'unnorm_rotations']
    # params_to_fix = ['log_scales', 'cam_m', 'cam_c']

    for param_group in optimizer.param_groups:
        if param_group["name"] in params_to_fix:
            param_group['lr'] = 0.0
    return optimizer

def remove_transition_band(img):
    # img2 = img.permute(1, 2, 0)
    lower_bound = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    upper_bound = torch.tensor([15.0/255, 15.0/255, 15.0/255], dtype=torch.float32, device="cuda")

    # Create mask for dark pixels in the transition range
    # img should be shape [H, W, 3] or [B, H, W, 3]
    mask = torch.all((img >= lower_bound) & (img <= upper_bound), dim=-1)
    result = img.clone()  
  
    # Replace transition band pixels with pure black  
    result[mask] = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    return result

def get_loss_weight(iteration, start_iter=1000, end_iter=1500, start_weight=10.0):
    if iteration < start_iter:
        return start_weight
    elif iteration >= end_iter:
        return 0.0
    else:
        # Linear interpolation
        progress = (iteration - start_iter) / (end_iter - start_iter)
        return start_weight * (1 - progress)
    
def pixel_to_3d(K, depth_map, device='cuda'):
    """
    Convert pixel coordinates to 3D world coordinates
    
    Args:
        K: Camera intrinsic matrix (3x3)
        depth_map: Depth values for each pixel (H x W)
        device: PyTorch device
    
    Returns:
        points_3d: 3D coordinates (H x W x 3)
    """
    H, W = depth_map.shape
    
    # Create pixel coordinate grids
    u, v = torch.meshgrid(torch.arange(W, device=device), 
                         torch.arange(H, device=device), 
                         indexing='xy')
    
    # Stack to get homogeneous coordinates (H x W x 3)
    # pixels_homo = torch.stack([u, v, torch.ones_like(u)], dim=-1).float()
    
    # Get camera parameters
    fx, fy = K[0][0], K[1][1]
    cx, cy = K[0][2], K[1][2]
    
    # Convert to 3D coordinates
    x = (u - cx) * depth_map / fx
    y = (v - cy) * depth_map / fy
    z = depth_map
    
    # Stack to get 3D points (H x W x 3)
    points_3d = torch.stack([x, y, z], dim=-1)
    
    return points_3d
def camera_to_world(points_3d, T_c2w):
    """
    Transform 3D points from camera to world coordinates.

    Args:
        points_3d: (H, W, 3) tensor of 3D points in camera coordinates
        T_c2w: (4, 4) camera-to-world transformation matrix

    Returns:
        points_world: (H, W, 3) tensor of 3D points in world coordinates
    """
    # H, W, _ = points_3d.shape
    # points_flat = points_3d
    # points_flat = points_3d.reshape(-1, 3)  # (N, 3)
    mask_depth = points_3d[...,2] < 12
    points_3d = points_3d[mask_depth]
    
    # Add homogeneous coordinate
    ones = torch.ones((points_3d.shape[0], 1), device=points_3d.device)
    points_homo = torch.cat([points_3d, ones], dim=1)  # (N, 4)
    
    # Transform
    points_world_homo = (T_c2w @ points_homo.T).T  # (N, 4)
    
    # Convert back to 3D
    points_world = points_world_homo[:, :3] / points_world_homo[:, 3:].clamp(min=1e-8)
    # points_world = points_world.reshape(H, W, 3)
 
    return points_world.squeeze(-1)

def keep_thick_regions(img, min_diameter_pixels=6, threshold=30):
    """
    Keep only white regions with diameter >= min_diameter_pixels
    
    Args:
        image_path: Path to your image
        min_diameter_pixels: Minimum diameter to keep (recommended: 4-8)
        threshold: Grayscale threshold for binarization
    """
    
    # Load and threshold image
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, binary_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    
    # Distance transform gives radius at each point
    dist_transform = cv2.distanceTransform(binary_img, cv2.DIST_L2, 5)
    
    # Keep only regions where radius >= min_diameter/2
    min_radius = min_diameter_pixels / 2.0
    _, thick_regions = cv2.threshold(dist_transform, min_radius, 255, cv2.THRESH_BINARY)
    
    return torch.from_numpy(thick_regions.astype(bool)).cuda()

def remove_outer(img, thickness=10):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, bin_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3,3), np.uint8)
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN,  kernel)

    # --- 2.  Get the main contour ------------------------------------------------
    cnts, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise RuntimeError("No contour found!")
    person_cnt = max(cnts, key=cv2.contourArea)

    # --- 3.  Build a mask for the contour itself --------------------------------
                               # <- your contour thickness
    cnt_mask = np.zeros_like(gray)          # single-channel mask
    cv2.drawContours(cnt_mask, [person_cnt], -1, 255, thickness)
    mask = cnt_mask == 255
    return torch.from_numpy(mask).cuda()
    # --- 4.A  Remove the contour pixels (set them to black) ----------------------
    result_bgr = img.copy()
    result_bgr[cnt_mask == 255] = (0, 0, 0)       # paint contour area black
    cv2.imwrite('no_contour_bgr.jpg', result_bgr)

    # -------- OPTIONAL 4.B  Make them transparent instead of black --------------
    b, g, r = cv2.split(img)
    alpha     = np.where(cnt_mask == 255, 0, 255).astype(np.uint8)  # 0 = transparent
    rgba      = cv2.merge([b, g, r, alpha])

def find_closest_points(pc1, pc2):
    """
    Find closest points between two point clouds
    
    Args:
        pc1: torch.Tensor of shape (N, 3) - first point cloud
        pc2: torch.Tensor of shape (M, 3) - second point cloud
    
    Returns:
        closest_indices: indices of closest points in pc2 for each point in pc1
        closest_points: actual closest points from pc2
        distances: minimum distances
    """
    # Compute pairwise distances (N x M)
    distances = torch.cdist(pc1, pc2)
    
    # Find closest points from pc2 for each point in pc1
    min_distances, closest_indices = torch.min(distances, dim=1)
    closest_points = pc2[closest_indices]
    
    return closest_indices, closest_points, min_distances

def farthest_point_sampling(points, n_samples):
    
 
    if points.dim() == 3:  
        points = points.squeeze(-1)
    N, D = points.shape  # N points, D dimensions (usually 3)
    centroids = torch.zeros(n_samples, dtype=torch.long, device=points.device)
    distance = torch.ones(N, device=points.device) * 1e10
    farthest = torch.randint(0, N, (1,), device=points.device).item()
    for i in range(n_samples):
        centroids[i] = farthest
        centroid =  points[farthest:farthest+1, :]  #points[farthest, :].unsqueeze(0)
        dist = torch.sum((points - centroid) ** 2, dim=1)
        distance = torch.minimum(distance, dist)
        farthest = torch.argmax(distance).item()
    return points[centroids]
def add_point_cloud_with_tolerance( vertices, new_points, new_indices, tolerance=1e-2):
    """Add points with floating-point tolerance"""
    # new_points = new_points.to(new_points.device, dtype=new_points.dtype)
    
    if vertices.size(0) == 0:
        vertices = new_points
        combined = torch.cat([vertices, new_indices.unsqueeze(-1)], dim=1)
        return combined
    
    # Find distances between new points and existing vertices
    distances = torch.cdist(new_points, vertices[:,:3])
    min_distances = torch.min(distances, dim=1)[0]
    
    # Keep only points that are far enough from existing ones
    mask = min_distances > tolerance
    truly_new_points = new_points[mask]
    truly_new_indices =  new_indices.unsqueeze(-1)[mask]
    combined = torch.cat([truly_new_points, truly_new_indices], dim=1)
    
    if truly_new_points.size(0) > 0:
        vertices = torch.cat([vertices, combined], dim=0)
        
    
    return vertices
def find_dense_regions(point_cloud, k=25, density_percentile=80):
    """Find dense regions in point cloud"""
    # Calculate KNN density
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(point_cloud.cpu().numpy())
    distances, _ = nbrs.kneighbors(point_cloud.cpu().numpy())
    
    # Density = inverse of mean distance to k neighbors
    mean_distances = distances[:, 1:].mean(axis=1)  # Exclude self
    density_scores = 1.0 / (mean_distances + 1e-8)
    density_scores = torch.tensor(density_scores, device=point_cloud.device)
    
    # Get dense points
    threshold = torch.quantile(density_scores, density_percentile/100.0)
    dense_mask = density_scores >= threshold
    
    return point_cloud[dense_mask], density_scores, dense_mask

def accumulate_corrupted_points(params, dataset, variables, optimizer):
    accum_point3d_world=torch.tensor([])
    rendervar = params2rendervar(params)
    for dt in dataset:
        with torch.no_grad():
            # if dt['id'] != 9: #and dt['id'] != 9 and dt['id'] != 35:
            #     continue
            im, radius, depth = Renderer(raster_settings=dt['cam'])(**rendervar)
           
            # torchvision.utils.save_image(im, f"im_{dt['id']}.png")
            # torchvision.utils.save_image(dt['im'], f"gt_im_{dt['id']}.png")

            im_per = im.permute(1,2,0).clone()
            dt_per = dt['im'].permute(1,2,0).clone()
         
            mask_thick = remove_outer(to8b_noper(im_per), thickness=10)
            im_per[mask_thick] =  torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=im_per.device)
            dt_per[mask_thick] = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=dt_per.device)
           
            # Image.fromarray((255*im_per).cpu().numpy().astype(np.uint8)).save(f"im_a_{dt['id']}.png")
            # Image.fromarray((255*dt_per).cpu().numpy().astype(np.uint8)).save(f"gt_im_a_{dt['id']}.png")
           
            diff = torch.abs(im_per - dt_per)
            mask = torch.any(diff > 0.1, dim=-1)
            # mask_expanded = mask.unsqueeze(0).expand_as(im)  # [3, H, W]
            output = torch.zeros_like(diff , dtype=torch.uint8)
            output[mask] = torch.tensor([255, 255, 255], dtype=torch.uint8, device=output.device)
            
            # img = Image.fromarray(output.cpu().numpy().astype(np.uint8))
            # img.save(f"mask_image_{dt['id']}.png")
            
            


            points_3d = pixel_to_3d(dt['cam'].k, depth[0]) 
            c2w = torch.inverse(dt['cam'].viewmatrix.transpose(1,2)).cuda().float()
            points_3d_world = camera_to_world(points_3d[mask], c2w)
            # downsampled_masked_points_3d_world = farthest_point_sampling(points_3d_world, 200)
            indices, closest_pts, dists = find_closest_points(points_3d_world, params['means3D'].detach())
            
            # xyz_s = [ tuple(v for v in ver[:3] ) for ver in closest_pts.detach().cpu().numpy()]
            # vertex_s = np.array(xyz_s, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
            # el_s = PlyElement.describe(vertex_s, 'vertex')
            # PlyData([el_s]).write(f"closest {dt['id']}.ply")

            accum_point3d_world = add_point_cloud_with_tolerance(accum_point3d_world, closest_pts, indices)
            torch.cuda.empty_cache()
  
    indices = accum_point3d_world[:, 3].long()

    unique_indices = torch.unique(indices)
    # Use these to index the original tensor
    mask_point = torch.zeros(accum_point3d_world.size(0), dtype=torch.bool)
    for idx in unique_indices:
        first_pos = (indices == idx).nonzero(as_tuple=True)[0][0]
        mask_point[first_pos] = True
    unique_tensor = accum_point3d_world[mask_point]

    indices = unique_tensor[:, 3].long()
    xyz_s = [ tuple(v for v in ver[:3] ) for ver in unique_tensor.detach().cpu().numpy()]
    vertex_s = np.array(xyz_s, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])

    el_s = PlyElement.describe(vertex_s, 'vertex')
    PlyData([el_s]).write(f"closest.ply")

    xyz =  [ tuple(v for v in ver[:3] ) for ver in params['means3D'].detach().cpu().numpy()]
    vertex = np.array(xyz, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el]).write(f"point_cloud.ply")
    torch.cuda.empty_cache()
    return indices

def update_params(indices, params, optimizer, variables):

    """
    Stop gradients for specific indices in parameters
    
    Args:
        params: dict of parameter names to tensors
        optimizer: the optimizer object
        indices: indices where gradients should be stopped
    """
    # indices = accum_point3d_world[:, 3].long()

    # unique_indices = torch.unique(indices)
    # # Use these to index the original tensor
    # mask_point = torch.zeros(accum_point3d_world.size(0), dtype=torch.bool)
    # for idx in unique_indices:
    #     first_pos = (indices == idx).nonzero(as_tuple=True)[0][0]
    #     mask_point[first_pos] = True
    # unique_tensor = accum_point3d_world[mask_point]

    # indices = unique_tensor[:, 3].long()
    
    # def mask_grad(grad):  
    #     return grad * mask.float()  #
    
    for k, v in params.items():
        # Find the parameter group
        if k  in ['cam_m', 'cam_c']:
            continue
        group = [x for x in optimizer.param_groups if x["name"] == k][0]
        old_param = group["params"][0]
        
        # Get stored optimizer state
        stored_state = optimizer.state.get(old_param, None)
        
        if stored_state is not None:
            # Create mask: True where we want to KEEP gradients
            mask = torch.zeros_like(old_param, dtype=torch.bool)
            mask[indices] = True  # False where we want to STOP gradients
            
            # Update momentum states to match the mask
            if "exp_avg" in stored_state:
                stored_state["exp_avg"] = stored_state["exp_avg"] * mask.float()
            if "exp_avg_sq" in stored_state:
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"] * mask.float()
            
            # Remove old state
            del optimizer.state[old_param]
        else:
            # Create mask even if no stored state
            mask = torch.zeros_like(old_param, dtype=torch.bool)
            mask[indices] = True
        
        
        # def zero_grad_hook(grad):  
        #     grad = grad * mask.float() # Apply mask to zero specific elements  
        #     return grad
        # old_param.register_hook(zero_grad_hook)  
        with torch.no_grad():
            old_param.grad *= mask

        # old_param.register_hook(mask_grad)  
        # Update parameter group
        group["params"][0] = old_param #new_param
        
        # Restore optimizer state for new parameter
        if stored_state is not None:
            optimizer.state[old_param] = stored_state
        
        # Update params dict
        params[k] = old_param

    with torch.no_grad():
       variables["means2D"].grad *= mask

    # if i == 800:
    lrs = {
        'means3D': 0.0,
        # 'rgb_colors': 0.0025,
        'rgb_colors': 0.0025,
        'seg_colors': 0.0,
        'unnorm_rotations': 0.0,
        'logit_opacities': 0.05, #0.05,
        'log_scales': 0.001, #0.001,
        'cam_m': 1e-4,
        'cam_c': 1e-4,
    }

    for param_group in optimizer.param_groups:
        if param_group["name"] in lrs.keys():
            param_group['lr'] = lrs[param_group["name"]]
        

    torch.cuda.empty_cache()

    return params, variables


# def update_params(accum_point3d_world, params, optimizer, variables):
#     indices = accum_point3d_world[:, 3].long()

#     for k, v in params.items():
#             # group = [x for x in optimizer.param_groups if x["name"] == k][0]
#             # stored_state = optimizer.state.get(group['params'][0], None)

#             # stored_state["exp_avg"] = torch.zeros_like(v)
#             # stored_state["exp_avg_sq"] = torch.zeros_like(v)
#             # del optimizer.state[group['params'][0]]
#             old_param = params[k].clone()

#             mask = torch.zeros_like(old_param, dtype=torch.bool) 
#             mask[indices] = 1  # False where you want to stop gradients  


#             old_param_detached = torch.where(mask, old_param, old_param.detach())  

#             params[k] = old_param_detached

#     num_pts = params['means3D'].shape[0]

#     variables['means2D_gradient_accum'] = torch.zeros(num_pts, device="cuda")
#     variables['denom'] = torch.zeros(num_pts, device="cuda")
#     variables['max_2D_radius'] = torch.zeros(num_pts, device="cuda")


#     lrs = {
#         'means3D': 0.00016 * variables['scene_radius']/2,
#         # 'rgb_colors': 0.0025,
#         'rgb_colors': 0.0025,
#         'seg_colors': 0.0,
#         'unnorm_rotations': 0.001,
#         'logit_opacities': 0.05, #0.05,
#         'log_scales': 0.001, #0.001,
#         'cam_m': 1e-4,
#         'cam_c': 1e-4,
#     }

#     for param_group in optimizer.param_groups:
#         if param_group["name"] in lrs.keys():
#             param_group['lr'] = lrs[param_group["name"]]

#     torch.cuda.empty_cache()

#     return params, variables

def get_loss(params, optimizer, curr_data, variables, is_initial_timestep, i, list_ids, vertices):
    
    ratio = 0.80
    losses = {}
    curr_id = curr_data['id']
    rendervar = params2rendervar(params)
    rendervar['means2D'].retain_grad()
    im, radius, depth = Renderer(raster_settings=curr_data['cam'])(**rendervar)
    if  i %3 == 0 and ( curr_id == 21 or  curr_id == 38):
        torchvision.utils.save_image(im, f'im1.png')
    


    im = torch.exp(params['cam_m'][curr_id])[:, None, None] * im + params['cam_c'][curr_id][:, None, None]

    ssim_value = fused_ssim(im.unsqueeze(0), curr_data['im'].unsqueeze(0))
    # ssim_value = calc_ssim(im, curr_data['im'])
    mask_im = torch.any(curr_data['im']!= 0, dim=0)
   

    losses['im'] = ratio * l1_loss_v1_noblack(im, curr_data['im'], mask_im) + (1 - ratio) * (1.0 - ssim_value)

   
    variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification
    # variables['means3D'] = rendervar['means3D']
    
    segrendervar = params2rendervar(params)
    segrendervar['colors_precomp'] = params['seg_colors']
    seg, _, _, = Renderer(raster_settings=curr_data['cam'])(**segrendervar)
    if i %5 == 0 and ( curr_id == 8 or  curr_id == 7):
        torchvision.utils.save_image(seg, f'seg1.png')

    ssim_value = fused_ssim(seg.unsqueeze(0), curr_data['seg'].unsqueeze(0))
    # mask_seg = torch.any(curr_data['seg']!= 0, dim=0)

    # ssim_value = calc_ssim(seg, curr_data['seg'])
    losses['seg'] = ratio * l1_loss_v1(seg, curr_data['seg']) + (1 - ratio) * (1.0 - ssim_value)

    

    weight = 1
    threshold_rgb = 2.5e-05# 0.00004
    penalty_weight = 10.0
    penalty = None
    loss_weights = {'im': 1.0*weight, 'seg': 3.0*weight, 'rigid': 4.0*weight, 'rot': 4.0*weight, 'iso': 2.0*weight, 'floor': 2.0, 'bg': 20.0,
            'soft_col_cons': 0.05,
            'gray_col_cons': 0.05,
            'cam_c':1,
            'cam_m':1,
          
            }


    if not is_initial_timestep:
        
      
        if i > 1000:
            # if i == 1000: 
                # optimizer = fix_lrs(optimizer)
            weight = 1
            loss_weights = {'im': 1.0*weight, 'seg': 3*weight, 'rigid': 4.0*weight, 'rot': 4.0*weight, 'iso': 2.0*weight, 'floor': 2.0, 'bg': 20.0,
                'soft_col_cons': 0.1,
                'gray_col_cons': 0.05,
                'cam_c':1,
                'cam_m':1,
            }
    
            losses['soft_col_cons'] = l1_loss_v1(torch.sigmoid(params['rgb_colors']), variables["prev_col"]) 

            # losses['cam_m'] = l1_loss_v1(params['cam_m'], variables['cam_m'])
            # losses['cam_c'] = l1_loss_v1(params['cam_c'], variables['cam_c'])
          
        else:
            is_fg = (params['seg_colors'][:, 0] > 0.5).detach()
            fg_pts = rendervar['means3D'][is_fg]
            fg_rot = rendervar['rotations'][is_fg]

            rel_rot = quat_mult(fg_rot, variables["prev_inv_rot_fg"])
            rot = build_rotation(rel_rot)
            neighbor_pts = fg_pts[variables["neighbor_indices"]]
            curr_offset = neighbor_pts - fg_pts[:, None]
            curr_offset_in_prev_coord = (rot.transpose(2, 1)[:, None] @ curr_offset[:, :, :, None]).squeeze(-1)
            losses['rigid'] = weighted_l2_loss_v2(curr_offset_in_prev_coord, variables["prev_offset"],
                                                variables["neighbor_weight"])

            losses['rot'] = weighted_l2_loss_v2(rel_rot[variables["neighbor_indices"]], rel_rot[:, None],
                                                variables["neighbor_weight"])

            curr_offset_mag = torch.sqrt((curr_offset ** 2).sum(-1) + 1e-20)
            losses['iso'] = weighted_l2_loss_v1(curr_offset_mag, variables["neighbor_dist"], variables["neighbor_weight"])

            # losses['floor'] = (torch.clamp(fg_pts[:, 1], max=2.85) - fg_pts[:,1]).mean() # torch.clamp(fg_pts[:, 1], max=2.85).mean()

            # bg_pts = rendervar['means3D'][~is_fg]
            # bg_rot = rendervar['rotations'][~is_fg]
            # losses['bg'] = l1_loss_v2(bg_pts, variables["init_bg_pts"]) + l1_loss_v2(bg_rot, variables["init_bg_rot"])

    
            losses['soft_col_cons'] = l1_loss_v1(torch.sigmoid(params['rgb_colors']), variables["prev_col"])

            # losses['gray_col_cons'] = separate_gray_loss_lab(torch.sigmoid(params['rgb_colors']), variables["prev_col"], threshold=8) 
            # penalty = torch.relu(losses['soft_col_cons'] * loss_weights['soft_col_cons'] - threshold_rgb)

            # losses['cam_m'] = l1_loss_v1(params['cam_m'], variables['cam_m'])
            # losses['cam_c'] = l1_loss_v1(params['cam_c'], variables['cam_c'])

            # loss_cam_m = l1_loss_v1(params['cam_m'], variables['cam_m'])
            # loss_cam_c = l1_loss_v1(params['cam_c'], variables['cam_c'])

            # losses['smooth_scale'] = l1_loss_v1(torch.exp(params['log_scales']), variables["prev_scales"])
            # losses['smooth_opacity'] = l1_loss_v1(torch.sigmoid(params['logit_opacities']), variables["prev_opacities"])
           
            # if i == 999:
            #     variables["prev_pts"] = params['means3D'].clone().detach()
            


   
   
    
    # if not is_initial_timesteps[1] and i >=1000:
    #     loss_weights['soft_col_cons']=2.0
    

    loss = sum([loss_weights[k] * v for k, v in losses.items()])
    # if penalty is not None:
    #     loss = loss +  penalty_weight * penalty  

    seen = radius > 0
    variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
    variables['seen'] = seen
    return loss, variables


def initialize_per_timestep(params, variables, optimizer, alpha=0.7):

    # optimizer = initialize_optimizer_per_timestep(variables, optimizer)

    pts = params['means3D']
    rgb_colors = torch.sigmoid(params['rgb_colors'])
    rot = torch.nn.functional.normalize(params['unnorm_rotations'])
    new_pts = pts + (pts - variables["prev_pts"])

    


    new_rot = torch.nn.functional.normalize(rot + (rot - variables["prev_rot"]))

    is_fg = params['seg_colors'][:, 0] > 0.5
    prev_inv_rot_fg = rot[is_fg]
    prev_inv_rot_fg[:, 1:] = -1 * prev_inv_rot_fg[:, 1:]
    fg_pts = pts[is_fg]
    
    
    # neighbor_sq_dist, neighbor_ curr_idices = o3d_knn(fg_pts.detach().cpu().numpy(), 20)
    # neighbor_weight = np.exp(-4000 * neighbor_sq_dist)
    # neighbor_dist = np.sqrt(neighbor_sq_dist)

    # variables["neighbor_indices"] = torch.tensor(neighbor_indices).cuda().long().contiguous()
    # variables["neighbor_weight"] = torch.tensor(neighbor_weight).cuda().float().contiguous()
    # variables["neighbor_dist"] = torch.tensor(neighbor_dist).cuda().float().contiguous()



    prev_offset = fg_pts[variables["neighbor_indices"]] - fg_pts[:, None]
    variables['prev_inv_rot_fg'] = prev_inv_rot_fg.detach()
    variables['prev_offset'] = prev_offset.detach()
    variables["prev_pts"] = pts.clone().detach()
    variables["prev_rot"] = rot.detach()

    variables["prev_col"] = rgb_colors.detach()

    # variables["cam_m"]= params['cam_m'].clone().detach()
    # variables["cam_c"]= params['cam_c'].clone().detach()

    num_pts = pts.shape[0]

    variables['means2D_gradient_accum'] = torch.zeros(num_pts, device="cuda")
    variables['denom'] = torch.zeros(num_pts, device="cuda")
    variables['max_2D_radius'] = torch.zeros(num_pts, device="cuda")

   
        
  
    lrs = {
        'means3D': 0.00016 * variables['scene_radius']/2,
        'rgb_colors': 0.0025,
        'seg_colors': 0.0,
        'unnorm_rotations': 0.001,
        'logit_opacities': 0.0, #0.05,
        'log_scales': 0.0, #0.001,
        'cam_m': 0.0,  #1e-4,
        'cam_c': 0.0, #1e-4,
    }

    for param_group in optimizer.param_groups:
        if param_group["name"] in lrs.keys():
            param_group['lr'] = lrs[param_group["name"]]
        



    # params_to_fix = ['logit_opacities', 'log_scales']

    # for param_group in optimizer.param_groups:
    #     if param_group["name"] in params_to_fix:
    #         param_group['lr'] = 0.0

    new_params = {'means3D': new_pts, 'unnorm_rotations': new_rot}
    params = update_params_and_optimizer(new_params, params, optimizer)

    return params, variables
def update_some_params(params, optimizer, variables):
    
    # rgb_colors = torch.sigmoid(params['rgb_colors'])
    # variables["prev_col"] = rgb_colors.detach()
    
    params_to_fix = ['means3D', 'unnorm_rotations']

    for param_group in optimizer.param_groups:
        if param_group["name"] in params_to_fix:
            param_group['lr'] = 0.0
    
    
    return params, variables

def initialize_post_first_timestep(params, variables, optimizer, num_knn=20):
    is_fg = params['seg_colors'][:, 0] > 0.5
    init_fg_pts = params['means3D'][is_fg]
    init_bg_pts = params['means3D'][~is_fg]
    rgb_colors =  torch.sigmoid(params['rgb_colors'])
    init_bg_rot = torch.nn.functional.normalize(params['unnorm_rotations'][~is_fg])
    neighbor_sq_dist, neighbor_indices = o3d_knn(init_fg_pts.detach().cpu().numpy(), num_knn)
    # _, neighbor_indices_rgb = o3d_knn(init_fg_pts.detach().cpu().numpy(), 50)
    neighbor_weight = np.exp(-4000 * neighbor_sq_dist)
    neighbor_dist = np.sqrt(neighbor_sq_dist)

    # neighbor_indices_rgb = torch.tensor(neighbor_indices_rgb).cuda().long().contiguous()
    variables["neighbor_indices"] = torch.tensor(neighbor_indices).cuda().long().contiguous()
    variables["neighbor_weight"] = torch.tensor(neighbor_weight).cuda().float().contiguous()
    variables["neighbor_dist"] = torch.tensor(neighbor_dist).cuda().float().contiguous()

    variables["init_bg_pts"] = init_bg_pts.detach()
    variables["init_bg_rot"] = init_bg_rot.detach()
    variables["prev_pts"] = params['means3D'].detach()
    variables["prev_rot"] = torch.nn.functional.normalize(params['unnorm_rotations']).detach()

   
   

    
    # new_params = {'rgb_colors': inverse_sigmoid(result_rgb)}
    # params = update_params_and_optimizer(new_params, params, optimizer)

    variables["prev_col"] = rgb_colors.detach()
    # variables["cam_m"]= params['cam_m'].clone().detach()
    # variables["cam_c"]= params['cam_c'].clone().detach()

    # variables["prev_scales"] = torch.exp(params['log_scales'].detach())
    # variables["prev_opacities"] =  torch.sigmoid(params['logit_opacities']).detach()

   
   

    params_to_fix = ['logit_opacities', 'log_scales', 'cam_m', 'cam_c']

    for param_group in optimizer.param_groups:
        if param_group["name"] in params_to_fix:
            param_group['lr'] = 0.0

    # params_clone = {k: v.clone() for k, v in params.items() }
    num_pts =  params['means3D'].shape[0]
    return variables, num_pts # params_clone


def report_progress(params, data, i, every_i=100):
    if i % every_i == 0:
        im, _, _, = Renderer(raster_settings=data['cam'])(**params2rendervar(params))
        curr_id = data['id']
        im = torch.exp(params['cam_m'][curr_id])[:, None, None] * im + params['cam_c'][curr_id][:, None, None]
        psnr = calc_psnr(im, data['im']).mean()
        # ssim_value = fused_ssim(im.unsqueeze(0), data['im'].unsqueeze(0))

        # progress_bar.set_postfix({"train img 0 PSNR": f"{psnr:.{7}f}"})
        # progress_bar.update(every_i)
        # return ssim_value.cpu().numpy().item()
        return psnr.cpu().numpy().item() #, ssim_value.cpu().numpy().item()
    else:
        return  None

def clone_cuda_dict(cuda_dict):
    return {k: v.clone() if torch.is_tensor(v) else v for k, v in cuda_dict.items() }

def clone_adam_optimizer(optimizer):
    # Get the original parameters from the optimizer
    orig_state_dict = optimizer.state_dict()

    # Get parameter groups from original optimizer
    param_groups = optimizer.param_groups

    # Create new parameter groups with cloned tensors
    new_param_groups = []
    for group in param_groups:
        new_group = {k: v if not torch.is_tensor(v) else v.clone() for k, v in group.items()}
        # Clone the parameters
        new_group['params'] = [p.clone().requires_grad_(p.requires_grad) for p in group['params']]
        new_param_groups.append(new_group)

    # Create new Adam optimizer with cloned parameters
    new_optimizer = torch.optim.Adam(new_param_groups, lr=param_groups[0]['lr'])

    # Create new state dict with cloned tensors
    new_state_dict = {
        'state': {},
        'param_groups': orig_state_dict['param_groups']
    }

    # Clone the state
    for k, v in orig_state_dict['state'].items():
        new_state_dict['state'][k] = {
            key: val.clone() if torch.is_tensor(val) else val
            for key, val in v.items()
        }

    # Load state into new optimizer
    new_optimizer.load_state_dict(new_state_dict)

    return new_optimizer

def load_initial_frame_params(params_init_file):
    params = dict(np.load(params_init_file))
    params = {k: torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True)) for k, v in
              params.items()}
    return params

def train(seq, exp, scale=1):
    
    if os.path.exists(f"./output/{exp}/{seq}"):
        print(f"Experiment '{exp}' for sequence '{seq}' already exists. Exiting.")
        return
    index_key_timestep = 0
    md = json.load(open(f"./data/{seq}/train_meta.json", 'r'))  # metadata
    num_timesteps =  len(md['fn'])
    num_cams = len(md['fn'][0])

    save_interval = 3
    initial_timestep = 0
    if num_timesteps > save_interval:
        save_iterations = [ iter  for iter in range(initial_timestep + save_interval, num_timesteps) if iter % save_interval == 0]
    else :
        save_iterations = []
    init_params_file = f"./data/{seq}/init_pt_cld.npz"
    # init_params_file = f"./data/{seq}/points3D_simplified.ply"
    params, variables = initialize_params(init_params_file, md)
    optimizer = initialize_optimizer(params, variables)
   
    output_params = []
    is_key_timestep = True
    dataset_id = 36
  
    ssim_lower_limit = 22 #0.988
    ssim_lower_limit_var = ssim_lower_limit
    ssim_lower_margin = 0.5 #0.001
    ssim_lower_margin2 = 0.5 #0.001
    ssim_init = 9999
    ssim_nonkeyfr_limit = 9999
    count_non_key_frames = 0
    every_small = 5
    every_big = 100
    key_frame_start_interval  = 20
    last_key_frame = 1000000
    every_i = every_big
   
    t = 0

    while True:
    # for t in range(num_timesteps):
        if t >= num_timesteps:
            break
        if t < initial_timestep:
            t += 1
            continue
        if t  == num_timesteps - 1:
            iters = [t-1]
        else:
            iters = [it for it in range(t+1,t+6) if it < num_timesteps]
        
        dataset = get_dataset(t, md, seq, iters, scale=scale)
        
        # dataset = get_dataset_noseg(t, md, seq, iters, coords, scale=scale )

        todo_dataset = []
        
        is_initial_key_timestep = (t == initial_timestep)
        


        is_key_timestep = is_initial_key_timestep or  is_key_timestep
        # if is_initial_key_timestep:
        #     params = load_initial_frame_params(f"./data/{seq}/params_init.npz")
        #     output_params.append(params2cpu(params, is_key_timestep))
        #     optimizer = initialize_optimizer(params, variables)
        #     optimizer.load_state_dict(torch.load(f"./data/{seq}/optimizer_checkpoint.pth"))  

        #     variables, num_pts = initialize_post_first_timestep(params, variables, optimizer)
            
        #     is_key_timestep = False
        #     first_smallest_ssim = ssim_lower_limit
           
        #     t += 1
        #     continue

        if not is_key_timestep:
          
            params, variables = initialize_per_timestep(params, variables, optimizer)
            # optimizer = fix_lrs(optimizer)
            count_non_key_frames += 1
            # index_key_timestep += 1
       
        else:
            count_non_key_frames = 0

        num_iter_per_timestep_for_key_frames = int(10000)
        limit_num_iter_per_timestep_for_key_frames = num_iter_per_timestep_for_key_frames * 3
        step_num_iter_per_timestep_for_key_frames = int(300)
        num_iter_per_timestep_for_non_key_frames = int(1500)
  
        num_iter_per_timestep = num_iter_per_timestep_for_key_frames if is_key_timestep else num_iter_per_timestep_for_non_key_frames
        progress_bar = tqdm(range(num_iter_per_timestep), desc=f"timestep {t}")
        skip_to_next_timestamp = False

        if t >= last_key_frame + key_frame_start_interval:  #and ssim_nonkeyfr_limit < ssim_lower_limit - ssim_lower_margin * 2:
            every_i = every_big
            ssim_lower_limit_var = ssim_lower_limit
            is_key_timestep = True
            with torch.no_grad():

                params, variables = initialize_params_from_prev(init_params_file, md, params)  
                optimizer = initialize_optimizer(params, variables)
                
                # optimizer = initialize_optimizer_from_prev(optimizer)

            # num_iter_per_timestep_for_key_frames=int(num_iter_per_timestep_for_key_frames/2)
            num_iter_per_timestep = num_iter_per_timestep_for_key_frames
            progress_bar = tqdm(range(num_iter_per_timestep), desc=f"timestep {t}")
            index_key_timestep += 1
            count_non_key_frames =0
            print("Key Frame Called!")
            print("ssim_nonkeyfr_limit:", ssim_nonkeyfr_limit)
            limit_num_iter_per_timestep_for_key_frames =  num_iter_per_timestep*3



        i = 0
        ssim_init_var = ssim_init
        running_loss = 0.0  
        list_ids = []
        vertices = torch.tensor([])
        while True:
            if i >= num_iter_per_timestep:
                break
          
            curr_data, index= get_batch(todo_dataset, dataset)

            try:
               

                is_key_timesteps = (is_initial_key_timestep, is_key_timestep )
                loss, variables = get_loss(params, optimizer, curr_data, variables, is_key_timestep, i, list_ids, vertices)
               
                # if i == 800  and not is_key_timestep:
                    
                #     indices = accumulate_corrupted_points(params, dataset, variables, optimizer)

            except Exception as e:
                print(f"An error occurred: {e}")
                print ("Skipping to the next iter!")
                continue

            loss.backward()
            if i == 1000  and not is_key_timestep:
                optimizer = set_lrs(optimizer, variables)

            # if not is_key_timestep and i >= 800: #and i % 100 == 0 :
            #     params, variables = update_params(indices, params, optimizer, variables)
            #     # params, variables, indices = densify_nokey_frames(indices, params, variables, optimizer, i)
            #     # if i == num_iter_per_timestep - 1:

            # if not is_key_timestep: 
            #     variables = accumulate_mean2d_gradient(variables)
            

            with torch.no_grad():
                try:
                 

                    smallest_ssim = 9999
                    ssim_values = []
                    for id in range(num_cams):
                        ssim_value = report_progress(params, dataset[id], i + 1, every_i)
                        if ssim_value is not None :
                            # print(id," psnr and ssim :", psnr, " ", ssim_value)
                           
                            ssim_values.append(ssim_value)
                        
                    if len(ssim_values) > 0:
                        ssim_values.sort()
                        # print ("ssim vals:", ssim_values)
                        smallest_ssim = ssim_values[0]
                        progress_bar.set_postfix({"train img 0 SSIM": f"{ssim_values[0]:.{7}f}, loss: {loss.cpu().numpy().item():.{7}f}"})
                        progress_bar.update(every_i)
                    else:
                        smallest_ssim = None
                
                except Exception as e:
                    print(f"An error occurred: {e}")
                    print ("Skipping to the next iter!")
                    continue   

                

                if is_key_timestep :
                    params, variables = densify(params, variables, optimizer, i)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
             
                
                if smallest_ssim is not None:
                    

                    if  is_key_timestep == False and  i == num_iter_per_timestep - 1 and smallest_ssim  < ssim_init - ssim_lower_margin :
                        every_i = every_small
                        num_iter_per_timestep += int(num_iter_per_timestep_for_non_key_frames/5)
                        if num_iter_per_timestep >= int(num_iter_per_timestep_for_non_key_frames*1.2) :
                          
                            ssim_init-= ssim_lower_margin2

                            every_i = every_small
                         
                       

                    if is_key_timestep == False and i >= num_iter_per_timestep_for_non_key_frames - 1 and i < num_iter_per_timestep_for_non_key_frames  and smallest_ssim  >= ssim_init - ssim_lower_margin:
                        every_i = every_big
                        ssim_lower_limit_var = ssim_lower_limit
                        ssim_nonkeyfr_limit = smallest_ssim
                        ssim_init = first_smallest_ssim

                        print("ssim_nonkeyfr_limit:", ssim_nonkeyfr_limit)
                        
                        break
                    if is_key_timestep == False  and i >= num_iter_per_timestep_for_non_key_frames :
                        every_i = every_big
                        ssim_lower_limit_var = ssim_lower_limit
                        ssim_nonkeyfr_limit = smallest_ssim
                        ssim_init = first_smallest_ssim

                        print("ssim_nonkeyfr_limit 1:", ssim_nonkeyfr_limit)
                        
                        break

                    if is_key_timestep == True and i == num_iter_per_timestep - 1 and num_iter_per_timestep < limit_num_iter_per_timestep_for_key_frames and smallest_ssim < ssim_lower_limit:
                        every_i = every_small
                        # if i >= num_iter_per_timestep_for_key_frames  + 199:
                        ssim_lower_limit_var -= ssim_lower_margin2
                        num_iter_per_timestep += step_num_iter_per_timestep_for_key_frames
                        progress_bar = tqdm(range(num_iter_per_timestep), desc=f"timestep {t}")
                    
                    if is_key_timestep == True and i >= num_iter_per_timestep_for_key_frames -1 and smallest_ssim >= ssim_lower_limit_var:
                        ssim_init = smallest_ssim
                        # if ssim_lower_limit_var < ssim_lower_limit - ssim_lower_margin2:
                        #     ssim_init = smallest_ssim + ssim_lower_margin
                        print("ssim_init 1:",smallest_ssim)
                        every_i = every_big
                        ssim_lower_limit_var = ssim_lower_limit
                        last_key_frame = t 
                        first_smallest_ssim=smallest_ssim

                        break
                    if is_key_timestep == True and num_iter_per_timestep >= limit_num_iter_per_timestep_for_key_frames:
                        ssim_init = smallest_ssim
                        print("ssim_init 2:",smallest_ssim)

                        every_i = every_big
                        ssim_lower_limit_var = ssim_lower_limit
                        last_key_frame = t
                        first_smallest_ssim=smallest_ssim


                        break
                   
                
                
                i += 1


       
       
        if is_key_timestep:
            variables, num_pts = initialize_post_first_timestep(params, variables, optimizer)
        # else:
        #     params, variables = keep_points(num_pts, params, variables, optimizer)

        
     
        progress_bar.close()
        output_params.append(params2cpu(params, True))

        if is_key_timestep: #and not skip_to_next_timestamp:
            # save_params_single(output_params, seq, exp)
            # torch.save(optimizer.state_dict(), f'./output/{exp}/{seq}/optimizer_checkpoint.pth') 
            is_key_timestep = False

            
        if t in save_iterations:
            # save_params(output_params, seq, exp, t)
            save_params_multi(output_params, seq, exp, index_key_timestep, t)
            if not os.path.exists(f'./output/{exp}/{seq}/init_pt_cld.npz'):
                shutil.copy2(f'./data/{seq}/init_pt_cld.npz', f'./output/{exp}/{seq}/init_pt_cld.npz')
        t += 1

    save_params(output_params, seq, exp, t)
    # save_params_multi(output_params, seq, exp, index_key_timestep, t)
    if not os.path.exists(f'./output/{exp}/{seq}/init_pt_cld.npz'):
        shutil.copy2(f'./data/{seq}/init_pt_cld.npz', f'./output/{exp}/{seq}/init_pt_cld.npz')



if __name__ == "__main__":
  
    
    

   




    

    # exp_name = "2025-08-06_16-09-24_3412x2500_combin2_3_test1"    
    # for sequence in ["2025-08-06_16-09-24_3412x2500_combin2_3"]:

    # exp_name = "2025-08-06_15-44-12_3412x2500_combin1_campose_test3"
    # for sequence in ["2025-08-06_15-44-12_3412x2500_combin1_campose"]:
    
    # exp_name = "init9"
    # for sequence in ["2025-08-06_15-44-12_3412x2500_combin1_167"]:

    # exp_name = "2025-08-06_16-09-24_3412x2500_combin2_all_test1_556"
    # for sequence in ["2025-08-06_16-09-24_3412x2500_combin2_all"]:

    # exp_name = "2025-08-06_16-02-28_3412x2500_combin2_test2"
    # for sequence in ["2025-08-06_16-02-28_3412x2500_combin2"]:

    # exp_name = "2025-08-06_16-41-05_3412x2500_combin3_1_28sc_test1_metashape_contrast11_sharp05_490_2"  
    # for sequence in ["2025-08-06_16-41-05_3412x2500_combin3_1_28sc"]:

    # exp_name = "2025-11-12_14-06-52_aliyenur_4_50-350_test1"
    # for sequence in ["2025-11-12_14-06-52_aliyenur_4_50-350"]:

    # exp_name = "2025-11-12_14-37-30_aliyenur_ahmet_1_50-150_test1_60"
    # for sequence in ["2025-11-12_14-37-30_aliyenur_ahmet_1_50-150"]:

    exp_name = "2025-11-12_14-37-30_aliyenur_ahmet_1_150-550_test1_keyframe_2"
    for sequence in ["2025-11-12_14-37-30_aliyenur_ahmet_1_150-550"]:




        train(sequence, exp_name, scale=1)
        torch.cuda.empty_cache()

