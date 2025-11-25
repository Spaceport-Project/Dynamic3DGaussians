import cv2
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
from helpers_scl import combined_color_loss_v2,combined_color_loss_v1, convert_rgb_to_hsv, l2_loss_v1, l2_loss_v2,  pytorch3d_knn, separate_gray_loss, separate_gray_loss_hsv, separate_gray_loss_hsv_smooth, separate_gray_loss_lab, separate_gray_loss_lab_smooth, setup_camera, l1_loss_v1, l1_loss_v2, weighted_l2_loss_v1, weighted_l2_loss_v2, quat_mult, \
    o3d_knn, params2rendervar, params2cpu, save_params, save_params_multi
from external import calc_ssim, calc_psnr, build_rotation, densify, inverse_sigmoid, np_inverse_sigmoid, update_params_and_optimizer
import shutil
import open3d as o3d
from fused_ssim import fused_ssim
from torch.optim.lr_scheduler import *  



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

def initialize_params_from_prev_old(init_file,  md, params_prev):
    init_pt_cld = np.load(init_file)["data"]
    num_points_init = len(init_pt_cld)
    points = params_prev["means3D"].cpu().numpy()
    num_points_prev = len(points)
    colors = params_prev["rgb_colors"].cpu().numpy()
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
    init_pt_cld = np.load(init_file)["data"]
    seg = init_pt_cld[:, 6]
    max_cams = 50
    sq_dist, ind = o3d_knn(init_pt_cld[:, :3], 3)
    sq_dist_2, ind_2, _ = pytorch3d_knn(torch.tensor(init_pt_cld[:, :3]).to(torch.float32), 3)
    mean3_sq_dist = sq_dist.mean(-1).clip(min=0.0000001)
    params = {
        'means3D': init_pt_cld[:, :3],
        # 'rgb_colors': init_pt_cld[:, 3:6],
        'rgb_colors': np_inverse_sigmoid(init_pt_cld[:, 3:6]),
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

def initialize_lrs(optimizer, variables):
    lrs = {
        # 'means3D': 0.00016 * variables['scene_radius']/2,
        # # 'rgb_colors': 0.0025,
        # 'rgb_colors': 0.0025,
        # 'seg_colors': 0.0,
        # 'unnorm_rotations': 0.001,
        'logit_opacities': 0.05, #0.05,
        'log_scales': 0.001, #0.001,
        'cam_m': 1e-4,
        'cam_c': 1e-4,
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

   

def get_loss_weight(iteration, start_iter=1000, end_iter=1500, start_weight=10.0):
    if iteration < start_iter:
        return start_weight
    elif iteration >= end_iter:
        return 0.0
    else:
        # Linear interpolation
        progress = (iteration - start_iter) / (end_iter - start_iter)
        return start_weight * (1 - progress)
def get_loss(params, optimizer, curr_data, variables, is_initial_timesteps, i, ind):
    
    ratio = 0.80
    losses = {}
    
    rendervar = params2rendervar(params)
    rendervar['means2D'].retain_grad()
    im, radius, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)
    if  i %3 == 0 and (ind == 8 or ind == 7):
        torchvision.utils.save_image(im, f'im1.png')
    curr_id = curr_data['id']


    im = torch.exp(params['cam_m'][curr_id])[:, None, None] * im + params['cam_c'][curr_id][:, None, None]

    ssim_value = fused_ssim(im.unsqueeze(0), curr_data['im'].unsqueeze(0))
    # ssim_value = calc_ssim(im, curr_data['im'])
    losses['im'] = ratio * l1_loss_v1(im, curr_data['im']) + (1 - ratio) * (1.0 - ssim_value)

    # current_psnr = calc_psnr(im, curr_data['im']).mean()
    # psnr_loss_exp = torch.exp(-current_psnr / 30.0)
    # # sim = calc_ssim(im, curr_data['im'])
    # psnr_loss = torch.clamp((target_psnr - current_psnr) / (target_psnr - min_psnr), 0.0, 1.0)
    # losses['im'] = ratio * l1_loss_v1(im, curr_data['im']) + (1 - ratio) * (psnr_loss_exp)

    variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification

    segrendervar = params2rendervar(params)
    segrendervar['colors_precomp'] = params['seg_colors']
    seg, _, _, = Renderer(raster_settings=curr_data['cam'])(**segrendervar)
    if i %5 == 0 and (ind == 8 or ind == 7):
        torchvision.utils.save_image(seg, f'seg1.png')

    ssim_value = fused_ssim(seg.unsqueeze(0), curr_data['seg'].unsqueeze(0))
    # ssim_value = calc_ssim(seg, curr_data['seg'])
    losses['seg'] = ratio * l1_loss_v1(seg, curr_data['seg']) + (1 - ratio) * (1.0 - ssim_value)

    # current_psnr = calc_psnr(seg, curr_data['seg']).mean()
    # psnr_loss_exp = torch.exp(-current_psnr / 30.0)
    # psnr_loss = torch.clamp((target_psnr - current_psnr) / (target_psnr - min_psnr), 0.0, 1.0)


    # losses['seg'] = ratio * l1_loss_v1(seg, curr_data['seg']) + (1 - ratio) * (psnr_loss_exp)

    # if is_initial_timesteps[1] and not is_initial_timesteps[0]:
    #     losses['cam_m'] = l1_loss_v1(params['cam_m'], variables['cam_m'])
        # losses['cam_c'] = l1_loss_v1(params['cam_m'], variables['cam_c'])

    # if not is_initial_timesteps[0] and is_initial_timesteps[1]:
    #     losses['soft_col_cons'] = l1_loss_v1(torch.sigmoid(params['rgb_colors']), variables["prev_col"]) 
    #     # losses['cam_m'] = l1_loss_v1(params['cam_m'], variables['cam_m'])
    #     # losses['cam_c'] = l1_loss_v1(params['cam_m'], variables['cam_c'])

    #     losses['smooth_scale'] = l1_loss_v1(torch.exp(params['log_scales']), variables["prev_scales"])
    #     losses['smooth_opacity'] = l1_loss_v1(torch.sigmoid(params['logit_opacities']), variables["prev_opacities"])

    weight = 1
    threshold_rgb = 2.5e-05# 0.00004
    penalty_weight = 10.0
    penalty = None
    loss_weights = {'im': 1.0, 'seg': 3.0, 'rigid': 4.0*weight, 'rot': 4.0*weight, 'iso': 2.0*weight, 'floor': 2.0, 'bg': 20.0,
            'soft_col_cons': 0.5,
            'gray_col_cons': 0.5,
            'cam_c':1,
            'cam_m':1,
            'smooth_scale': 1,
            'smooth_opacity': 0.5,
          
            
            }

    if not is_initial_timesteps[1]:
        
        # weight = get_loss_weight(i, start_iter=1000, end_iter=2000, start_weight=5)  
        
        if i >= 100000:
            # if i == 1000: 
                # optimizer = fix_lrs(optimizer)
            losses['soft_col_cons'] = l1_loss_v1(torch.sigmoid(params['rgb_colors']), variables["prev_col"]) 
            losses['means3D'] =  l1_loss_v1((params['means3D']), variables['prev_pts'])
            # losses['smooth_scale'] = l1_loss_v1(torch.exp(params['log_scales']), variables["prev_scales"])
            # losses['smooth_opacity'] = l1_loss_v1(torch.sigmoid(params['logit_opacities']), variables["prev_opacities"])
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

    
            # losses['soft_col_cons'] = l1_loss_v1(torch.sigmoid(params['rgb_colors']), variables["prev_col"])

            # losses['gray_col_cons'] = separate_gray_loss_lab(torch.sigmoid(params['rgb_colors']), variables["prev_col"], threshold=8) 
            # penalty = torch.relu(losses['soft_col_cons'] * loss_weights['soft_col_cons'] - threshold_rgb)

            losses['cam_m'] = l1_loss_v1(params['cam_m'], variables['cam_m'])
            losses['cam_c'] = l1_loss_v1(params['cam_c'], variables['cam_c'])

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
    
    
    # neighbor_sq_dist, neighbor_indices = o3d_knn(fg_pts.detach().cpu().numpy(), 20)
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

    variables["cam_m"]= params['cam_m'].clone().detach()
    variables["cam_c"]= params['cam_c'].clone().detach()

    # variables["prev_scales"] = torch.exp(params['log_scales'].detach())
    # variables["prev_opacities"] =  torch.sigmoid(params['logit_opacities']).detach()

    # hsv = convert_rgb_to_hsv(rgb_colors)
    # saturation = hsv[:, 1, :, :].squeeze()
    # gray_mask = (saturation < 0.1) & (saturation > 0.01)
    # result_rgb = rgb_colors.clone()
    # if gray_mask.any(): 
    #     gray_neighbor_indices =  variables["neighbor_indices"][gray_mask]  # Shape: [num_gray_points, k]  
          
    #     # Get RGB values of neighbors for gray points  
    #     neighbor_rgb = rgb_colors[gray_neighbor_indices]  # Shape:  
    #     smoothed_gray_rgb = torch.mean(neighbor_rgb, dim=1)  
    #     result_rgb[gray_mask] = smoothed_gray_rgb  

    # if t >= 1000:
    #     variables["prev_col"] = rgb_colors.detach()
    #     if t == 1000:
    #         params_to_fix = ['means3D', 'seg_colors', 'unnorm_rotations']

    #         for param_group in optimizer.param_groups:
    #             if param_group["name"] in params_to_fix:
    #                 param_group['lr'] = 0.0
       
        
    #     return params, variables
    # lrs = {
    #     'means3D': 0.00016 * variables['scene_radius']/2,
    #     'unnorm_rotations': 0.001,
    #     'cam_m': 1e-4,
    #     'cam_c': 1e-4,
      
    # }

    # for param_group in optimizer.param_groups:
    #     if param_group["name"] in lrs.keys():
    #         param_group['lr'] = lrs[param_group["name"]]
  

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

   
   
    # hsv = convert_rgb_to_hsv(rgb_colors)
    # saturation = hsv[:, 1, :, :].squeeze()
    # gray_mask = (saturation < 0.1) & (saturation > 0.01)
    # result_rgb = rgb_colors.clone()
    # if gray_mask.any(): 
    #     gray_neighbor_indices = neighbor_indices_rgb[gray_mask]  # Shape: [num_gray_points, k]  
          
    #     # Get RGB values of neighbors for gray points  
    #     neighbor_rgb = rgb_colors[gray_neighbor_indices]  # Shape:  
    #     smoothed_gray_rgb = torch.mean(neighbor_rgb, dim=1)  
    #     result_rgb[gray_mask] = smoothed_gray_rgb  

    
    # new_params = {'rgb_colors': inverse_sigmoid(result_rgb)}
    # params = update_params_and_optimizer(new_params, params, optimizer)

    variables["prev_col"] = rgb_colors.detach()
    variables["cam_m"]= params['cam_m'].clone().detach()
    variables["cam_c"]= params['cam_c'].clone().detach()
    # variables["prev_scales"] = torch.exp(params['log_scales'].detach())
    # variables["prev_opacities"] =  torch.sigmoid(params['logit_opacities']).detach()

   
   

    params_to_fix = ['logit_opacities', 'log_scales']

    for param_group in optimizer.param_groups:
        if param_group["name"] in params_to_fix:
            param_group['lr'] = 0.0


    return variables


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
           
def train(seq, exp, scale=1):
    
    if os.path.exists(f"./output/{exp}/{seq}"):
        print(f"Experiment '{exp}' for sequence '{seq}' already exists. Exiting.")
        return
    index_key_timestep = 0
    md = json.load(open(f"./data/{seq}/train_meta.json", 'r'))  # metadata
    num_timesteps = 1058 # len(md['fn'])
    num_cams = len(md['fn'][0])

    save_interval = 3
    initial_timestep = 20
    if num_timesteps > save_interval:
        save_iterations = [ iter  for iter in range(initial_timestep + save_interval, num_timesteps) if iter % save_interval == 0]
    else :
        save_iterations = []
    init_params_file = f"./data/{seq}/init_pt_cld.npz"
    params, variables = initialize_params(init_params_file, md)
    optimizer = initialize_optimizer(params, variables)
    # scheduler1 = ConstantLR(optimizer, factor=1.0, total_iters=50)
    # # scheduler2 = StepLR(optimizer, step_size=30, gamma=0.1)
    # # scheduler2 = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10, threshold= 1e-6, verbose=True)
    # scheduler2 =ExponentialLR(optimizer, gamma=0.999) 
    # # scheduler = SequentialLR(optimizer, 
    # #                     schedulers=[scheduler1, scheduler2], 
    # #                     milestones=[1000])
    output_params = []
    is_key_timestep = True
    dataset_id = 36
  
    ssim_lower_limit = 34 #0.988
    ssim_lower_limit_var = ssim_lower_limit
    ssim_lower_margin = 0.5 #0.001
    ssim_lower_margin2 = 0.5 #0.001
    ssim_init = 9999
    ssim_nonkeyfr_limit = 9999
    count_non_key_frames = 0
    every_small = 5
    every_big = 100
    key_frame_start_interval  = 30
    last_key_frame = 1000000
    every_i = every_big
    # with open('coordinates.txt', 'r') as file:
    # Read lines and split each line into numbers
    # coords = np.loadtxt('coordinates.txt', delimiter=',', dtype=np.int32)
        # coords = [np.array(map(float, line.strip().split(','))) for line in file]
    
    # coords = np.array(coords)  

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
        if not is_key_timestep:
            params, variables = initialize_per_timestep(params, variables, optimizer)
            # optimizer = fix_lrs(optimizer)
            count_non_key_frames += 1
        else:
            count_non_key_frames = 0

        num_iter_per_timestep_for_key_frames = int(10000)
        limit_num_iter_per_timestep_for_key_frames = num_iter_per_timestep_for_key_frames * 3
        step_num_iter_per_timestep_for_key_frames = int(300)
        num_iter_per_timestep_for_non_key_frames = int(1000)
  
        num_iter_per_timestep = num_iter_per_timestep_for_key_frames if is_key_timestep else num_iter_per_timestep_for_non_key_frames
        progress_bar = tqdm(range(num_iter_per_timestep), desc=f"timestep {t}")
        skip_to_next_timestamp = False

        # if t >= last_key_frame + key_frame_start_interval:  #and ssim_nonkeyfr_limit < ssim_lower_limit - ssim_lower_margin * 2:
        #     every_i = every_big
        #     ssim_lower_limit_var = ssim_lower_limit
        #     is_key_timestep = True
        #     with torch.no_grad():
        #         optimizer = initialize_optimizer_from_prev(optimizer)

        #         # optimizer = initialize_optimizer(params, variables)
        #     num_iter_per_timestep_for_key_frames=int(num_iter_per_timestep_for_key_frames/2)
        #     num_iter_per_timestep = num_iter_per_timestep_for_key_frames
        #     progress_bar = tqdm(range(num_iter_per_timestep), desc=f"timestep {t}")
        #     index_key_timestep += 1
        #     count_non_key_frames =0
        #     print("Key Frame Called!")
        #     print("ssim_nonkeyfr_limit:",ssim_nonkeyfr_limit)
        #     limit_num_iter_per_timestep_for_key_frames =  num_iter_per_timestep*3



        i = 0
        ssim_init_var = ssim_init
        running_loss = 0.0  
        while True:
            if i >= num_iter_per_timestep:
                break
            # if not is_key_timestep and i == 1000:
            #     params, variables = update_some_params(params, optimizer, variables)
        # for i in range(num_iter_per_timestep):
            curr_data, index= get_batch(todo_dataset, dataset)
            try:
                # if not is_key_timestep and i == 1000:
                #     optimizer = initialize_lrs(optimizer, variables)

                is_key_timesteps = (is_initial_key_timestep, is_key_timestep )
                loss, variables = get_loss(params, optimizer, curr_data, variables, is_key_timesteps, i, index )
            except Exception as e:
                print(f"An error occurred: {e}")
                print ("Skipping to the next iter!")
                continue
            loss.backward()
            with torch.no_grad():
                try:
                    # smallest_ssim = 9999
                    # for id in range(num_cams):
                    #     psnr = report_progress(params, dataset[id], i + 1, every_i)
                    #     if psnr is not None :
                    #         print(id," psnr and ssim :", psnr)
                    #         if psnr < smallest_ssim:
                    #             smallest_ssim = psnr
                    #     else:
                    #         smallest_ssim = None
                    # if smallest_ssim is not None:
                    #     progress_bar.set_postfix({"train img 0 PSNR": f"{smallest_ssim:.{7}f}"})
                    #     progress_bar.update(every_i)
                    # if  is_key_timestep and i == num_iter_per_timestep - 1:
                    #     psnr_init = smallest_ssim
                    #     print("smallest psnr init:", smallest_ssim)

                    smallest_ssim = 9999
                    ssim_values = []
                    for id in range(num_cams):
                        ssim_value = report_progress(params, dataset[id], i + 1, every_i)
                        if ssim_value is not None :
                            # print(id," psnr and ssim :", psnr, " ", ssim_value)
                            # if ssim_value < smallest_ssim:
                            #     smallest_ssim = ssim_value
                            ssim_values.append(ssim_value)
                        
                    if len(ssim_values) > 0:
                        ssim_values.sort()
                        # print ("ssim vals:", ssim_values)
                        smallest_ssim = ssim_values[1]
                        progress_bar.set_postfix({"train img 0 SSIM": f"{ssim_values[1]:.{7}f}, loss: {loss.cpu().numpy().item():.{7}f}"})
                        progress_bar.update(every_i)
                    else:
                        smallest_ssim = None
                    # if smallest_ssim is not None:
                    #     progress_bar.set_postfix({"train img 0 SSIM": f"{smallest_ssim:.{7}f}"})
                    #     progress_bar.update(every_i)
                    # if  is_key_timestep and i == num_iter_per_timestep - 1:
                    #     ssim_init = smallest_ssim
                    #     print("smallest ssim init:", smallest_ssim)
                except Exception as e:
                    print(f"An error occurred: {e}")
                    print ("Skipping to the next iter!")
                    continue   

                

                if is_initial_key_timestep :
                    params, variables = densify(params, variables, optimizer, i)
                
                
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                # running_loss += loss.item()  
                # if (is_initial_timestep) or (not is_initial_timestep  and i < 1000) :

                #     scheduler1.step()
                # else:
                #     scheduler1.step()
                #     # if i % 100 == 0:
                #     #     avg_loss = running_loss / 100
                #     #     scheduler2.step(avg_loss)
                #     #     running_loss = 0.0

                #     if i % 1000 == 0:  # Print every 1000 iterations  
                #         for par in optimizer.param_groups:
                #             current_lr = par['lr']
                #             name = par["name"]
                #             print(f"Iteration {i}, {name} - LR: {current_lr:.6f}")  
                
                if smallest_ssim is not None:
                    

                    if  is_key_timestep == False and  i == num_iter_per_timestep - 1 and smallest_ssim  < ssim_init - ssim_lower_margin :
                        every_i = every_small
                        num_iter_per_timestep += int(num_iter_per_timestep_for_non_key_frames/5)
                        if num_iter_per_timestep >= int(num_iter_per_timestep_for_non_key_frames*1.2) :
                            # ssim_lower_limit_var -= ssim_lower_margin2
                            # ssim_init_var -= ssim_lower_margin2

                            ssim_init-= ssim_lower_margin2

                            # smallest_ssim += 2*ssim_lower_margin
                            # smallest_ssim =  ssim_lower_limit - 2*ssim_lower_margin if  smallest_ssim < ssim_lower_limit - 2*ssim_lower_margin else smallest_ssim
                            # print("ssim_init 0:",smallest_ssim)
                            every_i = every_small
                            # ssim_lower_limit_var = ssim_lower_limit
                            # last_key_frame = t 
                            # break
                            
                            # optimizer = initialize_optimizer_from_prev( optimizer)  
                            # # params, variables = initialize_params(seq, md)  

                            # # optimizer = initialize_optimizer(params, variables)

                            # num_iter_per_timestep_for_key_frames = 5000
                            # is_key_timestep = True
                            # i = 0
                            # num_iter_per_timestep = num_iter_per_timestep_for_key_frames
                            # progress_bar = tqdm(range(num_iter_per_timestep), desc=f"timestep {t}")
                            # index_key_timestep += 1
                            # count_non_key_frames = 0
                            # every_i = every_big
                            # ssim_lower_limit_var = ssim_lower_limit
                        # else:
                        #     i +=1
                        # continue
                       

                    if is_key_timestep == False and i >= num_iter_per_timestep_for_non_key_frames - 1 and i < num_iter_per_timestep_for_non_key_frames * 1.5 and smallest_ssim  >= ssim_init - ssim_lower_margin:
                        every_i = every_big
                        ssim_lower_limit_var = ssim_lower_limit
                        ssim_nonkeyfr_limit = smallest_ssim
                        ssim_init = first_smallest_ssim

                        print("ssim_nonkeyfr_limit:", ssim_nonkeyfr_limit)
                        
                        break
                    if is_key_timestep == False  and i >= num_iter_per_timestep_for_non_key_frames * 1.5:
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



                

        progress_bar.close()
        output_params.append(params2cpu(params, is_key_timestep))
       
        if is_initial_key_timestep:
            variables = initialize_post_first_timestep(params, variables, optimizer)

        
        if is_key_timestep: #and not skip_to_next_timestamp:
            is_key_timestep = False
            # optimizer = initialize_optimizer_per_timestep(variables, optimizer)
            # optimizer = fix_lrs(optimizer)
            # optimizer = initialize_params_from_prev( optimizer)  

        

            
        if t in save_iterations:
            # save_params(output_params, seq, exp)
            save_params_multi(output_params, seq, exp, index_key_timestep, t)
            if not os.path.exists(f'./output/{exp}/{seq}/init_pt_cld.npz'):
                shutil.copy2(f'./data/{seq}/init_pt_cld.npz', f'./output/{exp}/{seq}/init_pt_cld.npz')
        t += 1

    # save_params(output_params, seq, exp)
    save_params_multi(output_params, seq, exp, index_key_timestep, t)
    if not os.path.exists(f'./output/{exp}/{seq}/init_pt_cld.npz'):
                shutil.copy2(f'./data/{seq}/init_pt_cld.npz', f'./output/{exp}/{seq}/init_pt_cld.npz')



if __name__ == "__main__":
  
    
    

   
    # exp_name = "2025-03-27_15-46-48_yoga_ai_adjusted_scl_1_test"        
    # for sequence in ["2025-03-27_15-46-48_yoga_ai_adjusted"]:
    # exp_name = "2025-03-27_15-46-48_yoga_ai_test_scl_1"        
    # for sequence in ["2025-03-27_15-46-48_yoga_ai_test"]:  

    # exp_name = "2025-03-27_15-46-48_yoga_ai_final_scl_1_intrv_637"        
    # for sequence in ["2025-03-27_15-46-48_yoga_ai_final"]:

    # exp_name = "2025-03-27_15-44-28_yoga_ai_final_scl_1_intrv_0"        
    # for sequence in ["2025-03-27_15-44-28_yoga_ai_final"]:
    # exp_name = "2025-05-20_17-44-29_tahir_1_fullsize_scl_1_ai_enhanced_2"
    # for sequence in ["2025-05-20_17-44-29_tahir_1"]:

    # exp_name = "2025-05-20_18-21-50_tahir2_fullsize_scl_2_ai_contrast_alpha14_aligned"
    # for sequence in ["2025-05-20_18-21-50_tahir2_aligned"]:

    # exp_name = "2025-05-20_18-24-20_hamit_burak_1_contrast_alpha12_aligned"
    # for sequence in ["2025-05-20_18-24-20_hamit_burak_1_scl_2"]:
    # exp_name = "2025-05-20_18-24-20_hamit_burak_1_contrast_alpha12_aligned_250"
    # for sequence in ["2025-05-20_18-24-20_hamit_burak_1_scl_2"]:
    # exp_name = "2025-05-20_18-21-50_tahir2_fullsize_scl_2_ai_contrast_alpha14_10cams"
    # for sequence in ["2025-05-20_18-21-50_tahir2_10cams"]:


    
    # exp_name = "2025-05-20_18-21-50_tahir2_fullsize_scl_1_contrast_alpha14_aligned_half_test12"
    # for sequence in ["2025-05-20_18-21-50_tahir_2_half_aligned"]:
    
    # exp_name = "2025-05-20_18-21-50_tahir_2_half_low_error_scl_1_enhanced_aligned_half_test1"
    # for sequence in ["2025-05-20_18-21-50_tahir_2_half_low_error"]:

    # exp_name = "2025-05-20_18-21-50_tahir_2_half_2_low_error_scl_1_contrast_alpha14_aligned_half_test1"
    # for sequence in ["2025-05-20_18-21-50_tahir_2_half_2_low_error"]:

   

    # exp_name = "2025-05-20_18-24-20_hamit_burak_1_half_60_scl_1_enhanced"
    # for sequence in ["2025-05-20_18-24-20_hamit_burak_1_half_60"]:

    # exp_name = "2025-05-20_18-18-09_hamit_1_half_60_scl_1_enhanced"
    # for sequence in ["2025-05-20_18-18-09_hamit_1_half_60"]: 

    # exp_name = "2025-05-20_18-24-20_hamit_burak_1_half_all_scl_1_enhanced_test1"
    # for sequence in ["2025-05-20_18-24-20_hamit_burak_1_half_all"]:

    # exp_name = "2025-05-20_18-28-33_hamit_burak_2_half_all_1_enhanced_test1_70"
    # for sequence in ["2025-05-20_18-28-33_hamit_burak_2_half_all"]:

    # exp_name = "2025-05-20_18-21-50_tahir_2_half_all_low_error_scl_1_enhanced_60_dev12"
    # for sequence in ["2025-05-20_18-21-50_tahir_2_half_all_low_error"]:

    # exp_name = "2025-05-20_18-24-20_hamit_burak_1_half_all_scl_1_enhanced_60_flicker_dev76_test"
    # for sequence in ["2025-05-20_18-24-20_hamit_burak_1_half_all"]:

    # exp_name = "2025-05-20_18-24-20_hamit_burak_1_lumi_adj_half_all_scl1_enhanced_test9"
    # for sequence in ["2025-05-20_18-24-20_hamit_burak_1_lumi_adj"]:

    # exp_name = "2025-05-20_18-21-50_tahir_2_lumi_adj_scl1_enhanced_test33"
    # for sequence in ["2025-05-20_18-21-50_tahir_2_lumi_adj"]:

    # exp_name = "2025-05-20_18-21-50_tahir_2_fullsize_lumi_adj_aligned_start_65-450_interpolated_exp_1_test18"
    # for sequence in ["2025-05-20_18-21-50_tahir_2_fullsize_lumi_adj_aligned"]:
    
    # exp_name = "2025-05-20_18-21-50_tahir_2_fullsize_lumi_adj_aligned_start_100-450_interpolated_exp_1_test1"
    # for sequence in ["2025-05-20_18-21-50_tahir_2_fullsize_lumi_adj_aligned"]:

    # exp_name = "2025-05-20_18-24-20_hamit_burak_1_lumi_adj_60fps_scl1_test1"
    # for sequence in ["2025-05-20_18-24-20_hamit_burak_1_lumi_adj_60fps"]:

    exp_name = "2025-05-20_18-24-20_hamit_burak_1_lumi_adj_scl1_test23"
    for sequence in ["2025-05-20_18-24-20_hamit_burak_1_lumi_adj"]:

    # exp_name = "2025-05-20_18-24-20_hamit_burak_1_new_scl1_test1"    
    # for sequence in ["2025-05-20_18-24-20_hamit_burak_1_new"]:

    # exp_name = "2025-05-20_18-24-20_hamit_burak_1_lumi_adj_cam6_scl1_test8"
    # for sequence in ["2025-05-20_18-24-20_hamit_burak_1_lumi_adj_cam6"]:    


        train(sequence, exp_name, scale=1)
        torch.cuda.empty_cache()

