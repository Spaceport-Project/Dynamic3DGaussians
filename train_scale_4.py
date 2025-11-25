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
from helpers_scl import setup_camera, l1_loss_v1, l1_loss_v2, weighted_l2_loss_v1, weighted_l2_loss_v2, quat_mult, \
    o3d_knn, params2rendervar, params2cpu, save_params, save_params_multi
from external import calc_ssim, calc_psnr, build_rotation, densify, update_params_and_optimizer
import shutil
import open3d as o3d
from fused_ssim import fused_ssim


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
  
        new_img_array = np.zeros((resized_seg.shape[0], resized_seg.shape[1]), dtype=np.uint8)

        # Set pixels to 255 where the original image is white, 0 otherwise
        new_img_array[np.all(resized_seg == [255, 255, 255], axis=-1)] = 1
        # new_img_array = np.zeros((resized_image.shape[1], resized_image.shape[2]), dtype=np.uint8)
    
        new_img_array = torch.tensor(new_img_array).float().cuda()
        seg_col = torch.stack((new_img_array, torch.zeros_like(new_img_array), 1 - new_img_array))
        dataset.append({'cam': cam, 'im': resized_image, 'seg': seg_col, 'id': c})
    return dataset


def get_batch(todo_dataset, dataset):
    if not todo_dataset:
        todo_dataset = dataset.copy()
    curr_data = todo_dataset.pop(randint(0, len(todo_dataset) - 1))
    return curr_data

def initialize_params_from_prev(init_file,  md, params_prev):
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

def initialize_params(init_file, md):
    init_pt_cld = np.load(init_file)["data"]
    seg = init_pt_cld[:, 6]
    max_cams = 50
    sq_dist, _ = o3d_knn(init_pt_cld[:, :3], 3)
    mean3_sq_dist = sq_dist.mean(-1).clip(min=0.0000001)
    params = {
        'means3D': init_pt_cld[:, :3],
        'rgb_colors': init_pt_cld[:, 3:6],
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
        'means3D': 0.00016 * 3.8 ,#variables['scene_radius'],
        'rgb_colors': 0.0025,
        'seg_colors': 0.0,
        'unnorm_rotations': 0.001,
        'logit_opacities': 0.05,
        'log_scales': 0.001,
        'cam_m': 1e-4,
        'cam_c': 1e-4,
    }
    param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items()]
    return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)


def get_loss(params, curr_data, variables, is_initial_timestep, t):
    losses = {}
    min_psnr = 10.0  # Poor quality  
    target_psnr = 50.0  # Excellent quality  
    rendervar = params2rendervar(params)
    rendervar['means2D'].retain_grad()
    im, radius, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)
    if t %100 == 0:
        torchvision.utils.save_image(im, f'im.png')
    curr_id = curr_data['id']


    im = torch.exp(params['cam_m'][curr_id])[:, None, None] * im + params['cam_c'][curr_id][:, None, None]

    ssim_value = fused_ssim(im.unsqueeze(0), curr_data['im'].unsqueeze(0))
    # ssim_value = calc_ssim(im, curr_data['im'])
    losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - ssim_value)

    # current_psnr = calc_psnr(im, curr_data['im']).mean()
    # psnr_loss_exp = torch.exp(-current_psnr / 30.0)
    # # sim = calc_ssim(im, curr_data['im'])
    # psnr_loss = torch.clamp((target_psnr - current_psnr) / (target_psnr - min_psnr), 0.0, 1.0)
    # losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (psnr_loss_exp)

    variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification

    segrendervar = params2rendervar(params)
    segrendervar['colors_precomp'] = params['seg_colors']
    seg, _, _, = Renderer(raster_settings=curr_data['cam'])(**segrendervar)
    if t %100 == 0:
        torchvision.utils.save_image(seg, f'seg.png')

    ssim_value = fused_ssim(seg.unsqueeze(0), curr_data['seg'].unsqueeze(0))
    # ssim_value = calc_ssim(seg, curr_data['seg'])
    losses['seg'] = 0.8 * l1_loss_v1(seg, curr_data['seg']) + 0.2 * (1.0 - ssim_value)

    # current_psnr = calc_psnr(seg, curr_data['seg']).mean()
    # psnr_loss_exp = torch.exp(-current_psnr / 30.0)
    # psnr_loss = torch.clamp((target_psnr - current_psnr) / (target_psnr - min_psnr), 0.0, 1.0)


    # losses['seg'] = 0.8 * l1_loss_v1(seg, curr_data['seg']) + 0.2 * (psnr_loss_exp)


    if not is_initial_timestep:
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

        # losses['floor'] = torch.clamp(fg_pts[:, 1], max=2.95).mean()

        bg_pts = rendervar['means3D'][~is_fg]
        bg_rot = rendervar['rotations'][~is_fg]
        losses['bg'] = l1_loss_v2(bg_pts, variables["init_bg_pts"]) + l1_loss_v2(bg_rot, variables["init_bg_rot"])

        losses['soft_col_cons'] = l1_loss_v2(params['rgb_colors'], variables["prev_col"])

    loss_weights = {'im': 1.0, 'seg': 1.0, 'rigid': 4.0, 'rot': 4.0, 'iso': 2.0, 'floor': 2.0, 'bg': 20.0,
                    'soft_col_cons': 0.01}
    loss = sum([loss_weights[k] * v for k, v in losses.items()])
    seen = radius > 0
    variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
    variables['seen'] = seen
    return loss, variables


def initialize_per_timestep(params, variables, optimizer):
    pts = params['means3D']
    rot = torch.nn.functional.normalize(params['unnorm_rotations'])
    new_pts = pts + (pts - variables["prev_pts"])
    new_rot = torch.nn.functional.normalize(rot + (rot - variables["prev_rot"]))

    is_fg = params['seg_colors'][:, 0] > 0.5
    prev_inv_rot_fg = rot[is_fg]
    prev_inv_rot_fg[:, 1:] = -1 * prev_inv_rot_fg[:, 1:]
    fg_pts = pts[is_fg]
    prev_offset = fg_pts[variables["neighbor_indices"]] - fg_pts[:, None]
    variables['prev_inv_rot_fg'] = prev_inv_rot_fg.detach()
    variables['prev_offset'] = prev_offset.detach()
    variables["prev_col"] = params['rgb_colors'].detach()
    variables["prev_pts"] = pts.detach()
    variables["prev_rot"] = rot.detach()

    new_params = {'means3D': new_pts, 'unnorm_rotations': new_rot}
    params = update_params_and_optimizer(new_params, params, optimizer)

    return params, variables


def initialize_post_first_timestep(params, variables, optimizer, num_knn=20):
    is_fg = params['seg_colors'][:, 0] > 0.5
    init_fg_pts = params['means3D'][is_fg]
    init_bg_pts = params['means3D'][~is_fg]
    init_bg_rot = torch.nn.functional.normalize(params['unnorm_rotations'][~is_fg])
    neighbor_sq_dist, neighbor_indices = o3d_knn(init_fg_pts.detach().cpu().numpy(), num_knn)
    neighbor_weight = np.exp(-2000 * neighbor_sq_dist)
    neighbor_dist = np.sqrt(neighbor_sq_dist)
    variables["neighbor_indices"] = torch.tensor(neighbor_indices).cuda().long().contiguous()
    variables["neighbor_weight"] = torch.tensor(neighbor_weight).cuda().float().contiguous()
    variables["neighbor_dist"] = torch.tensor(neighbor_dist).cuda().float().contiguous()

    variables["init_bg_pts"] = init_bg_pts.detach()
    variables["init_bg_rot"] = init_bg_rot.detach()
    variables["prev_pts"] = params['means3D'].detach()
    variables["prev_rot"] = torch.nn.functional.normalize(params['unnorm_rotations']).detach()
    params_to_fix = ['logit_opacities', 'log_scales', 'cam_m', 'cam_c']
    for param_group in optimizer.param_groups:
        if param_group["name"] in params_to_fix:
            param_group['lr'] = 0.0
    return variables


def report_progress(params, data, i, progress_bar, every_i=100):
    if i % every_i == 0:
        im, _, _, = Renderer(raster_settings=data['cam'])(**params2rendervar(params))
        curr_id = data['id']
        im = torch.exp(params['cam_m'][curr_id])[:, None, None] * im + params['cam_c'][curr_id][:, None, None]
        psnr = calc_psnr(im, data['im']).mean()
        # progress_bar.set_postfix({"train img 0 PSNR": f"{psnr:.{7}f}"})
        # progress_bar.update(every_i)
        return psnr.cpu().numpy().item()

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
    num_timesteps = len(md['fn'])
    num_cams = len(md['fn'][0])
    print("num of cams:", num_cams)

    save_interval = 3
    initial_timestep = 0
    if num_timesteps > save_interval:
        save_iterations = [ iter  for iter in range(initial_timestep + save_interval, num_timesteps) if iter % save_interval == 0]
    else :
        save_iterations = []
    init_params_file = f"./data/{seq}/init_pt_cld.npz"
    params, variables = initialize_params(init_params_file, md)
    optimizer = initialize_optimizer(params, variables)
    output_params = []
    is_key_timestep = True
    dataset_id = 36
    num_iter_per_timestep_for_key_frames = int(10000)
    step_num_iter_per_timestep_for_key_frames = int(1000)
    num_iter_per_timestep_for_non_key_frames = int(1000)
    limit_num_iter_per_timestep_for_key_frames = num_iter_per_timestep_for_key_frames * 2
    psnr_lower_limit = 42
    psnr_lower_limit_var = psnr_lower_limit
    psnr_lower_margin = 1
    psnr_init = 9999
    psnr_nonkeyfr_limit = 9999
    count_non_key_frames = 0
    every_small = 5
    every_big = 100
    every_i = every_big
    # with open('coordinates.txt', 'r') as file:
    # Read lines and split each line into numbers
    coords = np.loadtxt('coordinates.txt', delimiter=',', dtype=np.int32)
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
        
        is_initial_timestep = (t == initial_timestep)
        is_key_timestep = is_initial_timestep or  is_key_timestep
        if not is_key_timestep:
            params, variables = initialize_per_timestep(params, variables, optimizer)
            count_non_key_frames += 1
        else:
            count_non_key_frames = 0
            
        num_iter_per_timestep = num_iter_per_timestep_for_key_frames if is_key_timestep else num_iter_per_timestep_for_non_key_frames
        progress_bar = tqdm(range(num_iter_per_timestep), desc=f"timestep {t}")
        skip_to_next_timestamp = False
        if t % 10 == 0  and psnr_nonkeyfr_limit < psnr_lower_limit -0.5:
            every_i = every_big
            psnr_lower_limit_var = psnr_lower_limit
            is_key_timestep = True
            with torch.no_grad():
                params, variables = initialize_params_from_prev(init_params_file, md, params)  
                optimizer = initialize_optimizer(params, variables)
            num_iter_per_timestep = num_iter_per_timestep_for_key_frames
            progress_bar = tqdm(range(num_iter_per_timestep), desc=f"timestep {t}")
            index_key_timestep += 1
            count_non_key_frames =0
            print("Key Frame Called!")
            print("inside psnr_nonkeyfr_limit:",psnr_nonkeyfr_limit)


        i = 0
        while True:
            if i >= num_iter_per_timestep:
                break
        # for i in range(num_iter_per_timestep):
            curr_data = get_batch(todo_dataset, dataset)
            try:
                loss, variables = get_loss(params, curr_data, variables, is_key_timestep, i )
            except Exception as e:
                print(f"An error occurred: {e}")
                print ("Skipping to the next iter!")
                continue
            loss.backward()
            with torch.no_grad():
                try:
                    smallest_psnr = 9999
                    for id in range(num_cams):
                        psnr = report_progress(params, dataset[id], i + 1, progress_bar, every_i)
                        if psnr is not None:
                            # print(id," psnr:", psnr)
                            if psnr < smallest_psnr:
                                smallest_psnr = psnr
                        else:
                            smallest_psnr = None
                    if smallest_psnr is not None:
                        progress_bar.set_postfix({"train img 0 PSNR": f"{smallest_psnr:.{7}f}"})
                        progress_bar.update(every_i)
                    if  is_key_timestep and i == num_iter_per_timestep - 1:
                        psnr_init = smallest_psnr
                        print("smallest psnr init:", smallest_psnr)
                except Exception as e:
                    print(f"An error occurred: {e}")
                    print ("Skipping to the next iter!")
                    continue   

                

                if is_key_timestep:
                    params, variables = densify(params, variables, optimizer, i)
                
                
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                # print("every i", every_i)
                
                if smallest_psnr is not None:
                    

                    if  is_key_timestep == False and  i == num_iter_per_timestep - 1 and smallest_psnr  < psnr_init - psnr_lower_margin :
                        every_i = every_small
                        num_iter_per_timestep += num_iter_per_timestep_for_non_key_frames
                        if num_iter_per_timestep > num_iter_per_timestep_for_non_key_frames * 2:
                         
                            params, variables = initialize_params_from_prev(init_params_file, md, params)  
                            # params, variables = initialize_params(seq, md)  

                            optimizer = initialize_optimizer(params, variables)

                        
                            is_key_timestep = True
                            i = 0
                            num_iter_per_timestep = num_iter_per_timestep_for_key_frames
                            progress_bar = tqdm(range(num_iter_per_timestep), desc=f"timestep {t}")
                            index_key_timestep += 1
                            count_non_key_frames =0
                            every_i = every_big
                            psnr_lower_limit_var = psnr_lower_limit

                            continue
                       

                    if is_key_timestep == False and i >= num_iter_per_timestep_for_non_key_frames - 1 and i < num_iter_per_timestep_for_non_key_frames * 2 and smallest_psnr  >= psnr_init - psnr_lower_margin:
                        every_i = every_big
                        psnr_lower_limit_var = psnr_lower_limit
                        psnr_nonkeyfr_limit = smallest_psnr
                        # print("psnr_nonkeyfr_limit:",psnr_nonkeyfr_limit)
                        
                        break

                    if is_key_timestep == True and i == num_iter_per_timestep - 1 and num_iter_per_timestep < limit_num_iter_per_timestep_for_key_frames and smallest_psnr < psnr_lower_limit:
                        every_i = every_small
                        psnr_lower_limit_var -= 1
                        num_iter_per_timestep += step_num_iter_per_timestep_for_key_frames
                        progress_bar = tqdm(range(num_iter_per_timestep), desc=f"timestep {t}")
                    if is_key_timestep == True and i > num_iter_per_timestep_for_key_frames and smallest_psnr >= psnr_lower_limit_var:
                        psnr_init = smallest_psnr
                        every_i = every_big
                        psnr_lower_limit_var = psnr_lower_limit
                        last_key_frame = t

                        break
                    if is_key_timestep == True and num_iter_per_timestep >= limit_num_iter_per_timestep_for_key_frames:
                        psnr_init = smallest_psnr
                        every_i = every_big
                        psnr_lower_limit_var = psnr_lower_limit
                        last_key_frame = t

                        break
                   
                
                
                i += 1



                

        progress_bar.close()
        output_params.append(params2cpu(params, is_key_timestep))
       
        if is_key_timestep:
            variables = initialize_post_first_timestep(params, variables, optimizer)
        
        if is_key_timestep: #and not skip_to_next_timestamp:
            is_key_timestep = False
            
        if t in save_iterations:
            save_params_multi(output_params, seq, exp, index_key_timestep, t)
            if not os.path.exists(f'./output/{exp}/{seq}/init_pt_cld.npz'):
                shutil.copy2(f'./data/{seq}/init_pt_cld.npz', f'./output/{exp}/{seq}/init_pt_cld.npz')
        t += 1

    save_params_multi(output_params, seq, exp, index_key_timestep, t)
    if not os.path.exists(f'./output/{exp}/{seq}/init_pt_cld.npz'):
                shutil.copy2(f'./data/{seq}/init_pt_cld.npz', f'./output/{exp}/{seq}/init_pt_cld.npz')



if __name__ == "__main__":
  
    
    # exp_name ="hamit_3_27-11-2024_scl1_600_iter_enhanced_test1"
    # for sequence in ["hamit_3_27-11-2024_enhanced"]: # ["hamit_3_27-11-2024_calib"]:  #["hamit_3_27-11-2024_withbkgrnd"]:

    # exp_name ="hamit_2024-12-04_16-58-12_evenly_withbckgrnd_scl_4_it_600_test1"
    # for sequence in ["2024-12-04_16-58-12_evenly_withbckgrnd"]: 

    # exp_name = "hamit_2024-12-12_10-29-44_4096_scl_2_it_500_test1"
    # for sequence in ["2024-12-12_10-29-44_4096"]:
    
    # exp_name = "hamit_2024-12-19_19-12-14_4096_wo_bckgrnd_scl_2_it_700"
    # for sequence in ["2024-12-19_19-12-14_4096_wo_bckgrnd"]:

    # exp_name = "hamit_2024-12-19_19-12-14_4096_wo_bckgrnd_0-350_scl_2_it_700"
    # for sequence in ["2024-12-19_19-12-14_4096_wo_bckgrnd_0-350"]:

    # exp_name = "hamit_2024-12-19_20-11-26_4096_180_wo_bckgrnd_scl_2_it_700"
    # for sequence in ["2024-12-19_20-11-26_4096_180_wo_bckgrnd"]:
    
    # exp_name = "hamit_2024-12-19_19-12-14_4096_wo_bckgrnd_calib_scl_2_it_700"
    # for sequence in ["2024-12-19_19-12-14_4096_wo_bckgrnd_calib2"]:

    # exp_name = "hamit_2024-12-19_19-12-14_4096_wo_bckgrnd_enhanced_scl_1_it_700"
    # for sequence in ["2024-12-19_19-12-14_4096_wo_bckgrnd_enhanced"]:
    
    # exp_name = "hamit_2024-12-19_19-12-14_4096_wo_bckgrnd_calib_scl_2_it_700"
    # for sequence in ["2024-12-19_19-12-14_4096_wo_bckgrnd_calib_agisoft_test"]:
    
    # exp_name = "hamit_2024-12-19_19-12-14_4096_wo_bckgrnd_calib_scl_2_it_700_test1"
    # for sequence in ["2024-12-19_19-12-14_4096_wo_bckgrnd_calib_meshroom"]:
    # exp_name = "hamit_2024-12-19_19-12-14_4096_wo_bckgrnd_agisoft_scl_2_it_700"
    # for sequence in ["2024-12-19_19-12-14_4096_wo_bckgrnd_agisoft_test"]:
    
    # exp_name = "hamit_2024-12-19_19-12-14_4096_wo_bckgrnd_calib_colmap_scl_2_it_700"
    # for sequence in ["2024-12-19_19-12-14_4096_wo_bckgrnd_calib_colmap_basedon_calib"]:
    
    # exp_name = "hamit_2024-12-19_20-11-26_4096_180_with_table_scl_2_it_700"    
    # for sequence in ["2024-12-19_20-11-26_4096_180_with_table"]:
    
    # exp_name = "hamit_2024-12-19_20-11-26_4096_colmap_new_fix_scl_2_it_700"    
    # for sequence in ["2024-12-19_19-12-14_4096_wo_bckgrnd_calib_colmap_new_fix"]:
    
    # exp_name = "hamit_2024-12-19_20-11-26_4096_new_fix_multical_scl_2_it_700"    
    # for sequence in ["2024-12-19_19-12-14_4096_wo_bckgrnd_calib_new_fix_multical"]:

    # exp_name = "hamit_2024-12-19_20-11-26_4096_org_multical_scl_2_it_700"    
    # for sequence in ["2024-12-19_19-12-14_4096_wo_bckgrnd_calib_org_multical"]:
    
    # exp_name = "2025-02-05_14-06-52_gain_9_scl_1_it_700_2"        
    # for sequence in ["2025-02-05_14-06-52_gain_9"]:

    # exp_name = "2025-02-05_14-16-46_gain_9_scl_1_it_700_test_expanded_start_0"        
    # for sequence in ["2025-02-05_14-16-46_gain_9"]:
    
    # exp_name = "2025-02-05_14-29-36_scl_1_test_expanded_start_0"        
    # for sequence in ["2025-02-05_14-29-36"]:

    # exp_name = "2025-03-27_15-42-48_yoga_ai_scl_1_1000"        
    # for sequence in ["2025-03-27_15-42-48_yoga_ai"]:
    
    # exp_name = "2025-03-27_15-46-48_yoga_ai_new_scl_1_all_green"        
    # for sequence in ["2025-03-27_15-46-48_yoga_ai_new"]:

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

    
    exp_name = "2025-05-20_18-21-50_tahir2_fullsize_scl_1_contrast_alpha14_aligned_half_test2"
    for sequence in ["2025-05-20_18-21-50_tahir_2_half_aligned"]:



        train(sequence, exp_name, scale=1)
        torch.cuda.empty_cache()

