import torch
import os
import json
import copy
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from random import randint
from tqdm import tqdm
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from helpers import setup_camera, l1_loss_v1, l1_loss_v2, weighted_l2_loss_v1, weighted_l2_loss_v2, quat_mult, \
    o3d_knn, params2rendervar, params2cpu, save_params
from external import calc_ssim, calc_psnr, build_rotation, densify, update_params_and_optimizer
import shutil

class CustomError(Exception):
  pass

def get_dataset(t, md, seq, iters, scale=1):
    dataset = []
    for c in range(len(md['fn'][t])):
        w, h, k, w2c = md['w'], md['h'], md['k'][t][c], md['w2c'][t][c]
        cam = setup_camera(w, h, k, w2c, near=1.0, far=100, scale=scale)
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


def initialize_params(seq, md):
    init_pt_cld = np.load(f"./data/{seq}/init_pt_cld.npz")["data"]
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


def get_loss(params, curr_data, variables, is_initial_timestep):
    losses = {}

    rendervar = params2rendervar(params)
    rendervar['means2D'].retain_grad()
    im, radius, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)
    curr_id = curr_data['id']
    im = torch.exp(params['cam_m'][curr_id])[:, None, None] * im + params['cam_c'][curr_id][:, None, None]
    losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))
    variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification

    segrendervar = params2rendervar(params)
    segrendervar['colors_precomp'] = params['seg_colors']
    seg, _, _, = Renderer(raster_settings=curr_data['cam'])(**segrendervar)
    losses['seg'] = 0.8 * l1_loss_v1(seg, curr_data['seg']) + 0.2 * (1.0 - calc_ssim(seg, curr_data['seg']))

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

        losses['floor'] = torch.clamp(fg_pts[:, 1], max = 2.7).mean()

        bg_pts = rendervar['means3D'][~is_fg]
        bg_rot = rendervar['rotations'][~is_fg]
        losses['bg'] = l1_loss_v2(bg_pts, variables["init_bg_pts"]) + l1_loss_v2(bg_rot, variables["init_bg_rot"])

        losses['soft_col_cons'] = l1_loss_v2(params['rgb_colors'], variables["prev_col"])

    loss_weights = {'im': 1.0, 'seg': 3.0, 'rigid': 4.0, 'rot': 4.0, 'iso': 2.0, 'floor': 2.0, 'bg': 20.0,
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
        progress_bar.set_postfix({"train img 0 PSNR": f"{psnr:.{7}f}"})
        progress_bar.update(every_i)


def train(seq, exp,  scale=1):
    
    if os.path.exists(f"./output/{exp}/{seq}"):
        print(f"Experiment '{exp}' for sequence '{seq}' already exists. Exiting.")
        return

    md = json.load(open(f"./data/{seq}/train_meta.json", 'r'))  # metadata
    num_timesteps = len(md['fn'])

    save_interval = 10
    if num_timesteps > save_interval:
        save_iterations = [ iter  for iter in range(save_interval, num_timesteps) if iter % save_interval == 0]
    else :
        save_iterations = []

    params, variables = initialize_params(seq, md)
    optimizer = initialize_optimizer(params, variables)
    output_params = []
    for t in range(num_timesteps):
        # if t < 43:
        #     continue
        if t  == num_timesteps - 1:
            iters = [t-1]
        else:
            iters = [it for it in range(t+1,t+5) if it < num_timesteps]
        dataset = get_dataset(t, md, seq, iters, scale=scale)
        todo_dataset = []
        is_initial_timestep = (t == 0)
        if not is_initial_timestep:
            params, variables = initialize_per_timestep(params, variables, optimizer)
        num_iter_per_timestep = 10000 if is_initial_timestep else 700
        progress_bar = tqdm(range(num_iter_per_timestep), desc=f"timestep {t}")
        for i in range(num_iter_per_timestep):
            curr_data = get_batch(todo_dataset, dataset)
            try:
                loss, variables = get_loss(params, curr_data, variables, is_initial_timestep)
            except Exception as e:
                print(f"An error occurred: {e}")
                print ("Skipping to the next iter!")
                continue
            loss.backward()
            with torch.no_grad():
                try:
                    report_progress(params, dataset[0], i, progress_bar)
                except Exception as e:
                    print(f"An error occurred: {e}")
                    print ("Skipping to the next iter!")
                    continue   

                if is_initial_timestep:
                    params, variables = densify(params, variables, optimizer, i)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        progress_bar.close()
        output_params.append(params2cpu(params, is_initial_timestep))
        if is_initial_timestep:
            variables = initialize_post_first_timestep(params, variables, optimizer)

        if t in save_iterations:
            save_params(output_params, seq, exp)
            if not os.path.exists(f'./output/{exp}/{seq}/init_pt_cld.npz'):
                shutil.copy2(f'./data/{seq}/init_pt_cld.npz', f'./output/{exp}/{seq}/init_pt_cld.npz')


    save_params(output_params, seq, exp)
    if not os.path.exists(f'./output/{exp}/{seq}/init_pt_cld.npz'):
                shutil.copy2(f'./data/{seq}/init_pt_cld.npz', f'./output/{exp}/{seq}/init_pt_cld.npz')



if __name__ == "__main__":
    # exp_name = "exp_only_oguz_2_scl_4_it_500_20cams"
    # exp_name = "exp_witback_oguz_2_scl_4_it_500_green_test"
    # exp_name ="oguz_2_only_test3"
    # exp_name = "yoga_wo_background_onlypt_scale_4_it_500_pose_1_4"
    # exp_name ="yoga_pose_1_4_only_test2"
    # exp_name = "yoga_wo_background_onlypt_scale_4_it_500_pose_1_4_2"
    # exp_name = "yoga_wo_background_onlypt_scale_4_it_500_pose_1_4_all"

    # for sequence in ["10-09-2024_data/pose_1_4_all"]:
    # exp_name ="oguz_2_calib_scl_2_20_cams_4096x2950_test1"
    # for sequence in ["oguz_2_calib"]:

    # exp_name ="hamit_2024-12-04_16-58-12_evenly_scl_4_it_600_test1"
    # for sequence in ["2024-12-04_16-58-12_evenly"]: # ["hamit_3_27-11-2024_calib"]:  #["hamit_3_27-11-2024_withbkgrnd"]:

    # exp_name ="hamit_2024-12-04_17-14-42_scl_2_it_600_test1"
    # for sequence in ["2024-12-04_17-14-42"]: 
    
    # exp_name = "hamit_2024-12-19_19-12-14_4096_wo_bckgrnd_scl_2_it_1000_test1"
    # for sequence in ["2024-12-19_19-12-14_4096_wo_bckgrnd"]:

    # exp_name ="hamit_2024-12-04_17-14-42_withbckgrnd_scl_4_it_600_simplified"
    # for sequence in ["2024-12-04_17-14-42_withbckgrnd"]: 
    exp_name = "hamit_2024-12-19_19-12-14_4096_wo_bckgrnd_calib_trans_scl_2_it_700"
    for sequence in ["2024-12-19_19-12-14_4096_wo_bckgrnd_calib2_trans"]:
    
        train(sequence, exp_name, scale=2)
        torch.cuda.empty_cache()

