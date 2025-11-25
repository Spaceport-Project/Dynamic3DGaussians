
"""
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file found here:
# https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md
#
# For inquiries contact  george.drettakis@inria.fr

#######################################################################################################################
##### NOTE: CODE IN THIS FILE IS NOT INCLUDED IN THE OVERALL PROJECT'S MIT LICENSE #####
##### USE OF THIS CODE FOLLOWS THE COPYRIGHT NOTICE ABOVE #####
#######################################################################################################################
"""

import numpy as np
import torch
import torch.nn.functional as func
from torch.autograd import Variable
from math import exp
from plyfile import PlyElement, PlyData 


def build_rotation(q):
    norm = torch.sqrt(q[:, 0] * q[:, 0] + q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3])
    q = q / norm[:, None]
    rot = torch.zeros((q.size(0), 3, 3), device='cuda')
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rot[:, 0, 1] = 2 * (x * y - r * z)
    rot[:, 0, 2] = 2 * (x * z + r * y)
    rot[:, 1, 0] = 2 * (x * y + r * z)
    rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rot[:, 1, 2] = 2 * (y * z - r * x)
    rot[:, 2, 0] = 2 * (x * z - r * y)
    rot[:, 2, 1] = 2 * (y * z + r * x)
    rot[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rot


def calc_mse(img1, img2):
    return ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def calc_psnr(img1, img2):
    mse = ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def calc_ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    # mask = torch.any(img2 != 0, dim=0)
    # fil_img2 = img2[:, mask]
    # fil_img1 = img1[:,mask ]

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = func.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = func.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = func.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = func.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = func.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def calc_ssim_with_mask(img1, img2, mask, window_size=11):
    """
    Calculate SSIM normally, then only average over valid regions
    """
    mask = mask.float()
    
    # Ensure mask matches image shape
    if mask.shape != img1.shape:
        while mask.dim() < img1.dim():
            mask = mask.unsqueeze(0)
        if mask.shape[1] == 1 and img1.shape[1] > 1:
            mask = mask.expand(-1, img1.shape[1], -1, -1)
    
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    # Get SSIM map (not averaged)
    ssim_map = _ssim(img1, img2, window, window_size, channel, size_average=True)
    
    # Only average over valid regions
    valid_ssim = ssim_map * mask
    valid_pixels = mask.sum()
    
    if valid_pixels == 0:
        return torch.tensor(0.0, device=img1.device)
    
    return valid_ssim.sum() / valid_pixels

def accumulate_mean2d_gradient(variables):
    variables['means2D_gradient_accum'][variables['seen']] += torch.norm(
        variables['means2D'].grad[variables['seen'], :2], dim=-1)
    variables['denom'][variables['seen']] += 1
    return variables

def accumulate_mean3d_gradient(variables):
    variables['means3D_gradient_accum'][variables['seen']] += torch.norm(
        variables['means3D'].grad[variables['seen'], :3], dim=-1)
    variables['denom'][variables['seen']] += 1
    return variables


def update_params_and_optimizer(new_params, params, optimizer):
    for k, v in new_params.items():
        group = [x for x in optimizer.param_groups if x["name"] == k][0]
        stored_state = optimizer.state.get(group['params'][0], None)

        stored_state["exp_avg"] = torch.zeros_like(v)
        stored_state["exp_avg_sq"] = torch.zeros_like(v)
        del optimizer.state[group['params'][0]]

        group["params"][0] = torch.nn.Parameter(v.requires_grad_(True))
        optimizer.state[group['params'][0]] = stored_state
        params[k] = group["params"][0]
    return params

# def update_params( params, optimizer, indices):
    
#     # for group in optimizer.param_groups:  
#     #     for param in group['params']:  
#     #         mask = torch.zeros_like(param, dtype=torch.bool) 
#     #         mask[indices] = 1  # False where you want to stop gradients  
#     #         if param.grad is not None:  
#     #             # Apply mask to gradients  
#     #             param.grad *= mask  
    
#     #             # Also apply mask to optimizer state (e.g., for Adam)  
#     #             state = optimizer.state[param]  
#     #             if 'exp_avg' in state:  
#     #                 state['exp_avg'] *= mask  
#     #             if 'exp_avg_sq' in state:  
#     #                 state['exp_avg_sq'] *= mask  
    
#     # return 

#     for k, v in params.items():
#         # group = [x for x in optimizer.param_groups if x["name"] == k][0]
#         # stored_state = optimizer.state.get(group['params'][0], None)

#         # stored_state["exp_avg"] = torch.zeros_like(v)
#         # stored_state["exp_avg_sq"] = torch.zeros_like(v)
#         # del optimizer.state[group['params'][0]]
#         old_param = params[k].clone()

#         mask = torch.zeros_like(old_param, dtype=torch.bool) 
#         mask[indices] = 1  # False where you want to stop gradients  


#         old_param_detached = torch.where(mask, old_param, old_param.detach())  
#         params[k] = old_param_detached # torch.nn.Parameter(old_param_detached.requires_grad_(True))
#         # group["params"][0] = torch.nn.Parameter(old_param_detached.requires_grad_(True))
#         # optimizer.state[group['params'][0]] = stored_state
#         # params[k] = group["params"][0]
#     return params

def cat_params_to_optimizer2(new_params, params, optimizer, indices):
    for k, v in new_params.items():
        group = [g for g in optimizer.param_groups if g['name'] == k][0]
        old_param = group["params"][0]
        stored_state = optimizer.state.get(old_param, None)
        mask = torch.zeros_like(old_param, dtype=torch.bool) 
        mask[indices] = True  # False where you want to stop gradients  


        old_param_detached = torch.where(mask, old_param, old_param.detach())  

        # Detach the old param to stop gradients
        # old_param_detached = old_param.detach()
        # Ensure new tensor is a Parameter with requires_grad=True
        if not isinstance(v, torch.nn.Parameter):
            v = torch.nn.Parameter(v, requires_grad=True)
        
        # Concatenate: old (no grad) + new (with grad)
        new_cat_param = torch.nn.Parameter(torch.cat((old_param_detached, v), dim=0), requires_grad=True)
        
        # Update optimizer state if present
        if stored_state is not None:
            # Expand optimizer state for new params
            stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(v)), dim=0)
            stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(v)), dim=0)
            del optimizer.state[old_param]
            group["params"][0] = new_cat_param
            optimizer.state[new_cat_param] = stored_state
        else:
            group["params"][0] = new_cat_param
        
        params[k] = group["params"][0]
    return params

def cat_vars_params_to_optimizer(indices, new_params, params, variables, optimizer, n=1):
    for k, v in new_params.items():
        group = [g for g in optimizer.param_groups if g['name'] == k][0]
        stored_state = optimizer.state.get(group['params'][0], None)
        if stored_state is not None:
            stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(v)), dim=0)
            stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(v)), dim=0)
            del optimizer.state[group['params'][0]]
            old_param = group["params"][0]  
            new_param = torch.nn.Parameter(torch.cat((old_param, v), dim=0).requires_grad_(True))  
            if old_param.grad is not None: 
                if v.grad is not None:
                    new_param.grad = torch.cat((old_param.grad, v.grad), dim=0) 
                else:
                    new_param.grad = torch.cat((old_param.grad, torch.zeros_like(v)), dim=0) 
            # group["params"][0] = torch.nn.Parameter(torch.cat((group["params"][0], v), dim=0).requires_grad_(True))
            group["params"][0] = new_param  
            params[k] = new_param  
            optimizer.state[group['params'][0]] = stored_state
            # params[k] = group["params"][0]
        else:
            old_param = group["params"][0]  
            new_param = torch.nn.Parameter(torch.cat((old_param, v), dim=0).requires_grad_(True))  
            if old_param.grad is not None:  
                if v.grad is not None:
                    new_param.grad = torch.cat((old_param.grad, v.grad), dim=0) 
                else:
                    new_param.grad = torch.cat((old_param.grad, torch.zeros_like(v)), dim=0)
            group["params"][0] = new_param  
            params[k] = new_param  
            # group["params"][0] = torch.nn.Parameter(torch.cat((group["params"][0], v), dim=0).requires_grad_(True))
            # params[k] = group["params"][0]
    
    variables['prev_inv_rot_fg'] = torch.cat((variables['prev_inv_rot_fg'],  variables['prev_inv_rot_fg'][indices].repeat(n,1)), dim=0)
    variables['prev_offset'] = torch.cat((variables['prev_offset'],  variables['prev_offset'][indices].repeat(n,1,1)), dim=0)
    variables["prev_pts"] =  torch.cat((variables['prev_pts'],  variables['prev_pts'][indices].repeat(n,1)), dim=0)
    variables["prev_rot"] = torch.cat((variables['prev_rot'],  variables['prev_rot'][indices].repeat(n,1)), dim=0)

    variables["prev_col"] = torch.cat((variables['prev_col'],  variables['prev_col'][indices].repeat(n,1)), dim=0)
    variables["neighbor_indices"] = torch.cat((variables['neighbor_indices'],  variables['neighbor_indices'][indices].repeat(n,1)), dim=0)
    variables["neighbor_weight"] = torch.cat((variables['neighbor_weight'],  variables['neighbor_weight'][indices].repeat(n,1)), dim=0)
    variables["neighbor_dist"] = torch.cat((variables['neighbor_dist'],  variables['neighbor_dist'][indices].repeat(n,1)), dim=0)


    return params, variables

def cat_params_to_optimizer(new_params, params, optimizer):
    for k, v in new_params.items():
        group = [g for g in optimizer.param_groups if g['name'] == k][0]
        stored_state = optimizer.state.get(group['params'][0], None)
        if stored_state is not None:
            stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(v)), dim=0)
            stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(v)), dim=0)
            del optimizer.state[group['params'][0]]
            old_param = group["params"][0]  
            new_param = torch.nn.Parameter(torch.cat((old_param, v), dim=0).requires_grad_(True))  
            if old_param.grad is not None: 
                if v.grad is not None:
                    new_param.grad = torch.cat((old_param.grad, v.grad), dim=0) 
                else:
                    new_param.grad = torch.cat((old_param.grad, torch.zeros_like(v)), dim=0) 
            # group["params"][0] = torch.nn.Parameter(torch.cat((group["params"][0], v), dim=0).requires_grad_(True))
            group["params"][0] = new_param  
            params[k] = new_param  
            optimizer.state[group['params'][0]] = stored_state
            # params[k] = group["params"][0]
        else:
            old_param = group["params"][0]  
            new_param = torch.nn.Parameter(torch.cat((old_param, v), dim=0).requires_grad_(True))  
            if old_param.grad is not None:  
                if v.grad is not None:
                    new_param.grad = torch.cat((old_param.grad, v.grad), dim=0) 
                else:
                    new_param.grad = torch.cat((old_param.grad, torch.zeros_like(v)), dim=0) 
            group["params"][0] = new_param  
            params[k] = new_param  
            # group["params"][0] = torch.nn.Parameter(torch.cat((group["params"][0], v), dim=0).requires_grad_(True))
            # params[k] = group["params"][0]
    return params

def remove_points_with_vars(to_remove, params, variables, optimizer):
    to_keep = ~to_remove
    keys = [k for k in params.keys() if k not in ['cam_m', 'cam_c']]
    for k in keys:
        group = [g for g in optimizer.param_groups if g['name'] == k][0]
        stored_state = optimizer.state.get(group['params'][0], None)
        if stored_state is not None:
            stored_state["exp_avg"] = stored_state["exp_avg"][to_keep]
            stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][to_keep]
            del optimizer.state[group['params'][0]]
            old_param = group["params"][0]  
            # new_param= torch.nn.Parameter((group["params"][0][to_keep].requires_grad_(True)))
            # optimizer.state[group['params'][0]] = stored_state
            # params[k] = group["params"][0]
            new_param = torch.nn.Parameter( old_param[to_keep].requires_grad_(True))
            new_param.grad =  old_param.grad[to_keep]
            group["params"][0] = new_param 
            optimizer.state[group['params'][0]] = stored_state
            params[k] = new_param
           
        else:
            # group["params"][0] = torch.nn.Parameter(group["params"][0][to_keep].requires_grad_(True))
            old_param = group["params"][0]  
            new_param = torch.nn.Parameter((old_param[to_keep].requires_grad_(True)))
            new_param.grad =  old_param.grad[to_keep]
            group["params"][0] = new_param  
            params[k] = new_param
    
    variables['prev_inv_rot_fg'] = variables['prev_inv_rot_fg'][to_keep]
    variables['prev_offset'] = variables['prev_offset'][to_keep]
    variables["prev_pts"] =  variables['prev_pts'][to_keep]
    variables["prev_rot"] = variables['prev_rot'][to_keep]

    variables["prev_col"] = variables['prev_col'][to_keep]
    variables["neighbor_indices"] = variables['neighbor_indices'][to_keep]
    variables["neighbor_weight"] = variables['neighbor_weight'][to_keep]
    variables["neighbor_dist"] = variables['neighbor_dist'][to_keep]


    variables['means2D_gradient_accum'] = variables['means2D_gradient_accum'][to_keep]
    variables['denom'] = variables['denom'][to_keep]
    variables['max_2D_radius'] = variables['max_2D_radius'][to_keep]
    return params, variables


def remove_points(to_remove, params, variables, optimizer):
    to_keep = ~to_remove
    keys = [k for k in params.keys() if k not in ['cam_m', 'cam_c']]
    for k in keys:
        group = [g for g in optimizer.param_groups if g['name'] == k][0]
        stored_state = optimizer.state.get(group['params'][0], None)
        if stored_state is not None:
            stored_state["exp_avg"] = stored_state["exp_avg"][to_keep]
            stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][to_keep]
            del optimizer.state[group['params'][0]]
            old_param = group["params"][0]  
            # new_param= torch.nn.Parameter((group["params"][0][to_keep].requires_grad_(True)))
            # optimizer.state[group['params'][0]] = stored_state
            # params[k] = group["params"][0]

            new_param = torch.nn.Parameter( old_param[to_keep].requires_grad_(True))
            new_param.grad =  old_param.grad[to_keep]
            group["params"][0] = new_param 
            optimizer.state[group['params'][0]] = stored_state
            params[k] = new_param
           
        else:
            # group["params"][0] = torch.nn.Parameter(group["params"][0][to_keep].requires_grad_(True))
            old_param = group["params"][0]  
            new_param = torch.nn.Parameter((old_param[to_keep].requires_grad_(True)))
            new_param.grad =  old_param.grad[to_keep]
            group["params"][0] = new_param  
            params[k] = new_param
    variables['means2D_gradient_accum'] = variables['means2D_gradient_accum'][to_keep]
    variables['denom'] = variables['denom'][to_keep]
    variables['max_2D_radius'] = variables['max_2D_radius'][to_keep]
    return params, variables

def remove_points_with_indices(to_remove_indices, params, variables, optimizer):

    mask = torch.ones(params["means3D"].size(0), dtype=torch.bool)
    mask[to_remove_indices] = False
    # result = original_tensor[mask]
    to_keep = ~mask[to_remove_indices]
    keys = [k for k in params.keys() if k not in ['cam_m', 'cam_c']]
    for k in keys:
        group = [g for g in optimizer.param_groups if g['name'] == k][0]
        stored_state = optimizer.state.get(group['params'][0], None)
        if stored_state is not None:
            stored_state["exp_avg"] = stored_state["exp_avg"][to_keep]
            stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][to_keep]
            del optimizer.state[group['params'][0]]
            group["params"][0] = torch.nn.Parameter((group["params"][0][to_keep].requires_grad_(True)))
            optimizer.state[group['params'][0]] = stored_state
            params[k] = group["params"][0]
        else:
            group["params"][0] = torch.nn.Parameter(group["params"][0][to_keep].requires_grad_(True))
            params[k] = group["params"][0]
    variables['means2D_gradient_accum'] = variables['means2D_gradient_accum'][to_keep]
    variables['denom'] = variables['denom'][to_keep]
    variables['max_2D_radius'] = variables['max_2D_radius'][to_keep]
    return params, variables

def keep_points(num_pts, params, variables, optimizer):
    # to_keep = ~to_remove
    keys = [k for k in params.keys() if k not in ['cam_m', 'cam_c']]
    for k in keys:
        group = [g for g in optimizer.param_groups if g['name'] == k][0]
        stored_state = optimizer.state.get(group['params'][0], None)
        if stored_state is not None:
            stored_state["exp_avg"] = stored_state["exp_avg"][:num_pts]
            stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][:num_pts]
            del optimizer.state[group['params'][0]]
            group["params"][0] = torch.nn.Parameter((group["params"][0][:num_pts].requires_grad_(True)))
            optimizer.state[group['params'][0]] = stored_state
            params[k] = group["params"][0]
        else:
            group["params"][0] = torch.nn.Parameter(group["params"][0][:num_pts].requires_grad_(True))
            params[k] = group["params"][0]
    variables['means2D_gradient_accum'] = variables['means2D_gradient_accum'][:num_pts]
    variables['denom'] = variables['denom'][:num_pts]
    variables['max_2D_radius'] = variables['max_2D_radius'][:num_pts]
    return params, variables


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))
def np_inverse_sigmoid(x):
    x_clipped = np.clip(x, 1e-7, 1 - 1e-7)  
    return np.log(x_clipped / (1 - x_clipped))
def sigmoid(z):
    return 1/(1 + np.exp(-z))

def densify_nokeyframes(params, variables, optimizer, i):
    if i <= 500:
        variables = accumulate_mean2d_gradient(variables)
        grad_thresh = 0.0002
        if (i >= 200) and (i % 100 == 0):
            grads = variables['means2D_gradient_accum'] / variables['denom']
            grads[grads.isnan()] = 0.0
            
            if params['means3D'].shape[0] < 70000:
                to_clone = torch.logical_and(grads >= grad_thresh, (
                            torch.max(torch.exp(params['log_scales']), dim=1).values <= 0.01 * variables['scene_radius']))
                new_params = {k: v[to_clone] for k, v in params.items() if k not in ['cam_m', 'cam_c']}
                params = cat_params_to_optimizer(new_params, params, optimizer)
                
            num_pts = params['means3D'].shape[0]           


            padded_grad = torch.zeros(num_pts, device="cuda")
            padded_grad[:grads.shape[0]] = grads
            to_split = torch.logical_and(padded_grad >= grad_thresh,
                                            torch.max(torch.exp(params['log_scales']), dim=1).values > 0.01 * variables[
                                                'scene_radius'])
            n = 2  # number to split into
            new_params = {k: v[to_split].repeat(n, 1) for k, v in params.items() if k not in ['cam_m', 'cam_c']}
            stds = torch.exp(params['log_scales'])[to_split].repeat(n, 1)
            means = torch.zeros((stds.size(0), 3), device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(params['unnorm_rotations'][to_split]).repeat(n, 1, 1)
            new_params['means3D'] += torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
            new_params['log_scales'] = torch.log(torch.exp(new_params['log_scales']) / (0.8 * n))
            params = cat_params_to_optimizer(new_params, params, optimizer)
            num_pts = params['means3D'].shape[0]

            variables['means2D_gradient_accum'] = torch.zeros(num_pts, device="cuda")
            variables['denom'] = torch.zeros(num_pts, device="cuda")
            variables['max_2D_radius'] = torch.zeros(num_pts, device="cuda")
            to_remove = torch.cat((to_split, torch.zeros(n * to_split.sum(), dtype=torch.bool, device="cuda")))
            params, variables = remove_points(to_remove, params, variables, optimizer)

            remove_threshold = 0.25 if i == 500 else 0.005
            to_remove = (torch.sigmoid(params['logit_opacities']) < remove_threshold).squeeze()
            if i >= 300:
                big_points_ws = torch.exp(params['log_scales']).max(dim=1).values > 0.1 * variables['scene_radius']
                to_remove = torch.logical_or(to_remove, big_points_ws)
            params, variables = remove_points(to_remove, params, variables, optimizer)
            
        

            torch.cuda.empty_cache()

        if i > 0 and i % 300 == 0:
            new_params = {'logit_opacities': inverse_sigmoid(torch.ones_like(params['logit_opacities']) * 0.01)}
            params = update_params_and_optimizer(new_params, params, optimizer)
        
        


    return params, variables


def densify(params, variables, optimizer, i):
    if i <= 7000:
        variables = accumulate_mean2d_gradient(variables)
        grad_thresh = 0.0002
        if (i >= 500) and (i % 100 == 0):
            grads = variables['means2D_gradient_accum'] / variables['denom']
            grads[grads.isnan()] = 0.0
            
            # if params['means3D'].shape[0] < 50000:
            to_clone = torch.logical_and(grads >= grad_thresh, (
                        torch.max(torch.exp(params['log_scales']), dim=1).values <= 0.01 * variables['scene_radius']))
            new_params = {k: v[to_clone] for k, v in params.items() if k not in ['cam_m', 'cam_c']}
            params = cat_params_to_optimizer(new_params, params, optimizer)


           
          
            num_pts = params['means3D'].shape[0]           


            padded_grad = torch.zeros(num_pts, device="cuda")
            padded_grad[:grads.shape[0]] = grads
            to_split = torch.logical_and(padded_grad >= grad_thresh,
                                         torch.max(torch.exp(params['log_scales']), dim=1).values > 0.01 * variables[
                                             'scene_radius'])
            n = 2  # number to split into
            new_params = {k: v[to_split].repeat(n, 1) for k, v in params.items() if k not in ['cam_m', 'cam_c']}
            stds = torch.exp(params['log_scales'])[to_split].repeat(n, 1)
            means = torch.zeros((stds.size(0), 3), device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(params['unnorm_rotations'][to_split]).repeat(n, 1, 1)
            new_params['means3D'] += torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
            new_params['log_scales'] = torch.log(torch.exp(new_params['log_scales']) / (0.8 * n))
            params = cat_params_to_optimizer(new_params, params, optimizer)
            num_pts = params['means3D'].shape[0]

            variables['means2D_gradient_accum'] = torch.zeros(num_pts, device="cuda")
            variables['denom'] = torch.zeros(num_pts, device="cuda")
            variables['max_2D_radius'] = torch.zeros(num_pts, device="cuda")
            to_remove = torch.cat((to_split, torch.zeros(n * to_split.sum(), dtype=torch.bool, device="cuda")))
            params, variables = remove_points(to_remove, params, variables, optimizer)

            remove_threshold = 0.25 if i == 5000 else 0.005
            to_remove = (torch.sigmoid(params['logit_opacities']) < remove_threshold).squeeze()
            if i >= 3000:
                big_points_ws = torch.exp(params['log_scales']).max(dim=1).values > 0.1 * variables['scene_radius']
                to_remove = torch.logical_or(to_remove, big_points_ws)
            params, variables = remove_points(to_remove, params, variables, optimizer)
           
            # xyz = [ tuple(v for v in ver ) for ver in params['means3D'].detach().cpu().numpy()]
            # vertex=np.array(xyz, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
            # if i == 5400:

            #     el = PlyElement.describe(vertex, 'vertex')

            #     PlyData([el]).write(f"params_{i:03d}.ply")

            torch.cuda.empty_cache()

        if i > 0 and i % 3000 == 0:
            new_params = {'logit_opacities': inverse_sigmoid(torch.ones_like(params['logit_opacities']) * 0.01)}
            params = update_params_and_optimizer(new_params, params, optimizer)
        
       


    return params, variables

# def densify_nokey_frames(indices, params, variables, optimizer, i):

#     # variables = accumulate_mean2d_gradient(variables)
#     grad_thresh = 0.002
    
#     # if i % 100 == 0:
#     grads = variables['means2D_gradient_accum'] / variables['denom'] 
#     grads[grads.isnan()] = 0.0
#     grad_mask = (grads >= grad_thresh) & (grads != 0.0)
#     to_clone = torch.logical_and(grad_mask[indices], (
#                 torch.max(torch.exp(params['log_scales'][indices]), dim=1).values <= 0.01 * variables['scene_radius']))
    
#     # to_clone = torch.max(torch.exp(params['log_scales'][indices]), dim=1).values <= 0.01 * variables['scene_radius']
    
#     filtered_indices_clone = indices[to_clone]
#     unfiltered_indices_clone = indices[~to_clone]
#     # mask = ~to_clone
#     # mask = torch.zeros_like(params['means3D'], dtype=torch.bool)
#     # mask[unfiltered_indices_clone] = True
    
#     # unfiltered_indices_clone = indices[~to_clone]
#     # filtered_indices_split =  indices[~to_clone]
#     # new_params = {k: v[filtered_indices_clone]  for k, v in params.items() if k not in ['cam_m', 'cam_c']}
#     new_params = {}
#     for k, v in params.items():
#         if k not in ['cam_m', 'cam_c']:
#             new_v = v[filtered_indices_clone]
#             new_params[k] = new_v
            
#             # Copy gradients if they exist
#             if v.grad is not None:
#                 # Create a new tensor for the grad, same as new_v
#                 new_v_grad = v.grad[filtered_indices_clone]
#                 # Assign the grad to the new tensor
#                 new_v.grad = new_v_grad
#             v.grad[unfiltered_indices_clone] = 0.0 #torch.tensor([0.0,0.0,0.0], device="cuda")

#     params, variables = cat_vars_params_to_optimizer(filtered_indices_clone, new_params, params, variables, optimizer)

#     num_pts = params['means3D'].shape[0]


#     padded_grad = torch.zeros(num_pts, device="cuda")
#     padded_grad[:grads.shape[0]] = grads
#     padded_grad_mask = (padded_grad >= grad_thresh) & (padded_grad != 0.0)
#     to_split = torch.logical_and(padded_grad_mask[indices],
#                                     torch.max(torch.exp(params['log_scales'][indices]), dim=1).values > 0.01 * variables[
#                                         'scene_radius'])
    
#     filtered_indices_split = indices[to_split]

#     n = 2  

#     new_params = {}
#     for k, v in params.items():
#         if k not in ['cam_m', 'cam_c']:
#             # First filter, then repeat
#             filtered_v = v[filtered_indices_split]
#             new_v = filtered_v.repeat(n, 1)
#             new_params[k] = new_v

#             # Copy gradients if they exist
#             if v.grad is not None:
#                 # Filter the gradients, then repeat them too
#                 filtered_grad = v.grad[filtered_indices_split]
#                 new_v.grad = filtered_grad.repeat(n, 1)

#     stds = torch.exp(params['log_scales'])[filtered_indices_split].repeat(n, 1)
#     means = torch.zeros((stds.size(0), 3), device="cuda")
#     samples = torch.normal(mean=means, std=stds)
#     rots = build_rotation(params['unnorm_rotations'][filtered_indices_split]).repeat(n, 1, 1)
#     new_params['means3D'] += torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
#     new_params['log_scales'] = torch.log(torch.exp(new_params['log_scales']) / (0.8 * n))
#     # new_params['log_scales'].retain_grad()
   
#     params, variables= cat_vars_params_to_optimizer(filtered_indices_split, new_params, params, variables, optimizer, n=n)
#     num_pts = params['means3D'].shape[0]


#     variables['means2D_gradient_accum'] = torch.zeros(num_pts, device="cuda")
#     variables['denom'] = torch.zeros(num_pts, device="cuda")
#     variables['max_2D_radius'] = torch.zeros(num_pts, device="cuda")

#     to_remove = torch.zeros(num_pts, dtype=torch.bool, device="cuda")
#     to_remove[filtered_indices_split] = True
#     # result = original_tensor[mask]

#     params, variables = remove_points_with_vars(to_remove, params, variables, optimizer)
#     indices_split= torch.where(torch.all(params["means3D"].grad != 0, dim=-1))[0]

#     remove_threshold = 0.25 if i == 5000 else 0.005
#     to_remove_mask = (torch.sigmoid(params['logit_opacities'][indices_split]) < remove_threshold).squeeze()
    
#     big_points_ws = torch.exp(params['log_scales'][indices_split]).max(dim=1).values > 0.1 * variables['scene_radius']
#     to_remove_mask = big_points_ws # torch.logical_or(to_remove_mask, big_points_ws)
#     filtered_remove_indices = indices_split[to_remove_mask]
#     num_pts = params['means3D'].shape[0]

#     to_remove = torch.zeros(num_pts, dtype=torch.bool, device="cuda")
#     to_remove[filtered_remove_indices] = True

#     params, variables = remove_points_with_vars(to_remove, params, variables, optimizer)
    
#     final_indices= torch.where(torch.all(params["means3D"].grad != 0, dim=-1))[0]    

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

#     return params, variables, final_indices

# def densify_nokey_frames2(indices, params, variables, optimizer, i):

#     # variables = accumulate_mean2d_gradient(variables)
#     # grad_thresh = 0.002
    
#     # # if i % 100 == 0:
#     # grads = variables['means2D_gradient_accum'] / variables['denom'] 
#     # grads[grads.isnan()] = 0.0
#     # grad_mask = (grads >= grad_thresh) & (grads != 0.0)
#     # to_clone = torch.logical_and(grad_mask[indices], (
#     #             torch.max(torch.exp(params['log_scales'][indices]), dim=1).values <= 0.01 * variables['scene_radius']))
    
    
#     # filtered_indices_clone = indices[to_clone]
#     # unfiltered_indices_clone = indices[~to_clone]
   
#     # new_params = {}
#     # for k, v in params.items():
#     #     if k not in ['cam_m', 'cam_c']:
#     #         new_v = v[filtered_indices_clone]
#     #         new_params[k] = new_v
            
#     #         # Copy gradients if they exist
#     #         if v.grad is not None:
#     #             # Create a new tensor for the grad, same as new_v
#     #             new_v_grad = v.grad[filtered_indices_clone]
#     #             # Assign the grad to the new tensor
#     #             new_v.grad = new_v_grad
#     #         v.grad[unfiltered_indices_clone] = 0.0 #torch.tensor([0.0,0.0,0.0], device="cuda")

#     # params, variables = cat_vars_params_to_optimizer(filtered_indices_clone, new_params, params, variables, optimizer)

#     # num_pts = params['means3D'].shape[0]


#     # padded_grad = torch.zeros(num_pts, device="cuda")
#     # padded_grad[:grads.shape[0]] = grads
#     # padded_grad_mask = (padded_grad >= grad_thresh) & (padded_grad != 0.0)
#     # to_split = torch.logical_and(padded_grad_mask[indices],
#     #                                 torch.max(torch.exp(params['log_scales'][indices]), dim=1).values > 0.01 * variables[
#     #                                     'scene_radius'])
    
#     # filtered_indices_split = indices[to_split]

#     # n = 2  

#     # new_params = {}
#     # for k, v in params.items():
#     #     if k not in ['cam_m', 'cam_c']:
#     #         # First filter, then repeat
#     #         filtered_v = v[filtered_indices_split]
#     #         new_v = filtered_v.repeat(n, 1)
#     #         new_params[k] = new_v

#     #         # Copy gradients if they exist
#     #         if v.grad is not None:
#     #             # Filter the gradients, then repeat them too
#     #             filtered_grad = v.grad[filtered_indices_split]
#     #             new_v.grad = filtered_grad.repeat(n, 1)

#     # stds = torch.exp(params['log_scales'])[filtered_indices_split].repeat(n, 1)
#     # means = torch.zeros((stds.size(0), 3), device="cuda")
#     # samples = torch.normal(mean=means, std=stds)
#     # rots = build_rotation(params['unnorm_rotations'][filtered_indices_split]).repeat(n, 1, 1)
#     # new_params['means3D'] += torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
#     # new_params['log_scales'] = torch.log(torch.exp(new_params['log_scales']) / (0.8 * n))
#     # # new_params['log_scales'].retain_grad()
   
#     # params, variables= cat_vars_params_to_optimizer(filtered_indices_split, new_params, params, variables, optimizer, n=n)
#     # num_pts = params['means3D'].shape[0]


#     # variables['means2D_gradient_accum'] = torch.zeros(num_pts, device="cuda")
#     # variables['denom'] = torch.zeros(num_pts, device="cuda")
#     # variables['max_2D_radius'] = torch.zeros(num_pts, device="cuda")

#     # to_remove = torch.zeros(num_pts, dtype=torch.bool, device="cuda")
#     # to_remove[filtered_indices_split] = True
#     # # result = original_tensor[mask]

#     # params, variables = remove_points_with_vars(to_remove, params, variables, optimizer)
#     # indices_split= torch.where(torch.all(params["means3D"].grad != 0, dim=-1))[0]

#     # remove_threshold = 0.25 if i == 5000 else 0.005
#     # to_remove_mask = (torch.sigmoid(params['logit_opacities'][indices_split]) < remove_threshold).squeeze()
    
#     # big_points_ws = torch.exp(params['log_scales'][indices_split]).max(dim=1).values > 0.1 * variables['scene_radius']
#     # to_remove_mask = big_points_ws # torch.logical_or(to_remove_mask, big_points_ws)
#     # filtered_remove_indices = indices_split[to_remove_mask]
#     # num_pts = params['means3D'].shape[0]

#     # to_remove = torch.zeros(num_pts, dtype=torch.bool, device="cuda")
#     # to_remove[filtered_remove_indices] = True

#     # params, variables = remove_points_with_vars(to_remove, params, variables, optimizer)
    
#     final_indices= torch.where(torch.all(params["means3D"].grad != 0, dim=-1))[0]    

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

#     return params, variables, final_indices



