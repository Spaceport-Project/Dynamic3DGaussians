
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
import logging
import torch
import torch.nn.functional as func
from torch.autograd import Variable
from math import exp
from plyfile import PlyElement, PlyData 


logger = logging.getLogger(__name__)


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
    seen_mask = variables['seen']
    static_mask = variables.get('static_mask', None)

    if static_mask is not None:
        target_len = variables['means2D_gradient_accum'].shape[0]
        if static_mask.shape[0] < target_len:
            aligned_static = torch.zeros(target_len, dtype=torch.bool, device=static_mask.device)
            aligned_static[:static_mask.shape[0]] = static_mask
            static_mask = aligned_static
        elif static_mask.shape[0] > target_len:
            static_mask = static_mask[:target_len]

        seen_mask = torch.logical_and(seen_mask, ~static_mask)

    variables['means2D_gradient_accum'][seen_mask] += torch.norm(
        variables['means2D'].grad[seen_mask, :2], dim=-1)
    variables['denom'][seen_mask] += 1
    return variables

def accumulate_mean3d_gradient(variables):
    variables['means3D_gradient_accum'][variables['seen']] += torch.norm(
        variables['means3D'].grad[variables['seen'], :3], dim=-1)
    variables['denom'][variables['seen']] += 1
    return variables


def _align_static_mask_length(static_mask, target_len, device):
    if not isinstance(static_mask, torch.Tensor):
        static_mask = torch.tensor(static_mask, dtype=torch.bool, device=device)
    else:
        static_mask = static_mask.to(device=device, dtype=torch.bool)

    if static_mask.shape[0] < target_len:
        padded = torch.zeros(target_len, dtype=torch.bool, device=device)
        padded[:static_mask.shape[0]] = static_mask
        return padded
    if static_mask.shape[0] > target_len:
        return static_mask[:target_len]
    return static_mask


def _extend_static_mask_for_new_points(variables, target_len, device):
    static_mask = variables.get('static_mask', None)
    if static_mask is None:
        return
    variables['static_mask'] = _align_static_mask_length(static_mask, target_len, device)


def _align_static_confidence_length(static_confidence, target_len, device):
    if isinstance(static_confidence, torch.Tensor):
        static_confidence = static_confidence.to(device=device, dtype=torch.float32)
    else:
        static_confidence = torch.tensor(static_confidence, dtype=torch.float32, device=device)

    if static_confidence.shape[0] < target_len:
        padded = torch.zeros(target_len, dtype=torch.float32, device=device)
        padded[:static_confidence.shape[0]] = static_confidence
        return padded
    if static_confidence.shape[0] > target_len:
        return static_confidence[:target_len]
    return static_confidence


def _extend_static_confidence_for_new_points(variables, target_len, device):
    static_confidence = variables.get('static_confidence', None)
    if static_confidence is None:
        return
    variables['static_confidence'] = _align_static_confidence_length(static_confidence, target_len, device)


def _register_static_point_hooks(params, variables, freeze_keys=None):
    """
    Register backward hooks to suppress gradients for static points.
    Assumes static_mask is already aligned to current point count.
    """
    if freeze_keys is None:
        freeze_keys = ['means3D', 'f_dc', 'f_rest', 'unnorm_rotations', 'log_scales', 'logit_opacities']
    
    static_mask = variables.get('static_mask', None)
    if static_mask is None or 'means3D' not in params:
        return
    
    # Convert numpy array to tensor if needed
    if isinstance(static_mask, np.ndarray):
        static_mask = torch.tensor(static_mask, dtype=torch.bool, device=params['means3D'].device)
        variables['static_mask'] = static_mask
    
    # Clear any existing hooks first
    handles = variables.get('static_grad_hook_handles', None)
    if handles is not None:
        for handle in handles:
            try:
                handle.remove()
            except Exception:
                pass
    
    handles = []
    for key in freeze_keys:
        if key not in params:
            continue
        param = params[key]
        
        # Remove any existing hook on this parameter
        if hasattr(param, '_static_grad_hook_handle'):
            param._static_grad_hook_handle.remove()
        
        # Create gradient hook with current mask snapshot
        mask_snapshot = static_mask.clone()
        
        def grad_hook(grad, mask=mask_snapshot):
            if grad is None:
                return grad
            grad[mask] = 0
            return grad
        
        handle = param.register_hook(grad_hook)
        param._static_grad_hook_handle = handle
        handles.append(handle)
    
    variables['static_grad_hook_handles'] = handles


def _reindex_static_mask_after_filter(variables, to_keep):
    """
    Reindex static mask after point filtering.
    Expects mask to be already aligned to current point count before filtering.
    If not aligned, will align it first as a safety fallback.
    """
    static_mask = variables.get('static_mask', None)
    if static_mask is None:
        return
    
    # Safety check: if mask size doesn't match, align it
    if static_mask.shape[0] != to_keep.shape[0]:
        static_mask = _align_static_mask_length(static_mask, to_keep.shape[0], to_keep.device)
    
    # Reindex with the to_keep filter
    variables['static_mask'] = static_mask[to_keep]


def _reindex_static_confidence_after_filter(variables, to_keep):
    static_confidence = variables.get('static_confidence', None)
    if static_confidence is None:
        return

    if not isinstance(static_confidence, torch.Tensor):
        static_confidence = torch.tensor(static_confidence, dtype=torch.float32, device=to_keep.device)
    else:
        static_confidence = static_confidence.to(device=to_keep.device, dtype=torch.float32)

    if static_confidence.shape[0] != to_keep.shape[0]:
        static_confidence = _align_static_confidence_length(static_confidence, to_keep.shape[0], to_keep.device)

    variables['static_confidence'] = static_confidence[to_keep]



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


def update_params_and_optimizer2(new_params, params, indices, optimizer):
    for k, v in new_params.items():
        group = [x for x in optimizer.param_groups if x["name"] == k][0]
        old_param = group['params'][0]
        stored_state = optimizer.state.get(old_param, None)
        
        # Create a copy of the old parameter
        updated_param = old_param.data.clone()
        
        # Update only the specified indices with corresponding values from new_params
        # v has the same length as indices, so v[i] goes to position indices[i]
        updated_param[indices] = v
        
        # Update optimizer state for the specified indices
        if stored_state is not None:
            # Reset momentum only for updated indices
            stored_state["exp_avg"][indices] = 0
            stored_state["exp_avg_sq"][indices] = 0
            
            # Remove old state
            del optimizer.state[old_param]
        
        # Create new parameter with updated values
        new_param = torch.nn.Parameter(updated_param.requires_grad_(True))
        group["params"][0] = new_param
        
        # Restore optimizer state with new parameter
        if stored_state is not None:
            optimizer.state[new_param] = stored_state
        
        params[k] = new_param
    
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

    if 'means3D' in params:
        _extend_static_mask_for_new_points(variables, params['means3D'].shape[0], params['means3D'].device)
        _extend_static_confidence_for_new_points(variables, params['means3D'].shape[0], params['means3D'].device)


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
    
    # variables['prev_inv_rot_fg'] = variables['prev_inv_rot_fg'][to_keep]
    # variables['prev_offset'] = variables['prev_offset'][to_keep]
    # variables["prev_pts"] =  variables['prev_pts'][to_keep]
    # variables["prev_rot"] = variables['prev_rot'][to_keep]

    # variables["prev_col"] = variables['prev_col'][to_keep]
    # variables["neighbor_indices"] = variables['neighbor_indices'][to_keep]
    # variables["neighbor_weight"] = variables['neighbor_weight'][to_keep]
    # variables["neighbor_dist"] = variables['neighbor_dist'][to_keep]


    variables['means2D_gradient_accum'] = variables['means2D_gradient_accum'][to_keep]
    variables['denom'] = variables['denom'][to_keep]
    variables['max_2D_radius'] = variables['max_2D_radius'][to_keep]
    _reindex_static_mask_after_filter(variables, to_keep)
    # _reindex_static_confidence_after_filter(variables, to_keep)
    return params, variables

def mask_points(to_remove, params, variables, optimizer):
    """
    Masks points by setting them to zero and disabling gradient computation.
    """
    keys = [k for k in params.keys() if k not in ['cam_m', 'cam_c']]
    
    # Initialize or expand the mask to match current number of points
    current_num_points = to_remove.shape[0]
    
    if 'point_mask' not in variables:
        # First time: create mask with all points active
        variables['point_mask'] = torch.ones(current_num_points, dtype=torch.bool, device=to_remove.device)
    elif variables['point_mask'].shape[0] < current_num_points:
        # Points were added: expand mask (new points are active by default)
        num_new_points = current_num_points - variables['point_mask'].shape[0]
        new_mask = torch.ones(num_new_points, dtype=torch.bool, device=to_remove.device)
        variables['point_mask'] = torch.cat([variables['point_mask'], new_mask], dim=0)
    
    # Now mask the points to remove
    variables['point_mask'][to_remove] = False
    
    for k in keys:
        group = [g for g in optimizer.param_groups if g['name'] == k][0]
        stored_state = optimizer.state.get(group['params'][0], None)
        
        # Mask optimizer state
        if stored_state is not None:
            stored_state["exp_avg"][to_remove] = 0
            stored_state["exp_avg_sq"][to_remove] = 0
        
        # Mask the parameter values
        with torch.no_grad():
            group["params"][0][to_remove] = float('nan')  # or 0, depending on how you want to handle masked points
        
        # Register gradient hook
        param = group["params"][0]
        
        if hasattr(param, '_grad_hook_handle'):
            param._grad_hook_handle.remove()
        
        def grad_hook(grad, mask=variables['point_mask']):
            grad[~mask] = 0
            return grad
        
        param._grad_hook_handle = param.register_hook(grad_hook)
    
    # Mask variables
    variables['means2D_gradient_accum'][to_remove] = 0
    variables['denom'][to_remove] = 0
    variables['max_2D_radius'][to_remove] = 0
    
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
    _reindex_static_mask_after_filter(variables, to_keep)
    # _reindex_static_confidence_after_filter(variables, to_keep)
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
    _reindex_static_mask_after_filter(variables, to_keep)
    _reindex_static_confidence_after_filter(variables, to_keep)
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
    if variables.get('static_mask', None) is not None:
        variables['static_mask'] = _align_static_mask_length(
            variables['static_mask'],
            num_pts,
            params['means3D'].device
        )
    # if variables.get('static_confidence', None) is not None:
    #     variables['static_confidence'] = _align_static_confidence_length(
    #         variables['static_confidence'],
    #         num_pts,
    #         params['means3D'].device
    #     )
    return params, variables


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))
def np_inverse_sigmoid(x):
    x_clipped = np.clip(x, 1e-7, 1 - 1e-7)  
    return np.log(x_clipped / (1 - x_clipped))
def sigmoid(z):
    return 1/(1 + np.exp(-z))



def densify_nokeyframes(params, variables, optimizer, i, max_num_splat):
    if i <= 500:
        variables = accumulate_mean2d_gradient(variables)
        grad_thresh = 0.0002
        if (i >= 200) and (i % 100 == 0):
            grads = variables['means2D_gradient_accum'] / variables['denom']
            grads[grads.isnan()] = 0.0
            
            if params['means3D'].shape[0] < max_num_splat:
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
            # new_params = {k: v[to_split].repeat(n, 1) for k, v in params.items() if k not in ['cam_m', 'cam_c']}

            new_params = {}
            for k, v in params.items():
                if k not in ['cam_m', 'cam_c']:
                    if k in ['f_dc', 'f_rest']:
                       new_params[k] = v[to_split].repeat(n, 1, 1)
                    else:
                       new_params[k] = v[to_split].repeat(n, 1)
                       
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


def densify_nokeyframes_dev(params, variables, optimizer, i, max_num_splat):

    if i <= 600:
        variables = accumulate_mean2d_gradient(variables)
        grad_thresh = 0.0002
        if (i >= 200) and (i % 100 == 0):
            grads = variables['means2D_gradient_accum'] / variables['denom']
            grads[grads.isnan()] = 0.0


            # (params['means3D'] - variables['prev_pts']).norm(dim=-1) <= 
            # xyz = [ tuple(v for v in ver ) for ver in new_params['means3D'].detach().cpu().numpy()]
            # vertex=np.array(xyz, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
            # # if i == 5400:

            # el = PlyElement.describe(vertex, 'vertex')

            # PlyData([el]).write(f"to_clone_{i:03d}.ply")

            
            if params['means3D'].shape[0] < max_num_splat  or i == 300:
                
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
            n = 2 
            


            new_params = {}
            for k, v in params.items():
                if k not in ['cam_m', 'cam_c']:
                    if k in ['f_dc', 'f_rest']:
                        new_params[k] = v[to_split].repeat(n, 1, 1)
                    else:
                        new_params[k] = v[to_split].repeat(n, 1)
                    
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
            
            params, variables = mask_points(to_remove, params, variables, optimizer)

            remove_threshold = 0.25 if i == 600 else 0.005
            to_remove = (torch.sigmoid(params['logit_opacities']) < remove_threshold).squeeze()
            if i >= 300:
                big_points_ws = torch.exp(params['log_scales']).max(dim=1).values > 0.1 * variables['scene_radius']
                to_remove = torch.logical_or(to_remove, big_points_ws)
            
            # accumulated_indices = torch.unique(torch.cat([accumulated_indices, new_indices2]))  
            params, variables = mask_points(to_remove, params, variables, optimizer)
        
        

            torch.cuda.empty_cache()

        if i > 0 and i % 300 == 0:
            new_params = {'logit_opacities': inverse_sigmoid(torch.ones_like(params['logit_opacities']) * 0.01)}
            params = update_params_and_optimizer(new_params, params, optimizer)
        
        


    return params, variables

def densify_nokeyframes_v2(params, variables, optimizer, i, max_num_splat):

    if i <= 600:
        variables = accumulate_mean2d_gradient(variables)
        grad_thresh = 0.0002
        if (i >= 200) and (i % 100 == 0):
            grads = variables['means2D_gradient_accum'] / variables['denom']
            grads[grads.isnan()] = 0.0


            
            if params['means3D'].shape[0] < max_num_splat  or i == 300:
                
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
            n = 2 
            


            new_params = {}
            for k, v in params.items():
                if k not in ['cam_m', 'cam_c']:
                    if k in ['f_dc', 'f_rest']:
                        new_params[k] = v[to_split].repeat(n, 1, 1)
                    else:
                        new_params[k] = v[to_split].repeat(n, 1)
                    
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
            
            params, variables = mask_points(to_remove, params, variables, optimizer)

            remove_threshold = 0.25 if i == 600 else 0.005
            to_remove = (torch.sigmoid(params['logit_opacities']) < remove_threshold).squeeze()
            if i >= 300:
                big_points_ws = torch.exp(params['log_scales']).max(dim=1).values > 0.1 * variables['scene_radius']
                to_remove = torch.logical_or(to_remove, big_points_ws)
            
            params, variables = mask_points(to_remove, params, variables, optimizer)
        
        

            torch.cuda.empty_cache()

        if i > 0 and i % 300 == 0:
            new_params = {'logit_opacities': inverse_sigmoid(torch.ones_like(params['logit_opacities']) * 0.01)}
            params = update_params_and_optimizer(new_params, params, optimizer)
        
        


    return params, variables

def densify(params, variables, optimizer, i, max_num_splat=999999):
    if i <= 6000:
        variables = accumulate_mean2d_gradient(variables)
        grad_thresh = 0.0002
        if (i >= 500) and (i % 100 == 0):
            grads = variables['means2D_gradient_accum'] / variables['denom']
            grads[grads.isnan()] = 0.0
            
            if params['means3D'].shape[0] < max_num_splat:
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
            # new_params = {k: v[to_split].repeat(n, 1) for k, v in params.items() if k not in ['cam_m', 'cam_c'] }
            new_params = {}
            for k, v in params.items():
                if k not in ['cam_m', 'cam_c']:
                    if k in ['f_dc', 'f_rest']:
                       new_params[k] = v[to_split].repeat(n, 1, 1)
                    else:
                       new_params[k] = v[to_split].repeat(n, 1)
            
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
            if i >= 4000:
                big_points_ws = torch.exp(params['log_scales']).max(dim=1).values > 0.1 * variables['scene_radius']
                to_remove = torch.logical_or(to_remove, big_points_ws)
            params, variables = remove_points(to_remove, params, variables, optimizer)
           
            # xyz = [ tuple(v for v in ver ) for ver in params['means3D'].detach().cpu().numpy()]
            # vertex=np.array(xyz, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
            # # if i == 5400:

            # el = PlyElement.describe(vertex, 'vertex')

            # PlyData([el]).write(f"params_{i:03d}.ply")

            torch.cuda.empty_cache()

        if i > 0 and i % 4000 == 0:
            new_params = {'logit_opacities': inverse_sigmoid(torch.ones_like(params['logit_opacities']) * 0.01)}
            params = update_params_and_optimizer(new_params, params, optimizer)
        
       


    return params, variables


def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper



