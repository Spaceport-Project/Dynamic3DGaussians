from typing import NamedTuple
import kornia
import torch
import os
import open3d as o3d
import numpy as np
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera
from pytorch3d.ops import ball_query, knn_points
import torchvision
import torchvision.transforms.functional as F

from external import np_inverse_sigmoid  

from plyfile import PlyData, PlyElement

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array


def searchForMaxIteration(folder):
    if os.path.exists(os.path.join(folder,'params.npz')):
        return None
    try:
        saved_iters = [int(os.path.splitext(fname.split("_")[-1])[0]) for fname in os.listdir(folder)  if fname.startswith('params_')]
    except FileNotFoundError:
        return None
    return max(saved_iters)

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))
def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top  
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def getWorld2View(w2c):

    Rt = np.zeros((4, 4))

    Rt[:3, :3] = w2c[:3, :3].transpose()
    Rt[:3, 3] = w2c[:3, 3]
    Rt[3, 3] = 1.0

    return np.float32(Rt)

def setup_camera(w, h, k, w2c, near=0.01, far=100, scale=1, bg=torch.tensor([0, 0, 0])):
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    w2c = torch.tensor(w2c).cuda().float()
    cam_center = torch.inverse(w2c)[:3, 3]
    w2c = w2c.unsqueeze(0).transpose(1, 2)
    
    fx/=scale
    fy/=scale
    cx/=scale
    cy/=scale
    w/=scale
    h/=scale
    # FovY = focal2fov(fy, h)
    # FovX = focal2fov(fx, w)
    # opengl_proj = getProjectionMatrix(znear=near, zfar=far, fovX=FovX, fovY=FovY).cuda().float().unsqueeze(0).transpose(1, 2)
    opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
                                [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                                [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                                [0.0, 0.0, 1.0, 0.0]]).cuda().float().unsqueeze(0).transpose(1, 2)
    full_proj = w2c.bmm(opengl_proj)
    cam = Camera(
        image_height=int(h),
        image_width=int(w),
        tanfovx=w / (2 * fx),
        tanfovy=h / (2 * fy),
        bg=  bg.cuda().float(), #torch.tensor([0, 177./255, 64.0/255 dtype=torch.float32, device="cuda"), #
        scale_modifier=1.0,
        viewmatrix=w2c,
        projmatrix=full_proj,
        sh_degree=0,
        campos=cam_center,
        prefiltered=False,
        # k=k
        # debug=False,
        # antialiasing=False
    )
    return cam


def params2rendervar(params):
    rendervar = {
        'means3D': params['means3D'],
        'colors_precomp': torch.sigmoid(params['rgb_colors']),
        'rotations': torch.nn.functional.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(params['log_scales']),
        'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0
    }
    return rendervar


def l1_loss_v1_noblack(x, y, mask):
    # mask = torch.any(y != 0, dim=0)
    fil_y = y[:, mask]
    fil_x = x[:, mask ]
    return torch.abs((fil_x - fil_y)).mean()

def l1_loss_v1(x, y):
    return torch.abs((x - y)).mean()

def l2_loss_v1(x, y):
    return torch.sqrt((torch.pow((x - y), 2) + 1e-20).mean())

def l2_loss_v2(x, y):
    return torch.sqrt((torch.pow((x - y), 2).sum(-1) + 1e-20).mean())

def l1_loss_v2(x, y):
    return (torch.abs(x - y).sum(-1)).mean()


def weighted_l2_loss_v1(x, y, w):
    return torch.sqrt(((x - y) ** 2) * w + 1e-20).mean()

def combined_color_loss_v1(current_colors, prev_colors, gray_threshold=0.1, factor=5.0):
    rgb_std = torch.std(current_colors, dim=-1)
    gray_mask = rgb_std < gray_threshold
    
    # L1 for non-gray, L2 for gray
    non_gray_loss = torch.pow(current_colors[~gray_mask] - prev_colors[~gray_mask], 2).mean()
    gray_loss = torch.pow(current_colors[gray_mask] - prev_colors[gray_mask], 2).mean() * factor
    
    return non_gray_loss + gray_loss

def combined_color_loss_v2(current_colors, prev_colors, gray_threshold=0.1, factor=5.0):
    rgb_std = torch.std(current_colors, dim=-1)
    gray_mask = rgb_std < gray_threshold
    
    # L1 for non-gray, L2 for gray
    non_gray_loss = torch.pow(current_colors[~gray_mask] - prev_colors[~gray_mask], 2).sum(-1).mean()
    gray_loss = torch.pow(current_colors[gray_mask] - prev_colors[gray_mask], 2).sum(-1).mean() * factor
    
    return non_gray_loss + gray_loss

def separate_gray_loss(x, y, threshold=0.1):
    rgb_std = torch.std(x, dim=-1)
    gray_mask = rgb_std < threshold
    
    if gray_mask.sum() > 0:
        xg = x[gray_mask]
        yg = y[gray_mask]
        # gray_loss = torch.sqrt((torch.pow((xg - yg), 2) + 1e-20).mean())
        gray_loss =  (torch.abs(xg - yg)).mean()
        return gray_loss
    return torch.tensor(0.0, device=x.device)

def convert_rgb_to_hsv(x):
    img = x.unsqueeze(0).permute(0, 2, 1).unsqueeze(-1)  # (1, 3, N, 1)  
    # Convert RGB to HSV
    hsv = kornia.color.rgb_to_hsv(img)
    
    return hsv

def separate_gray_loss_hsv(x, y, threshold=0.1):
   
   # Ensure img has batch dimension
  
    img = x.unsqueeze(0).permute(0, 2, 1).unsqueeze(-1)  # (1, 3, N, 1)  
    # Convert RGB to HSV
    hsv = kornia.color.rgb_to_hsv(img)
    
    # Extract saturation channel (index 1)
    saturation = hsv[:, 1, :, :].squeeze()
    # Create gray mask (low saturation = gray)
    gray_mask = saturation < threshold
    
    if gray_mask.sum() > 0:
        xg = x[gray_mask]
        yg = y[gray_mask]
        # gray_loss = torch.sqrt((torch.pow((xg - yg), 2) + 1e-20).mean())
        gray_loss =  (torch.abs(xg - yg)).mean()
        return gray_loss
    return torch.tensor(0.0, device=x.device)

def separate_gray_loss_hsv_smooth(x, y, neighbor_indices, threshold=0.1):
   
   # Ensure img has batch dimension
  
    img = x.unsqueeze(0).permute(0, 2, 1).unsqueeze(-1)  # (1, 3, N, 1)  
    # Convert RGB to HSV
    hsv = kornia.color.rgb_to_hsv(img)
    
    # Extract saturation channel (index 1)
    saturation = hsv[:, 1, :, :].squeeze()
    # Create gray mask (low saturation = gray)
    gray_mask = (saturation < threshold) & (saturation > 0.01)

    
    if gray_mask.sum() > 0:
        # xg = x[gray_mask]
        # yg = y[gray_mask]
        neighbor_indices_g = neighbor_indices[gray_mask]
        neighbor_xg = x[neighbor_indices_g]
        neighbor_yg = y[neighbor_indices_g]
        smoothed_xg = torch.mean(neighbor_xg, dim=1)
        smoothed_yg = torch.mean(neighbor_yg, dim=1)
        # gray_loss = torch.sqrt((torch.pow((xg - yg), 2) + 1e-20).mean())
        gray_loss =  (torch.abs(smoothed_xg-  smoothed_yg)).mean()
        # gray_loss = torch.sqrt((torch.pow((xg - yg), 2) + 1e-20).mean())
        return gray_loss
    return torch.tensor(0.0, device=x.device)

def separate_gray_loss_lab(x, y, threshold=10):
    img = x.unsqueeze(0).permute(0, 2, 1).unsqueeze(-1)  # (1, 3, N, 1)  
    # Convert RGB to HSV
    lab = kornia.color.rgb_to_lab(img)


    # Extract a and b channels
    L = lab[:, 0, :, :].squeeze()
    # a = lab[:, 1, :, :].squeeze()  # Shape: (N,)
    # b = lab[:, 2, :, :].squeeze()  # Shape: (N,)
    mask_L =  (L <= 100)  & (L > 50 ) 
    # mask_ab = (torch.abs(a) < threshold) & (torch.abs(b) < threshold)
    # Calculate the Euclidean distance from the gray point (a=0, b=0)
    # distance_from_gray = torch.sqrt(a**2 + b**2)

    # Create gray mask (low distance = gray)
    # gray_mask = distance_from_gray < threshold
    gray_mask = mask_L #& mask_ab
    if gray_mask.sum() > 0:
        xg = x[gray_mask]
        yg = y[gray_mask]
        # gray_loss = torch.sqrt((torch.pow((xg - yg), 2) + 1e-20).mean())
        gray_loss =  (torch.abs(xg - yg)).mean()
        return gray_loss
    return torch.tensor(0.0, device=x.device)


def separate_gray_loss_lab_smooth(x, y, neighbor_indices, threshold=10):
    img = x.unsqueeze(0).permute(0, 2, 1).unsqueeze(-1)  # (1, 3, N, 1)  
    # Convert RGB to HSV
    lab = kornia.color.rgb_to_lab(img)

    # Extract a and b channels
    a = lab[:, 1, :, :].squeeze()  # Shape: (N,)
    b = lab[:, 2, :, :].squeeze()  # Shape: (N,)

    # Calculate the Euclidean distance from the gray point (a=0, b=0)
    distance_from_gray = torch.sqrt(a**2 + b**2)

    # Create gray mask (low distance = gray)
    gray_mask = distance_from_gray < threshold
    if gray_mask.sum() > 0:
        # xg = x[gray_mask]
        # yg = y[gray_mask]
        neighbor_indices_g = neighbor_indices[gray_mask]
        neighbor_xg = x[neighbor_indices_g]
        neighbor_yg = y[neighbor_indices_g]
        smoothed_xg = torch.mean(neighbor_xg, dim=1)
        smoothed_yg = torch.mean(neighbor_yg, dim=1)
        # gray_loss = torch.sqrt((torch.pow((xg - yg), 2) + 1e-20).mean())
        gray_loss =  (torch.abs(smoothed_xg-  smoothed_yg)).mean()
        return gray_loss
    return torch.tensor(0.0, device=x.device)
def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def weighted_l2_loss_v2(x, y, w):
    return torch.sqrt(((x - y) ** 2).sum(-1) * w + 1e-20).mean()


def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1.T
    w2, x2, y2, z2 = q2.T
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z]).T


def o3d_knn(pts, num_knn):
    indices = []
    sq_dists = []
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, np.float64))
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    for p in pcd.points:
        [_, i, d] = pcd_tree.search_knn_vector_3d(p, num_knn + 1)
        indices.append(i[1:])
        sq_dists.append(d[1:])
    return np.array(sq_dists), np.array(indices)

#
def pytorch3d_knn(pts, num_knn):
    """
    pts: (N, 3) tensor
    """
    # Make sure no grad is tracked
    with torch.no_grad():
        # Add batch dim: (1, N, 3)
        pts_b = pts.unsqueeze(0)

        # Single KNN call for all points
        dists, idx, _ = knn_points(pts_b, pts_b, K=num_knn + 1)  # (1, N, K)

        # Remove self-neighbor (index 0) and batch dim
        dists = dists[0, :, 1:]  # (N, num_knn)
        idx   = idx[0, :, 1:]    # (N, num_knn)

    return dists, idx

def params2cpu2(params, is_initial_timestep):
    if is_initial_timestep:
        res = {k: v.detach().cpu().contiguous().numpy() for k, v in params.items()}
    else:
        res = {k: v.detach().cpu().contiguous().numpy() for k, v in params.items() if
               k in ['means3D', 'rgb_colors', 'unnorm_rotations']}
    return res

def params2cpu(params, is_initial_timestep):
    if is_initial_timestep:
        res = {k: v.detach().cpu().contiguous().numpy() for k, v in params.items() if k not in ['cam_m', 'cam_c']}
    else:
        res = {k: v.detach().cpu().contiguous().numpy() for k, v in params.items() if
               k in ['means3D', 'rgb_colors', 'unnorm_rotations']}
    return res

def save_params(output_params, seq, exp, t):
    # if iter != "":
    #     iter=f"_{iter}"

    points = output_params[-1]["means3D"]
    # num_points_prev = len(points)
    colors = output_params[-1]["rgb_colors"]
    seg = np.ones_like(points[:, 0])[:, None] 
    pt_cld = dict()
    colors = np_inverse_sigmoid(colors)
    pt_cld = np.concatenate((points, colors, seg), axis=1).tolist()
    np.savez(f"./data/{seq}/init_pt_cld_{t}.npz", data=pt_cld)
    to_save = {}
    for k in output_params[0].keys():
        if k in output_params[1].keys():
            to_save[k] = np.stack([params[k] for params in output_params])
        else:
            to_save[k] = output_params[0][k]
    os.makedirs(f"./output/{exp}/{seq}", exist_ok=True)
    np.savez(f"./output/{exp}/{seq}/params", **to_save)

def save_params_single(output_params, seq, exp, ind):
  
    to_save = {}
    for k in output_params.keys():
        
        # to_save[k] = np.stack([params[k] for params in output_params])
       
        to_save[k] = output_params[k]

    os.makedirs(f"./output/{exp}/{seq}", exist_ok=True)
    np.savez(f"./output/{exp}/{seq}/params_{ind}", **to_save)

iteration = 0

def save_params_multi(output_params, seq, exp, index, t):
    points = output_params[-1]["means3D"]
    # num_points_prev = len(points)
    colors = output_params[-1]["rgb_colors"]
    seg = np.ones_like(points[:, 0])[:, None] 
    pt_cld = dict()
    colors = np_inverse_sigmoid(colors)
    pt_cld = np.concatenate((points, colors, seg), axis=1).tolist()
    np.savez(f"./data/{seq}/init_pt_cld_{t}.npz", data=pt_cld)
  
    global iteration
   
    k = 0
    ranges = []
    for i in range(len(output_params)):
        if i < k :
            continue
        
        to_save = {}
        if i+1 < len(output_params):
            for k in range(i+1,len(output_params)):
                # if 'log_scales' in output_params[k]: 
                if output_params[i]["means3D"].shape[0] != output_params[k]["means3D"].shape[0]:
                    break
        
            for l in output_params[i].keys():
                if l in output_params[i+1].keys():
                    to_save[l] = np.stack([output_params[s][l] for s in range(i,k)])
                else:
                    to_save[l] = output_params[i][l]
            os.makedirs(f"./output/{exp}/{seq}", exist_ok=True)
            np.savez(f"./output/{exp}/{seq}/params_{iteration}", **to_save)
            
            if index > iteration:
                ranges.append((i,k))
               
                iteration += 1
        
    for ran in reversed(ranges):
        for s in range(*ran):
            for key in list(output_params[s].keys()): 
                del output_params[s][key]
    
    if len(ranges) > 0:
        for it in range(ranges[-1][1]- 1, -1, -1):
            del output_params[it]
    torch.cuda.empty_cache()  

def matching_point_clouds(pc2, pc1):
            # Compute pairwise distances: (N2, N1)
    dist = torch.cdist(pc2, pc1)

    N2, N1 = dist.shape
    used_pc1 = torch.zeros(N1, dtype=torch.bool, device=dist.device)

    matched_indices = torch.empty(N2, dtype=torch.long, device=dist.device)

    for i in range(N2):
        # Get distances from pc2[i] to all pc1
        d = dist[i].clone()

        # Mask out already used pc1 points by setting distance to +inf
        d[used_pc1] = float('inf')

        # Pick the closest unused pc1 point
        j = torch.argmin(d)
        matched_indices[i] = j
        used_pc1[j] = True

    # Now get the matched pc1 points; shape: (N2, 3)
    return matched_indices
def greedy_one_to_many_from_knn(pc1, pc2):
    with torch.no_grad():
        pc1_b = pc1.unsqueeze(0)
        pc2_b = pc2.unsqueeze(0)

        dists, idx, _ = knn_points(pc2_b, pc1_b, K=1)  # idx: (1, N2, 1)
        idx_pc1_for_pc2 = idx[0, :, 0]                 # (N2,)

        unique_pc1_idx = torch.unique(idx_pc1_for_pc2) # (K,)
    return unique_pc1_idx

def greedy_one_to_one_from_knn(pc1, pc2, k_neighbors=5):
    """
    Greedy 1-to-1 matching from pc2 -> pc1 using pytorch3d.knn_points
    Uses top-k neighbors as candidates, then picks unused ones greedily.
    """
    pc1_b = pc1.unsqueeze(0)  # (1, N1, 3)
    pc2_b = pc2.unsqueeze(0)  # (1, N2, 3)

    # Get K nearest neighbors for each pc2 point in pc1
    dists, idx, _ = knn_points(pc2_b, pc1_b, K=k_neighbors)  # idx: (1, N2, K)
    idx = idx[0]        # (N2, K)
    dists = dists[0]    # (N2, K)

    N1 = pc1.shape[0]
    N2 = pc2.shape[0]
    device = pc1.device

    used_pc1 = torch.zeros(N1, dtype=torch.bool, device=device)
    matched_idx = torch.empty(N2, dtype=torch.long, device=device)

    for i in range(N2):
        # go through the K nearest neighbors in order of increasing distance
        row_idx = idx[i]      # (K,)
        for j in range(k_neighbors):
            cand = row_idx[j].item()
            if not used_pc1[cand]:
                matched_idx[i] = cand
                used_pc1[cand] = True
                break
        else:
            # all k candidates already used → fallback: allow reuse of nearest
            # (or you could search again with larger K)
            matched_idx[i] = row_idx[0]
            # no change to used_pc1 here if you allow reuse

    # pc1_matched = pc1[matched_idx]  # (N2, 3)
    return matched_idx #, pc1_matched

def greedy_match_large_clouds(pc1, pc2, batch_size=1024):
    """
    Performs a greedy 1-to-1 matching from pc2 to pc1,
    avoiding full distance matrix computation for large point clouds.

    Args:
        pc1 (torch.Tensor): Point cloud 1, shape (N1, 3).
        pc2 (torch.Tensor): Point cloud 2, shape (N2, 3).
        batch_size (int): How many points from pc2 to process at once.
                          Adjust based on your GPU memory.

    Returns:
        torch.Tensor: Indices in pc1 corresponding to the matched points for pc2.
                      Shape (N2,).
    """
    N1 = pc1.shape[0]
    N2 = pc2.shape[0]
    device = pc1.device

    # Ensure N1 >= N2 for a meaningful 1-to-1 matching from pc2 to pc1
    if N1 < N2:
        print("Warning: pc1 has fewer points than pc2. Some pc2 points will not find a unique match.")
        # You might want to handle this case differently, e.g., by padding pc1 or raising an error.

    used_pc1 = torch.zeros(N1, dtype=torch.bool, device=device)
    matched_indices = torch.empty(N2, dtype=torch.long, device=device)

    # Process pc2 in batches
    for i_start in range(0, N2, batch_size):
        i_end = min(i_start + batch_size, N2)
        pc2_batch = pc2[i_start:i_end]
        batch_size_actual = pc2_batch.shape[0]

        # Compute distances for this batch of pc2 points to ALL pc1 points
        # This still creates a (batch_size_actual, N1) matrix, which should be manageable.
        # For N1=100k, batch_size=1024, this is (1024, 100000) = 10^8 elements = 400MB.
        # This should fit in most GPUs.
        batch_distances = torch.cdist(pc2_batch, pc1) # Shape: (batch_size_actual, N1)

        # Iterate through each point in the current pc2 batch
        for k in range(batch_size_actual):
            current_pc2_idx = i_start + k
            d = batch_distances[k].clone()

            # Mask out already used pc1 points
            d[used_pc1] = float('inf')

            # Find the closest unused pc1 point
            # Check if all points are used (can happen if N1 < N2 or many collisions)
            if torch.isinf(d).all():
                # No available pc1 point for this pc2 point.
                # Assign a dummy index or handle as an unmatched point.
                # For now, we'll assign -1, which will cause an error if used to index pc1.
                # You might want to assign the closest available, even if used, or skip.
                matched_indices[current_pc2_idx] = -1 # Or some other indicator
                print(f"Warning: pc2 point {current_pc2_idx} could not find a unique match in pc1.")
            else:
                j = torch.argmin(d)
                matched_indices[current_pc2_idx] = j
                used_pc1[j] = True

    return matched_indices