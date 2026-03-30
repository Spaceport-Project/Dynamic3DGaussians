# train_parameterized.py
"""
Parameterized training script for dynamic 3D Gaussian Splatting.

This script trains a 3D Gaussian Splatting model for dynamic scenes with temporal consistency.
It supports multi-view reconstruction with segmentation masks and temporal regularization.
"""

from dataclasses import dataclass  

from sklearn.neighbors import NearestNeighbors
import torch
import os
import json
import copy
import numpy as np
from PIL import Image, ImageFile
import torchvision

from utils.sh_utils import RGB2SH
ImageFile.LOAD_TRUNCATED_IMAGES = True

from random import randint
from tqdm import tqdm
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera
from diff_gaussian_rasterization import SparseGaussianAdam

from helpers import  fetchPly,  l1_loss_v1_noblack, params2rendervarseg, params2rendervarshsaccel, \
    pytorch3d_knn, save_params_to_ply_shs, setup_camera, l1_loss_v1,  weighted_l2_loss_v1, weighted_l2_loss_v2, quat_mult, \
    o3d_knn
from external import  calc_psnr, build_rotation,  densify, densify_nokeyframes_v2, get_expon_lr_func
import shutil
import open3d as o3d
from fused_ssim import fused_ssim
from torch.optim.lr_scheduler import *  
from plyfile import PlyElement, PlyData 
import threading
from queue import Queue
from config import TrainingConfig


# Utility functions for converting tensors to 8-bit images
to8b = lambda x : (255*np.clip(x.permute(1,2,0).cpu().detach().numpy(),0,1)).astype(np.uint8)
to8b_noper = lambda x : (255*np.clip(x.cpu().detach().numpy(),0,1)).astype(np.uint8)


class AsyncDatasetLoader:
    """
    Asynchronous dataset loader for efficient data loading during training.
    
    This class loads datasets in the background while training is ongoing,
    reducing I/O bottlenecks and improving training speed. It uses a queue
    to maintain a buffer of pre-loaded datasets.
    
    Attributes:
        md: Metadata dictionary containing camera parameters and file paths
        seq: Sequence name/identifier
        config: Training configuration object
        start_t: Starting timestep for loading
        queue: Thread-safe queue for storing loaded datasets
        current_t: Current timestep being processed
        initialized: Flag indicating if the loader has been initialized
        total_iterations: Total number of timesteps to process
        lock: Threading lock for thread-safe operations
    """
    
    def __init__(self, md, seq, config: TrainingConfig, start_t, total_iterations):
        """
        Initialize the async dataset loader.
        
        Args:
            md: Metadata dictionary with camera and image information
            seq: Sequence identifier
            config: Training configuration object
            start_t: Starting timestep
            total_iterations: Total number of timesteps in the sequence
        """
        self.md = md
        self.seq = seq
        self.config = config
        self.start_t = start_t
        self.queue = Queue(maxsize=config.async_queue_size)
        self.current_t = start_t
        self.initialized = False
        self.total_iterations = total_iterations
        self.lock = threading.Lock()
        
    def load_dataset(self, t, iters, active_sh_degree):
        """
        Load a single dataset for a specific timestep.
        
        Args:
            t: Timestep to load
            iters: List of future timesteps for handling missing images
            active_sh_degree: Current spherical harmonics degree
            
        Returns:
            Loaded dataset containing images, segmentation masks, and camera parameters
        """
        bg = torch.tensor(self.config.bg_color)
        dataset = get_dataset(t, self.md, self.seq, iters, active_sh_degree=active_sh_degree, 
                            scale=self.config.scale, bg=bg, config=self.config)
        return dataset
    
    def _async_load_worker(self, t, iters, active_sh_degree):
        """
        Worker function that runs in a separate thread to load datasets asynchronously.
        
        Args:
            t: Timestep to load
            iters: List of future timesteps
            active_sh_degree: Current spherical harmonics degree
        """
        if t < self.total_iterations:
            dataset = self.load_dataset(t, iters, active_sh_degree)
            if dataset is not None:
                self.queue.put((t, dataset))
                print(f"Loaded dataset {t}")
        
    
    def get_next_dataset(self, iters, active_sh_degree):
        """
        Get the next dataset from the queue and trigger async loading of the following one.
        
        On first call, loads the initial dataset synchronously. Subsequent calls retrieve
        pre-loaded datasets from the queue and trigger background loading of the next dataset.
        
        Args:
            iters: List of future timesteps
            active_sh_degree: Current spherical harmonics degree
            
        Returns:
            Tuple of (timestep, dataset) or (None, None) if queue is empty
        """
        # First iteration: load first dataset synchronously
        if not self.initialized:
            print("First iteration: Loading first dataset...")
            for i in range(1):
                t = self.start_t + i
                dataset = self.load_dataset(t, iters, active_sh_degree)
                self.queue.put((t, dataset))
                print(f"Loaded dataset {t}")
            self.current_t = self.start_t + 1
            self.initialized = True
        
        # Get dataset from queue
        if self.queue.empty():
            print("Queue is empty, Has reached the end of the dataset...")
            return None, None
        
        t, dataset = self.queue.get()
        
        # Asynchronously load next dataset in background thread
        next_t = self.current_t
        self.current_t += 1
        thread = threading.Thread(target=self._async_load_worker, args=(next_t, iters, active_sh_degree))
        thread.daemon = True
        thread.start()
        print(f"Started async loading of dataset {next_t}")
        
        return t, dataset


class CustomError(Exception):
    """Custom exception class for application-specific errors."""
    pass


def get_dataset(t, md, seq, iters, active_sh_degree, scale=1, bg=torch.tensor([0, 0, 0]), config=None):
    """
    Load and preprocess a complete dataset for a specific timestep.
    
    This function loads RGB images and segmentation masks for all cameras at a given timestep,
    applies scaling, and prepares them for rendering. If an image is missing, it attempts to
    copy from future timesteps.
    
    Args:
        t: Current timestep
        md: Metadata dictionary containing camera parameters and file paths
        seq: Sequence identifier
        iters: List of future timesteps to check for missing images
        active_sh_degree: Current spherical harmonics degree for rendering
        scale: Downscaling factor for images (default: 1, no scaling)
        bg: Background color tensor (default: black)
        config: Training configuration object
        
    Returns:
        List of dictionaries, each containing:
            - cam: Camera parameters for rendering
            - im: RGB image tensor (C, H, W)
            - seg: Segmentation mask tensor (C, H, W)
            - id: Camera ID
        Returns None if images cannot be found
    """
    dataset = []
    for c in range(len(md['fn'][t])):
        # Extract camera parameters
        w, h, k, w2c = md['w'], md['h'], md['k'][t][c], md['w2c'][t][c]
        
        # Setup camera with proper parameters
        cam = setup_camera(w, h, k, w2c, active_sh_degree=active_sh_degree, 
                          near=config.near_plane if config else 1.0, 
                          far=config.far_plane if config else 100, 
                          scale=scale, bg=bg)
        
        # Construct file paths
        fn = md['fn'][t][c]
        data_dir = config.data_dir if config else "./data"
        im_file_path = f"{data_dir}/{seq}/ims/{fn}"
        seg_file_path = f"{data_dir}/{seq}/seg/{fn}"
        
        # Use file existence checks instead of a hard file-size threshold.
        # PNGs can be valid and still be much smaller than 100KB.
        if os.path.getsize(im_file_path) / 1024 > 100 and os.path.isfile(seg_file_path):

       
            
            # Load and preprocess RGB image
            image = copy.deepcopy(Image.open(im_file_path))
            width, height = image.size
            if scale != 1:
                resized_image = image.resize((int(width/scale), int(height/scale)))
            else:
                resized_image = image

            # Convert to tensor and normalize to [0, 1]
            resized_image = np.array(resized_image)
            resized_image = torch.tensor(resized_image).float().cuda().permute(2, 0, 1) / 255 
            threshold = 128

            # Load and preprocess segmentation mask
            seg = copy.deepcopy(Image.open(seg_file_path))
            if scale != 1:
                resized_seg = seg.resize((int(width/scale), int(height/scale)))
            else:
                resized_seg = seg

            # Binarize segmentation mask (white/bright = foreground, black/dark = background).
            # Some datasets store masks as 2D grayscale, others as 3D RGB.
            resized_seg = np.array(resized_seg)
            if resized_seg.ndim == 2:
                new_img_array = (resized_seg >= threshold).astype(np.uint8)
            else:
                resized_seg = np.where(resized_seg < threshold, 0, 255).astype(np.uint8)
                new_img_array = np.all(resized_seg == [255, 255, 255], axis=-1).astype(np.uint8)
            
            # Convert segmentation to 3-channel tensor (R=foreground, G=0, B=0)
            new_img_array = torch.tensor(new_img_array).float().cuda()
            seg_col = torch.stack((new_img_array, torch.zeros_like(new_img_array), torch.zeros_like(new_img_array)))
            
            dataset.append({'cam': cam, 'im': resized_image, 'seg': seg_col, 'id': c})
        else:
            dataset.append({'cam': cam, 'im': None, 'seg': None, 'id': c})
    
    return dataset


def get_batch(todo_dataset, dataset):
    """
    Randomly sample a single data item from the dataset.
    
    This function implements random sampling for stochastic training. It maintains
    a todo list and refills it when empty to ensure all data is used.
    
    Args:
        todo_dataset: List of remaining data items to process
        dataset: Complete dataset to sample from
        
    Returns:
        Tuple of (sampled_data, index) where:
            - sampled_data: Dictionary containing camera, image, and segmentation data
            - index: Index of the sampled item in the original dataset
    """
    # Refill todo list if empty
    if not todo_dataset:
        todo_dataset = dataset.copy()
    
    # # Randomly select and remove an item
    # index_to_remove = randint(0, len(todo_dataset) - 1)
    # curr_data = todo_dataset.pop(index_to_remove)
    # return curr_data, index_to_remove

    # Keep trying until a valid item is found
    while True:
        # Randomly select an index
        index_to_remove = randint(0, len(todo_dataset) - 1)
        curr_data = todo_dataset[index_to_remove]
        
        # Check if "im" key exists and is not None
        if "im" in curr_data and curr_data["im"] is None:
            # If invalid, remove it from todo_dataset and continue the loop
            # todo_dataset.pop(index_to_remove)
            # # If todo_dataset becomes empty, refill it and try again
            # if not todo_dataset:
            #     todo_dataset = dataset.copy()
            continue # Try sampling again
        else:
            # If valid, remove it and return
            # todo_dataset.pop(index_to_remove)
            return curr_data, index_to_remove


def initialize_params(init_file, md, config: TrainingConfig):
    """
    Initialize 3D Gaussian parameters from a point cloud file.
    
    This function loads an initial point cloud (from .npz or .ply file) and initializes
    all Gaussian parameters including positions, colors, rotations, scales, and opacities.
    It also computes the scene radius for learning rate scaling.
    
    Args:
        init_file: Path to initialization file (.npz or .ply)
        md: Metadata dictionary containing camera information
        config: Training configuration object
        
    Returns:
        Tuple of (params, variables) where:
            - params: Dictionary of trainable parameters (as torch.nn.Parameter)
            - variables: Dictionary of non-trainable variables for tracking
            
    Raises:
        CustomError: If file format is not recognized
    """
    # Load point cloud from .npz file
    if init_file.endswith("npz"):
        try:
            init_pt_cld = np.load(init_file)["data"]
            points = init_pt_cld[:, :3]
            colors = init_pt_cld[:, 3:6]
            seg = init_pt_cld[:, 6]
            sq_dist, ind = o3d_knn(points, 3)
        except:
            init_pt_cld = np.load(init_file)
            points = init_pt_cld["means3D"]
            colors = init_pt_cld['rgb_colors']
            seg = np.ones((points.shape[0], 1), dtype=np.float32)
            seg = np.squeeze(seg)
            sq_dist, ind = o3d_knn(points, 3)

    # Load point cloud from .ply file
    elif init_file.endswith("ply"):
        pcd = fetchPly(init_file)
        points = pcd.points
        colors = RGB2SH(np.asarray(pcd.colors))  # Convert RGB to spherical harmonics
        
        # Initialize spherical harmonics features
        features = np.zeros((colors.shape[0], 3, (config.max_sh_degree + 1) ** 2))
        features[:, :3, 0] = colors  # DC component
        features[:, 3:, 1:] = 0.0    # Higher-order components initialized to zero
        
        seg = np.ones((points.shape[0], 1), dtype=np.float32)
        seg = np.squeeze(seg)
        sq_dist, ind = o3d_knn(points, 3)
    else:
        raise CustomError("No recognized file format for initializing params")

    # Compute mean distance to 3 nearest neighbors for scale initialization
    mean3_sq_dist = sq_dist.mean(-1).clip(min=0.0000001)
    num_cams = len(md['fn'][0])

    # Initialize all Gaussian parameters
    params = {
        'means3D': points,  # 3D positions
        'f_dc': np.transpose(features[:, :, 0:1], (0, 2, 1)),  # DC spherical harmonics
        'f_rest': np.transpose(features[:, :, 1:], (0, 2, 1)),  # Higher-order SH
        'seg_colors': np.stack((seg, np.zeros_like(seg), 1 - seg), -1),  # Segmentation colors
        'unnorm_rotations': np.tile([1, 0, 0, 0], (seg.shape[0], 1)),  # Quaternions (identity)
        'logit_opacities': np.zeros((seg.shape[0], 1)),  # Opacity in logit space
        'log_scales': np.tile(np.log(np.sqrt(mean3_sq_dist))[..., None], (1, 3)),  # Log scales
        'cam_m': np.zeros((num_cams, 3)),  # Camera exposure multiplier
        'cam_c': np.zeros((num_cams, 3)),  # Camera exposure bias
    }
    
    # Convert to PyTorch parameters
    params = {k: torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True)) 
              for k, v in params.items()}
    
    # Compute scene radius for learning rate scaling
    cam_centers = np.linalg.inv(md['w2c'][0])[:, :3, 3]
    scene_radius = config.scene_radius_multiplier * np.max(np.linalg.norm(cam_centers - np.mean(cam_centers, 0)[None], axis=-1))
    
    # Initialize tracking variables
    variables = {
        'max_2D_radius': torch.zeros(params['means3D'].shape[0]).cuda().float(),
        'scene_radius': scene_radius,
        'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0]).cuda().float(),
        'denom': torch.zeros(params['means3D'].shape[0]).cuda().float()
    }
    return params, variables


def initialize_optimizer(params, variables, config: TrainingConfig):
    """
    Initialize the Adam optimizer with parameter-specific learning rates.
    
    Learning rates are scaled based on the scene radius for spatial parameters
    to ensure consistent optimization across different scene scales.
    
    Args:
        params: Dictionary of trainable parameters
        variables: Dictionary containing scene_radius for LR scaling
        config: Training configuration with learning rate settings
        
    Returns:
        torch.optim.Adam optimizer with parameter groups
    """
    # Define learning rates for each parameter type
    lrs = {
        'means3D': config.lr_means3D_init * variables['scene_radius'] ,  # Scaled by scene size
        'f_dc': config.lr_f_dc,
        'f_rest': config.lr_f_rest,
        'seg_colors': config.lr_seg_colors,
        'unnorm_rotations': config.lr_unnorm_rotations,
        'logit_opacities': config.lr_logit_opacities,
        'log_scales': config.lr_log_scales,
        'cam_m': config.lr_cam_m,
        'cam_c': config.lr_cam_c,
    }
    
    # Create parameter groups with individual learning rates
    param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items()]
    return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)

def update_learning_rate(optimizer, means3D_scheduler_args, iteration):
    ''' Learning rate scheduling per step '''
    # for param_group in exposure_optimizer.param_groups:
    #     param_group['lr'] = exposure_scheduler_args(iteration)

    for param_group in optimizer.param_groups:
        if param_group["name"] == "means3D":
            lr = means3D_scheduler_args(iteration)
            param_group['lr'] = lr
            return lr

def set_lrs(optimizer, variables, config: TrainingConfig):
    """
    Update learning rates in the optimizer (typically after densification).
    
    This function is called when transitioning between training phases,
    particularly after the densification phase to adjust learning rates.
    Camera parameters are typically frozen (lr=0) after initial optimization.
    
    Args:
        optimizer: PyTorch optimizer to update
        variables: Dictionary containing scene_radius for LR scaling
        config: Training configuration with learning rate settings
        
    Returns:
        Updated optimizer with new learning rates
    """
    lrs = {
        'means3D': config.lr_means3D_init * variables['scene_radius'] ,
        'f_dc': config.lr_f_dc,
        'f_rest': config.lr_f_rest,
        'seg_colors': config.lr_seg_colors,
        'unnorm_rotations': config.lr_unnorm_rotations,
        'logit_opacities': config.lr_logit_opacities,
        'log_scales': config.lr_log_scales,
        'cam_m': 0.0,  # Freeze camera parameters
        'cam_c': 0.0,
    }

    # Update learning rates for each parameter group
    for param_group in optimizer.param_groups:
        if param_group["name"] in lrs.keys():
            param_group['lr'] = lrs[param_group["name"]]
    return optimizer


def get_loss(params, curr_data, variables, is_initial_timestep, i, config: TrainingConfig, active_sh_degree):
    """
    Compute the total loss for the current training iteration.
    
    This function renders the scene from the current viewpoint, computes multiple loss terms
    including RGB reconstruction, segmentation, and temporal consistency losses, and combines
    them with appropriate weights.
    
    Loss components:
        - RGB reconstruction: L1 + SSIM loss between rendered and ground truth images
        - Segmentation: L1 + SSIM loss for foreground/background segmentation
        - Rigid motion: Encourages locally rigid deformation (non-initial timesteps)
        - Rotation consistency: Smoothness of rotation field (non-initial timesteps)
        - Isometric: Preserves local distances (non-initial timesteps)
        - Feature consistency: Temporal smoothness of appearance (non-initial timesteps)
    
    Args:
        params: Dictionary of trainable Gaussian parameters
        curr_data: Current data batch (camera, image, segmentation)
        variables: Dictionary of tracking variables
        is_initial_timestep: Whether this is the first timestep
        i: Current iteration number
        config: Training configuration
        active_sh_degree: Current spherical harmonics degree
        
    Returns:
        Tuple of (loss, radius, variables) where:
            - loss: Total weighted loss scalar
            - radius: 2D radius of each Gaussian in screen space
            - variables: Updated tracking variables
        Returns (None, None, None) if rendering fails
    """
    losses = {}
    curr_id = curr_data['id']
    
    # Convert parameters to rendering format
    rendervar = params2rendervarshsaccel(params)
    if rendervar is None:
        return None, None, None

    # Enable gradient tracking for 2D means (needed for densification)
    rendervar['means2D'].retain_grad()
    
    # Setup rasterization settings
    raster_settings = Camera(
        image_height=curr_data["cam"].image_height,
        image_width=curr_data["cam"].image_width,
        tanfovx=curr_data['cam'].tanfovx,
        tanfovy=curr_data['cam'].tanfovy,
        bg=curr_data['cam'].bg,
        scale_modifier=curr_data['cam'].scale_modifier,
        viewmatrix=curr_data['cam'].viewmatrix,
        projmatrix=curr_data['cam'].projmatrix,
        sh_degree=active_sh_degree,
        campos=curr_data['cam'].campos,
        prefiltered=curr_data['cam'].prefiltered,
        debug=False,
        antialiasing=False
    )
    
    # Render RGB image
    im, radius, _ = Renderer(raster_settings=raster_settings)(**rendervar)

    # Save visualization images periodically
    if i % config.save_images_every == 0 and curr_id in config.save_image_ids:
        torchvision.utils.save_image(im, f'{config.data_dir}/{config.sequence}/im_{curr_id}.png')

    # Apply camera-specific exposure correction
    im = torch.exp(params['cam_m'][curr_id])[:, None, None] * im + params['cam_c'][curr_id][:, None, None]

    # Compute RGB reconstruction loss (L1 + SSIM)
    ssim_value = fused_ssim(im.unsqueeze(0), curr_data['im'].unsqueeze(0))
    mask_im = torch.any(curr_data['im'] != 0, dim=0)  # Mask out black pixels
    losses['im'] = config.loss_ratio * l1_loss_v1_noblack(im, curr_data['im'], mask_im) + (1 - config.loss_ratio) * (1.0 - ssim_value)
    
    # losses['im'] = config.loss_ratio * l1_loss_v1(im, curr_data['im']) + (1 - config.loss_ratio) * (1.0 - ssim_value)

    # Store 2D means for densification
    variables['means2D'] = rendervar['means2D']
    
    # Render segmentation
    segrendervar = params2rendervarseg(params)
    segrendervar['colors_precomp'] = params['seg_colors']
    seg, _, _, = Renderer(raster_settings=curr_data['cam'])(**segrendervar)

    # Compute segmentation loss (L1 + SSIM)
    ssim_value = fused_ssim(seg.unsqueeze(0), curr_data['seg'].unsqueeze(0))
    losses['seg'] = config.loss_ratio * l1_loss_v1(seg, curr_data['seg']) + (1 - config.loss_ratio) * (1.0 - ssim_value)

    # Initialize loss weights
    weight = 1
    loss_weights = {
        'im': config.loss_weight_im * weight,
        'seg': config.loss_weight_seg * weight,
        'rigid': config.loss_weight_rigid * weight,
        'rot': config.loss_weight_rot * weight,
        'iso': config.loss_weight_iso * weight,
        'floor': config.loss_weight_floor,
        'bg': config.loss_weight_bg,
        'soft_f_dc_cons': config.loss_weight_soft_f_dc_cons,
        'soft_f_rest_cons': config.loss_weight_soft_f_rest_cons,
        'cam_c': config.loss_weight_cam_c,
        'cam_m': config.loss_weight_cam_m,
    }

    # Add temporal consistency losses for non-initial timesteps
    if not is_initial_timestep:
        # After densification phase, simplify loss weights
        if i >= config.index_for_densify:
            weight = 1
            loss_weights = {
                'im': config.loss_weight_im * weight,
                'seg': config.loss_weight_seg * weight,
                'rigid': config.loss_weight_rigid * weight,
                'rot': config.loss_weight_rot * weight,
                'iso': config.loss_weight_iso * weight,
                'floor': config.loss_weight_floor,
                'bg': config.loss_weight_bg,
                'cam_c': config.loss_weight_cam_c,
                'cam_m': config.loss_weight_cam_m,
            }
        else:
            # Before densification, add temporal regularization losses
            is_fg = (params['seg_colors'][:, 0] > 0.5).detach()  # Foreground mask
            fg_pts = rendervar['means3D'][is_fg]
            fg_rot = rendervar['rotations'][is_fg]

            # Compute relative rotation from previous timestep
            rel_rot = quat_mult(fg_rot, variables["prev_inv_rot_fg"])
            rot = build_rotation(rel_rot)
            
            # Rigid motion loss: encourages locally rigid deformation
            neighbor_pts = fg_pts[variables["neighbor_indices"]]
            curr_offset = neighbor_pts - fg_pts[:, None]
            curr_offset_in_prev_coord = (rot.transpose(2, 1)[:, None] @ curr_offset[:, :, :, None]).squeeze(-1)
            losses['rigid'] = weighted_l2_loss_v2(curr_offset_in_prev_coord, variables["prev_offset"],
                                                variables["neighbor_weight"])

            # Rotation consistency loss: smooth rotation field
            losses['rot'] = weighted_l2_loss_v2(rel_rot[variables["neighbor_indices"]], rel_rot[:, None],
                                                variables["neighbor_weight"])

            # Isometric loss: preserve local distances
            curr_offset_mag = torch.sqrt((curr_offset ** 2).sum(-1) + 1e-20)
            losses['iso'] = weighted_l2_loss_v1(curr_offset_mag, variables["neighbor_dist"], variables["neighbor_weight"])

            # Appearance consistency loss: temporal smoothness
            losses['soft_f_dc_cons'] = l1_loss_v1(params['f_dc'], variables["prev_f_dc"])
            
            if active_sh_degree > 0:
                losses['soft_f_rest_cons'] = l1_loss_v1(params['f_rest'], variables["prev_f_rest"])

    # Compute total weighted loss
    loss = sum([loss_weights[k] * v for k, v in losses.items()])
    
    # Update maximum 2D radius for each Gaussian (used for densification)
    seen = radius > 0
    variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
    variables['seen'] = seen
    return loss, radius, variables


def oneupSHdegree(active_sh_degree, max_sh_degree):
    """
    Increment the active spherical harmonics degree.
    
    Spherical harmonics are progressively activated during training to allow
    the model to first learn coarse appearance, then refine with higher frequencies.
    
    Args:
        active_sh_degree: Current active SH degree
        max_sh_degree: Maximum SH degree allowed
        
    Returns:
        Updated active_sh_degree (incremented by 1 if below max)
    """
    if active_sh_degree < max_sh_degree:
        active_sh_degree += 1
    return active_sh_degree


def initialize_per_timestep(params, variables, optimizer, active_sh_degree, config: TrainingConfig):
    """
    Initialize parameters and variables for a new timestep using motion prediction.
    
    This function is called at the start of each non-initial timestep. It:
    1. Matches Gaussians between current and previous timestep
    2. Predicts new positions and rotations using linear extrapolation
    3. Computes k-nearest neighbors for temporal regularization
    4. Resets tracking variables and adjusts learning rates
    
    Args:
        params: Current Gaussian parameters
        variables: Tracking variables from previous timestep
        optimizer: PyTorch optimizer
        active_sh_degree: Current spherical harmonics degree
        config: Training configuration
        
    Returns:
        Tuple of (updated_params, updated_variables)
    """
    pts = params['means3D']
    f_dc = params['f_dc']
    rot = torch.nn.functional.normalize(params['unnorm_rotations'])
    
    # Active mask to consider only points that were present in the previous timestep (for motion prediction)
    active_mask = variables.get('point_mask', torch.ones_like(pts[:, 0], dtype=torch.bool))

    mask = active_mask & (torch.arange(len(active_mask), device=active_mask.device) < variables["prev_pts"].shape[0])

    

    pts_sub = pts[mask]
    rot_sub = rot[mask]
    
    # Predict new positions using linear extrapolation: p_new = p_curr + (p_curr - p_prev)

    new_pts = pts_sub + (pts_sub - variables["prev_pts"][mask[:variables["prev_pts"].shape[0]]])
    new_rot = torch.nn.functional.normalize(rot_sub) + (rot_sub - variables["prev_rot"][mask[:variables["prev_rot"].shape[0]]])
    
    # 
    new_params_dict = {}
    for k, v in params.items():
        if k in ['cam_m', 'cam_c']:
            new_params_dict[k] = v
        else:
            # Slice to active and break graph
            new_params_dict[k] = v[active_mask].detach().clone()
    
    mask_in_active_space = mask[active_mask]  # Boolean array of length = num_active_points
    
    # Update means3D and rotations for those positions
    new_params_dict['means3D'][mask_in_active_space] = new_pts.detach()
    new_params_dict['unnorm_rotations'][mask_in_active_space] = new_rot.detach()

        # 3. Now compute variables that depend on the active params
    pts = new_params_dict['means3D']
    rot = new_params_dict['unnorm_rotations']
    f_dc = new_params_dict['f_dc']
    seg_colors = new_params_dict['seg_colors']
    f_rest = new_params_dict['f_rest']
    
    is_fg = seg_colors[:, 0] > 0.5
    prev_inv_rot_fg = rot[is_fg]
    prev_inv_rot_fg[:, 1:] = -1 * prev_inv_rot_fg[:, 1:]
    fg_pts = pts[is_fg]

    neighbor_sq_dist, neighbor_indices = pytorch3d_knn(fg_pts, 20)
    neighbor_weight = torch.exp(-4000 * neighbor_sq_dist)
    neighbor_dist = torch.sqrt(neighbor_sq_dist)

    variables["neighbor_indices"] = neighbor_indices
    variables["neighbor_weight"] = neighbor_weight.detach()
    variables["neighbor_dist"] = neighbor_dist.detach()

    prev_offset = fg_pts[variables["neighbor_indices"]] - fg_pts[:, None]
    variables['prev_inv_rot_fg'] = prev_inv_rot_fg.detach()
    variables['prev_offset'] = prev_offset.detach()
    variables["prev_pts"] = pts.detach()
    variables["prev_rot"] = rot.detach()
    variables["prev_f_dc"] = f_dc.detach()

    if active_sh_degree > 0:
        variables["prev_f_rest"] = f_rest.detach()

    num_pts = pts.shape[0]
    variables['means2D_gradient_accum'] = torch.zeros(num_pts, device="cuda")
    variables['denom'] = torch.zeros(num_pts, device="cuda")
    variables['max_2D_radius'] = torch.zeros(num_pts, device="cuda")

    if 'point_mask' in variables:
        del variables['point_mask']
   
    # 4. Set learning rates
    # lrs = {
    #     'means3D': 0.00016 * variables['scene_radius'],
    #     'f_dc': 0.0025/5,
    #     'f_rest': 0.0025/(20*5),
    #     'unnorm_rotations': 0.001,
    #     'logit_opacities': 0.0,
    #     'log_scales': 0.0,
    #     'cam_m': 0.0,
    #     'cam_c': 0.0,
    # }

    lrs = {
        'means3D': config.lr_means3D_init * variables['scene_radius'] ,  # Scaled by scene size
        'f_dc': config.lr_f_dc,
        'f_rest': config.lr_f_rest,
        'unnorm_rotations': config.lr_unnorm_rotations,
        'logit_opacities': 0.0,  # Freeze opacities after initial timestep
        'log_scales': 0.0,  # Freeze scales after initial timestep
        'cam_m': 0.0,  # Freeze camera parameters after initial timestep
        'cam_c': 0.0
    }

    for param_group in optimizer.param_groups:
        if param_group["name"] in lrs.keys():
            param_group['lr'] = lrs[param_group["name"]]

    # 5. PHYSICALLY REPLACE parameters in optimizer with new leaf tensors
    for param_group in optimizer.param_groups:
        name = param_group['name']
        if name in new_params_dict:
            old_param = param_group['params'][0]
            
            # Remove old state
            if old_param in optimizer.state:
                del optimizer.state[old_param]
            
            # Create new leaf parameter
            new_leaf = torch.nn.Parameter(new_params_dict[name].requires_grad_(True))
            
            # Update optimizer
            param_group['params'][0] = new_leaf
            new_params_dict[name] = new_leaf
            
            # Fresh optimizer state
            optimizer.state[new_leaf] = {}

    return new_params_dict, variables

   


def initialize_post_first_timestep(params, variables, optimizer, config: TrainingConfig):
    """
    Perform initialization after completing the first timestep.
    
    This function is called once after the initial timestep is fully trained. It:
    1. Separates foreground and background Gaussians
    2. Computes k-nearest neighbors for foreground points
    3. Stores initial state for temporal tracking
    4. Freezes certain parameters (opacity, scale, camera parameters)
    
    Args:
        params: Trained Gaussian parameters from first timestep
        variables: Tracking variables
        optimizer: PyTorch optimizer
        config: Training configuration
        
    Returns:
        Tuple of (updated_variables, num_points)
    """
    # Separate foreground and background based on segmentation
    is_fg = params['seg_colors'][:, 0] > 0.5

    init_fg_pts = params['means3D'][is_fg]
    init_bg_pts = params['means3D'][~is_fg]
    init_bg_rot = torch.nn.functional.normalize(params['unnorm_rotations'][~is_fg])
    
    # Compute k-nearest neighbors for foreground points
    neighbor_sq_dist, neighbor_indices = pytorch3d_knn(init_fg_pts, config.num_knn)
    neighbor_weight = torch.exp(config.knn_weight_exp * neighbor_sq_dist)
    neighbor_dist = torch.sqrt(neighbor_sq_dist)

    # Store neighbor information
    variables["neighbor_indices"] = neighbor_indices
    variables["neighbor_weight"] = neighbor_weight 
    variables["neighbor_dist"] = neighbor_dist

    # Store initial background state (for potential background tracking)
    variables["init_bg_pts"] = init_bg_pts.detach()
    variables["init_bg_rot"] = init_bg_rot.detach()
    
    # Store current state as "previous" for next timestep
    variables["prev_pts"] = params['means3D'].detach()
    variables["prev_rot"] = torch.nn.functional.normalize(params['unnorm_rotations']).detach()

    # Freeze certain parameters after initial timestep
    params_to_fix = ['logit_opacities', 'log_scales', 'cam_m', 'cam_c']

    for param_group in optimizer.param_groups:
        if param_group["name"] in params_to_fix:
            param_group['lr'] = 0.0

    num_pts = params['means3D'].shape[0]
    return variables, num_pts


def report_progress(params, data, progress_bar, loss, i, config: TrainingConfig):
    """
    Report training progress by computing metrics and updating progress bar.
    
    This function periodically renders an image, computes PSNR, and updates
    the progress bar with current metrics.
    
    Args:
        params: Current Gaussian parameters
        data: Data dictionary containing ground truth image and camera
        progress_bar: tqdm progress bar object
        loss: Current loss value
        i: Current iteration number
        config: Training configuration
    """
    if i % config.report_every_i == 0:
        # Render image from first camera
        im, _, _, = Renderer(raster_settings=data['cam'])(**params2rendervarshsaccel(params))
        curr_id = data['id']
        
        # Apply camera exposure correction
        im = torch.exp(params['cam_m'][curr_id])[:, None, None] * im + params['cam_c'][curr_id][:, None, None]
        
        # Compute PSNR metric
        psnr = calc_psnr(im, data['im']).mean()

        # Update progress bar with metrics
        progress_bar.set_postfix({"train img 0 PSNR": f"{psnr:.{7}f}, loss: {loss.cpu().numpy().item():.{7}f}"})
        progress_bar.update(config.report_every_i)


def train(config: TrainingConfig):
    """
    Main training loop for dynamic 3D Gaussian Splatting.
    
    This function orchestrates the entire training process:
    1. Loads metadata and initializes parameters
    2. Iterates through all timesteps in the sequence
    3. For each timestep:
        - Loads data asynchronously
        - Performs optimization iterations
        - Applies densification (adding/removing Gaussians)
        - Saves trained parameters
    4. Handles special cases for initial vs. subsequent timesteps
    
    Training phases:
        - Initial timestep: Full optimization with densification (12k iterations)
        - Subsequent timesteps: Motion prediction + refinement (2k iterations)
    
    Args:
        config: Training configuration object containing all hyperparameters
    """
    seq = config.sequence
    exp_prefix = config.exp_name_prefix

    # Load metadata (camera parameters, image paths, etc.)
    md = json.load(open(f"{config.data_dir}/{seq}/train_meta.json", 'r'))
    num_timesteps = len(md['fn']) if config.final_timestep is None else min(config.final_timestep, len(md['fn']))

  

    exp = f"{exp_prefix}SH_{config.max_sh_degree}_maxnumsplat_{config.max_num_splat}_scale_{config.scale}_initialtimestep_{config.initial_timestep}_finaltimestep_{num_timesteps}"
    output_path = f"{config.output_dir}/{seq}/{exp}"
    
    if os.path.exists(output_path):
        print(f"Experiment '{exp}' for sequence '{seq}' already exists. Exiting.")
        return
    

    # Initialize Gaussian parameters from point cloud
    init_params_file = config.get_init_params_file()
    params, variables = initialize_params(init_params_file, md, config)
    optimizer = initialize_optimizer(params, variables, config)
 
    means3D_scheduler_args = get_expon_lr_func(lr_init=config.lr_means3D_init*variables['scene_radius'],
                                                lr_final=config.lr_means3D_final*variables['scene_radius'],
                                                lr_delay_mult=config.lr_means3D_delay_mult,
                                                max_steps=config.lr_means3D_1_max_steps)

    # Setup asynchronous data loader
    loader = AsyncDatasetLoader(md, seq, config, start_t=config.initial_timestep, 
                                total_iterations=num_timesteps)

    # Main training loop over timesteps
    t = 0
    while True:
        if t >= num_timesteps:
            break
        if t < config.initial_timestep:
            t += 1
            continue
        
        # Determine which future timesteps to check for missing images
        if t == num_timesteps - 1:
            iters = [t - 1]
        else:
            iters = [it for it in range(t + 1, t + config.lookahead_frames + 1) if it < num_timesteps]
        
        todo_dataset = []
        is_initial_timestep = (t == config.initial_timestep)
        
        # Set spherical harmonics degree (start from 0 for initial timestep)
        if is_initial_timestep:
            active_sh_degree = 0
        else:
            active_sh_degree = config.max_sh_degree
        
        # Load dataset asynchronously
        _, dataset = loader.get_next_dataset(iters=iters, active_sh_degree=active_sh_degree) 
        
        if dataset is None:
            print(f"Training of the dataset has ended!")
            return

        # Initialize parameters for new timestep (motion prediction)
        if not is_initial_timestep:
            params, variables = initialize_per_timestep(params, variables, optimizer, active_sh_degree, config)
        
        

        # Determine number of iterations for this timestep
        num_iter_per_timestep = (config.num_iter_per_timestep_for_initial_frame if is_initial_timestep 
                                 else config.num_iter_per_timestep_for_non_initial_frames)
        progress_bar = tqdm(range(num_iter_per_timestep), desc=f"timestep {t}")

        i = 1
        k = 0  # Densification counter
      
        # Optimization loop for current timestep
        while True:
            if i > num_iter_per_timestep:
                break

           
                
            if is_initial_timestep:   
                update_learning_rate(optimizer, means3D_scheduler_args, i)
                pass
            else:
                
                if i == 1 :
                    means3D_scheduler_args = get_expon_lr_func(lr_init=config.lr_means3D_init*variables['scene_radius'],
                                            lr_final=config.lr_means3D_final*variables['scene_radius'],
                                            lr_delay_mult=config.lr_means3D_delay_mult,
                                            max_steps=config.lr_means3D_1_max_steps)
                if i == config.index_for_densify + 1:
                     means3D_scheduler_args = get_expon_lr_func(lr_init=config.lr_means3D_init*variables['scene_radius'],
                                            lr_final=config.lr_means3D_final*variables['scene_radius'],
                                            lr_delay_mult=config.lr_means3D_delay_mult,
                                            max_steps=config.lr_means3D_1_max_steps)

                if i <= config.index_for_densify:
                    ii = i 
                else:
                    ii = i - config.index_for_densify 
                update_learning_rate(optimizer, means3D_scheduler_args, ii)

            # Progressively increase SH degree during initial timestep
            if is_initial_timestep and i % 2000 == 0:
                active_sh_degree = oneupSHdegree(active_sh_degree, config.max_sh_degree)

            # Sample random batch
            curr_data, _ = get_batch(todo_dataset, dataset)

            try:
                # Compute loss and gradients
                loss, radius, variables = get_loss(params, curr_data, variables, is_initial_timestep, 
                                                   i, config, active_sh_degree)
                if loss is None:
                    continue
            except Exception as e:
                raise Exception(f"An error occurred in loss function {e}")

            # Backpropagation
            loss.backward()

            with torch.no_grad():
                try:
                    # Report progress periodically
                    report_progress(params, dataset[0], progress_bar, loss, i + 1, config)
                    
                    # Densification for non-initial timesteps (after warmup)
                    if i >= config.index_for_densify and not is_initial_timestep:
                        if i == config.index_for_densify:
                            optimizer = set_lrs(optimizer, variables, config)

                        params, variables = densify_nokeyframes_v2(params, variables, optimizer, k, config.max_num_splat) 
                        k += 1

                    # Densification for initial timestep (more aggressive)
                    if is_initial_timestep:
                        params, variables = densify(params, variables, optimizer, i, max_num_splat=config.max_num_splat * 2)

                except Exception as e:
                    raise Exception(f"An error occurred in report progress or densify functions {e}")

                # Optimizer step
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
             
            i += 1

        progress_bar.close()

        # Post-processing after initial timestep
        if is_initial_timestep:
            variables, num_pts = initialize_post_first_timestep(params, variables, optimizer, config)
        
        # Save trained parameters to PLY file
        if "point_mask" in variables:
            save_params_to_ply_shs(params, variables['point_mask'], output_path, t)
        else:
            save_params_to_ply_shs(params, torch.ones(params['means3D'].shape[0], dtype=torch.bool), output_path, t)

        t += 1


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Gaussian Splatting model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file (YAML or JSON)')
    args = parser.parse_args()
    
    # Load configuration from file
    if args.config.endswith('.yaml') or args.config.endswith('.yml'):
        config = TrainingConfig.from_yaml(args.config)
    elif args.config.endswith('.json'):
        config = TrainingConfig.from_json(args.config)
    else:
        raise ValueError("Config file must be YAML or JSON")
    
    # Start training
    train(config)
    torch.cuda.empty_cache()