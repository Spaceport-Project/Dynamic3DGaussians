# config.py
from dataclasses import dataclass, field
from typing import List, Optional
import yaml
import json

@dataclass
class TrainingConfig:
    # Sequence and experiment settings
    sequence: str
    exp_name_prefix: str
    initial_timestep: int = 0
    final_timestep: Optional[int] = None

    # Paths
    data_dir: str = "./data"
    output_dir: str = "./output"
    init_params_file: Optional[str] = None  # If None, uses default
    
    # Model parameters
    max_num_splat: int = 30000
    index_for_densify: int = 700
    max_sh_degree: int = 1
    scale: float = 1.0
    use_gpu_projections: bool = False
    motion_use_gpu: bool = False
    motion_gpu_backend: str = "raft"
    motion_flow_threshold: float = 1.0
    motion_raft_onnx_path: Optional[str] = "/home/hamit/Softwares/Dynamic3DGaussians/models_onnx/optical_flow_estimation_raft_2023aug.onnx"
    motion_raft_onnx_width: Optional[int] = None
    motion_raft_onnx_height: Optional[int] = None
    motion_raft_onnx_use_cuda: bool = True
    # PyTorch RAFT memory optimization: max height/width before downscaling (None = no downscale)
    motion_raft_pytorch_max_height: Optional[int] = 720  # Downscale to 720p for memory efficiency
    motion_raft_pytorch_max_width: Optional[int] = 1280
    motion_raft_pytorch_use_mixed_precision: bool = True  # Use float16 for inference
    motion_raft_pytorch_gpu_device: Optional[int] = None  # Which GPU for RAFT (None = same as training, 0/1/2 = specific GPU)
    motion_raft_pytorch_disable_batch: bool = True  # Stability: skip batch RAFT and use per-camera mode
    use_amp: bool = False
    amp_dtype: str = "fp16"
    
    # Training iterations
    num_iter_per_timestep_for_initial_frame: int = 12000
    num_iter_per_timestep_for_non_initial_frames: int = 2000
    
    # Optimizer learning rates
    lr_means3D_init: float = 0.00016
    lr_means3D_final: float = 0.0000016
    lr_means3D_delay_mult: float = 0.01
    lr_means3D_1_max_steps: int = num_iter_per_timestep_for_initial_frame
    lr_means3D_2_max_steps: int = index_for_densify
    lr_means3D_3_max_steps: int = num_iter_per_timestep_for_non_initial_frames - index_for_densify


    lr_f_dc: float = 0.0025
    lr_f_rest: float = 0.000125  # 0.0025/20
    lr_seg_colors: float = 0.0
    lr_unnorm_rotations: float = 0.001
    lr_logit_opacities: float = 0.05
    lr_log_scales: float = 0.001
    lr_cam_m: float = 1e-4
    lr_cam_c: float = 1e-4
    
    # Loss weights
    loss_weight_im: float = 1.0
    loss_weight_seg: float = 3.0
    loss_weight_rigid: float = 4.0
    loss_weight_rot: float = 4.0
    loss_weight_iso: float = 2.0
    loss_weight_floor: float = 2.0
    loss_weight_bg: float = 20.0
    loss_weight_soft_f_dc_cons: float = 0.01
    loss_weight_soft_f_rest_cons: float = 0.01
    loss_weight_cam_c: float = 1.0
    loss_weight_cam_m: float = 1.0
    
    # Loss parameters
    loss_ratio: float = 0.80  # L1 vs SSIM ratio
    image_ssim_every: int = 1
    image_ssim_scale_when_sparse: bool = True
    seg_loss_every: int = 1
    seg_loss_scale_when_sparse: bool = True
    
    # Densification parameters
    densify_grad_threshold: float = 0.0002
    densification_interval: int = 100
    opacity_reset_interval: int = 3000
    densify_from_iter: int = 500
    densify_until_iter: int = 15000
    percent_dense: float = 0.01

    # Geometric outlier pruning (disabled when threshold <= 0)
    geometric_prune_distance_threshold: float = 0.0
    geometric_prune_interval: int = 100
    geometric_prune_query_chunk_size: int = 8192
    geometric_prune_ref_chunk_size: int = 8192

    # Segmentation-region pruning (disabled when eps <= 0)
    seg_region_prune_eps: float = 0.0
    seg_region_prune_min_cluster_points: int = 32
    seg_region_prune_interval: int = 200
    seg_region_prune_clustering_algorithm: str = "dbscan"
    seg_region_prune_fg_threshold: float = 0.5
    seg_region_prune_target: str = "foreground"
    seg_region_prune_remove_noise: bool = False
    seg_region_prune_preserve_largest_cluster: bool = True
    seg_region_prune_max_prune_fraction: float = 0.25
    seg_region_prune_cluster_size_threshold: int = 1500
    
    # KNN parameters
    num_knn: int = 20
    knn_weight_exp: float = -4000.0
    
    # Camera parameters
    near_plane: float = 1.0
    far_plane: float = 100.0
    max_cams: int = 50
    
    # Data loader parameters
    async_queue_size: int = 2
    lookahead_frames: int = 5
    
    # Reporting
    report_every_i: int = 100
    enable_timing_breakdown: bool = True
    timing_sync_cuda: bool = False
    timing_report_every_i: Optional[int] = None
    save_images_every: int = 5
    save_image_ids: List[int] = field(default_factory=lambda: [0, 1, 26, 40])
    
    
    
    # Background color
    bg_color: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    
    # Misc
    scene_radius_multiplier: float = 1.1
    
    @classmethod
    def from_yaml(cls, yaml_path: str):
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, json_path: str):
        """Load configuration from JSON file"""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file"""
        with open(yaml_path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)
    
    def to_json(self, json_path: str):
        """Save configuration to JSON file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
    
    def get_init_params_file(self):
        """Get the initialization parameters file path"""
        if self.init_params_file is not None:
            return  f"{self.data_dir}/{self.sequence}/{self.init_params_file}"
        return f"{self.data_dir}/{self.sequence}/points3D_sparse.ply"