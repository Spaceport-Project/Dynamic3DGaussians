#!/usr/bin/env python3
"""
MP4 to COLMAP Pipeline

This script extracts frames from an MP4 video at regular intervals, runs the
complete COLMAP reconstruction pipeline on those frames, and then removes the
backgrounds from the undistorted COLMAP images with rembg.

Usage:
    python mp4_to_colmap.py --input video.mp4 --output /path/to/output [--frame_interval 30] [--camera OPENCV]
"""

import os
import sys
import logging
from argparse import ArgumentParser
import subprocess
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import collections
import yaml
import re
import math
import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial.transform import Rotation as R_scipy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def aggregate_cam_params_and_images(k, w2c, interval):
    """Aggregate camera paths, IDs, intrinsics and poses across a timestep interval."""
    images = []
    cam_ids = []
    k_all = []
    w2c_all = []

    indices_to_remove = []
    w2c = [x for i, x in enumerate(w2c) if i not in indices_to_remove]

    if len(k) == 1:
        k_temp = [k[0] for _ in range(len(w2c))]
    else:
        k = [x for i, x in enumerate(k) if i not in indices_to_remove]

    for ii in range(interval[0], interval[1]):
        image_list = []
        cam_list = []
        for idx in range(len(w2c)):
            if idx in indices_to_remove:
                continue
            image_list.append(f"{idx}/{ii:06d}.png")
            cam_list.append(idx)

        images.append(image_list)
        cam_ids.append(cam_list)
        if len(k) == 1:
            k_all.append(k_temp)
        else:
            k_all.append(k)
        w2c_all.append(w2c)

    return images, cam_ids, k_all, w2c_all


def parse_arguments():
    """Parse command line arguments."""
    parser = ArgumentParser(description="Extract frames from MP4, run COLMAP, and remove backgrounds afterward")
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        type=str,
        help="Input MP4 video file"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        type=str,
        help="Output directory for COLMAP results"
    )
    parser.add_argument(
        "--frame_interval",
        type=int,
        default=30,
        help="Fallback: extract every Nth frame when auto-detection fails (default: 30)"
    )
    parser.add_argument(
        "--target_frames",
        type=int,
        default=100,
        help="Target number of extracted frames per video (default: 100)"
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="OPENCV",
        help="COLMAP camera model (default: OPENCV)"
    )
    parser.add_argument(
        "--colmap_executable",
        type=str,
        default="",
        help="Path to COLMAP executable (default: 'colmap' from PATH)"
    )
    parser.add_argument(
        "--no_gpu",
        action='store_true',
        help="Disable GPU acceleration for COLMAP"
    )
    parser.add_argument(
        "--skip_frame_extraction",
        action='store_true',
        help="Skip frame extraction (reuse existing frames)"
    )
    parser.add_argument(
        "--colmap",
        action='store_true',
        help="Run the COLMAP reconstruction pipeline"
    )
    parser.add_argument(
        "--background_removal",
        action='store_true',
        help="Run rembg background removal on undistorted COLMAP images"
    )
    parser.add_argument(
        "--rembg_model",
        type=str,
        default="birefnet-general",
        help="rembg model to use for background removal (default: birefnet-general)"
    )
    parser.add_argument(
        "--origin_margin",
        type=float,
        default=5.0,
        help="Keep only 3-D points within this distance from the origin (0,0,0) (default: 5.0)"
    )
    parser.add_argument(
        "--auto_margin",
        action="store_true",
        help="Auto-detect the origin margin by finding where point density drops drastically"
    )
    parser.add_argument(
        "--density_bins",
        type=int,
        default=50,
        help="Number of radial bins used for density-dropoff detection (default: 50)"
    )
    parser.add_argument(
        "--origin_filter",
        action='store_true',
        help="Run origin-based point cloud filtering (keep points within --origin_margin of the origin)"
    )
    parser.add_argument(
        "--create_meta",
        action='store_true',
        help="Create train_meta.json in a dataset subfolder"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="dataset",
        help="Name of the dataset subfolder (default: 'dataset')"
    )
    parser.add_argument(
        "--frame_start",
        type=int,
        default=0,
        help="Start frame index for train_meta.json (default: 0)"
    )
    parser.add_argument(
        "--frame_end",
        type=int,
        default=None,
        help="End frame index for train_meta.json (default: all frames)"
    )
    parser.add_argument(
        "--train_config_template",
        type=str,
        default="configs/config_template.yaml",
        help="Template YAML used to generate a training config"
    )
    parser.add_argument(
        "--train_config_name",
        type=str,
        default="train_config.yaml",
        help="Generated training config filename inside dataset folder"
    )
    parser.add_argument(
        "--run_training",
        action="store_true",
        help="Run train.py after creating dataset metadata"
    )
    parser.add_argument(
        "--skip_to_meta_training",
        action="store_true",
        help="Skip frame extraction/COLMAP/rembg and run only create_meta + optional training from existing outputs"
    )
    
    return parser.parse_args()


def verify_input_video(input_path):
    """Verify that the input video file exists."""
    if not os.path.isfile(input_path):
        logger.error(f"Input video file not found: {input_path}")
        sys.exit(1)
    
    if not input_path.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
        logger.warning(f"Input file may not be a video: {input_path}")
    
    logger.info(f"Input video verified: {input_path}")


def setup_directories(output_dir):
    """Create necessary directory structure."""
    input_dir = os.path.join(output_dir, "input")
    os.makedirs(input_dir, exist_ok=True)
    logger.info(f"Created input directory: {input_dir}")
    return input_dir


def extract_frames(input_video, output_dir, frame_interval):
    """
    Extract frames from MP4 at regular intervals using ffmpeg.
    
    Args:
        input_video: Path to input MP4 file
        output_dir: Directory to save extracted frames
        frame_interval: Extract every Nth frame
    """
    logger.info(f"Starting frame extraction (every {frame_interval}th frame)...")

    # FFmpeg command to extract frames
    # select=not(mod(n\,{frame_interval})) means "select frames where n % frame_interval == 0"
    # vsync vfr means variable frame rate synchronization
    output_pattern = os.path.join(output_dir, "%d.png")

    common_args = [
        "-i", input_video,
        "-an", "-sn", "-dn",
        "-vf", f"select=not(mod(n\\,{frame_interval}))",
        "-vsync", "vfr",
        "-start_number", "0",
        "-compression_level", "1",
        "-threads", "0",
        output_pattern,
    ]

    ffmpeg_hw_cmd = ["ffmpeg", "-hwaccel", "auto"] + common_args
    ffmpeg_sw_cmd = ["ffmpeg"] + common_args

    logger.info(f"Running (hardware-accelerated): {' '.join(ffmpeg_hw_cmd)}")

    try:
        subprocess.run(ffmpeg_hw_cmd, check=True, capture_output=False)
        logger.info("Frame extraction completed successfully (hardware path)")
    except subprocess.CalledProcessError as e:
        logger.warning(
            "Hardware-accelerated frame extraction failed with code "
            f"{e.returncode}. Falling back to software decoding."
        )
        try:
            logger.info(f"Running (software fallback): {' '.join(ffmpeg_sw_cmd)}")
            subprocess.run(ffmpeg_sw_cmd, check=True, capture_output=False)
            logger.info("Frame extraction completed successfully (software fallback)")
        except subprocess.CalledProcessError as sw_e:
            logger.error(f"Frame extraction failed with code {sw_e.returncode}")
            sys.exit(1)
    except FileNotFoundError:
        logger.error("ffmpeg not found. Please install ffmpeg or add it to PATH")
        sys.exit(1)


def _estimate_total_frames_with_ffprobe(input_video):
    """Estimate total video frames using ffprobe metadata."""
    ffprobe_cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames,avg_frame_rate,duration:format=duration",
        "-of", "json",
        input_video,
    ]

    try:
        result = subprocess.run(ffprobe_cmd, check=True, capture_output=True, text=True)
        probe = json.loads(result.stdout)
        streams = probe.get("streams", [])
        if not streams:
            return None

        stream = streams[0]
        nb_frames = stream.get("nb_frames")
        if nb_frames not in (None, "", "N/A"):
            total_frames = int(float(nb_frames))
            return total_frames if total_frames > 0 else None

        avg_frame_rate = stream.get("avg_frame_rate", "0/0")
        duration_value = stream.get("duration") or probe.get("format", {}).get("duration")
        if duration_value in (None, "", "N/A"):
            return None

        duration = float(duration_value)
        num, den = avg_frame_rate.split("/")
        fps = float(num) / float(den) if float(den) != 0.0 else 0.0
        if duration <= 0.0 or fps <= 0.0:
            return None

        total_frames = int(round(duration * fps))
        return total_frames if total_frames > 0 else None
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError, json.JSONDecodeError, ZeroDivisionError):
        return None


def extract_around_n_frames(input_video, output_dir, target_frames=100, fallback_interval=30):
    """
    Extract approximately target_frames from an input video by auto-computing
    frame_interval from estimated total frame count.

    Falls back to fallback_interval when ffprobe metadata is unavailable.
    """
    safe_target = max(1, int(target_frames))
    safe_fallback = max(1, int(fallback_interval))

    total_frames = _estimate_total_frames_with_ffprobe(input_video)
    if total_frames is None:
        logger.warning(
            "Could not estimate total frame count with ffprobe. "
            f"Falling back to frame_interval={safe_fallback}."
        )
        extract_frames(input_video, output_dir, safe_fallback)
        return

    frame_interval = max(1, int(round(total_frames / float(safe_target))))
    estimated_output_frames = int(math.ceil(total_frames / float(frame_interval)))
    logger.info(
        f"Auto frame selection: total_frames={total_frames}, target={safe_target}, "
        f"frame_interval={frame_interval}, estimated_extracted={estimated_output_frames}"
    )
    extract_frames(input_video, output_dir, frame_interval)


def remove_backgrounds(input_dir, output_dir, model_name):
    """Remove frame backgrounds with rembg and store the results in output_dir."""
    if not os.path.exists(input_dir) or not os.listdir(input_dir):
        logger.error(f"No input frames found for background removal: {input_dir}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    for entry in os.listdir(output_dir):
        entry_path = os.path.join(output_dir, entry)
        if os.path.isdir(entry_path):
            shutil.rmtree(entry_path)
        else:
            os.remove(entry_path)

    logger.info(f"Starting background removal with rembg model '{model_name}'...")

    rembg_cmd = [
        "rembg",
        "p",
        "-m", model_name,
        input_dir,
        output_dir,
    ]

    logger.info(f"Running: {' '.join(rembg_cmd)}")

    try:
        subprocess.run(rembg_cmd, check=True, capture_output=False)
        logger.info("Background removal completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Background removal failed with code {e.returncode}")
        sys.exit(1)
    except FileNotFoundError:
        logger.error("rembg not found. Please install rembg in the active environment")
        sys.exit(1)


def reorganize_rembg_outputs(rembg_dir, ims_root_dir, seg_root_dir):
    """Move rembg outputs to ims/<frame_id>/000000.png and seg/<frame_id>/000000.png."""
    if not os.path.isdir(rembg_dir):
        logger.error(f"rembg output directory not found: {rembg_dir}")
        sys.exit(1)

    image_files = [
        f for f in os.listdir(rembg_dir)
        if os.path.isfile(os.path.join(rembg_dir, f)) and f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    ]

    if not image_files:
        logger.error(f"No rembg output images found in: {rembg_dir}")
        sys.exit(1)

    # Sort numerically when the filename stem is a frame index, fallback to lexicographic.
    image_files.sort(key=lambda name: (0, int(os.path.splitext(name)[0])) if os.path.splitext(name)[0].isdigit() else (1, name))

    os.makedirs(ims_root_dir, exist_ok=True)
    os.makedirs(seg_root_dir, exist_ok=True)

    for root_dir in (ims_root_dir, seg_root_dir):
        for entry in os.listdir(root_dir):
            entry_path = os.path.join(root_dir, entry)
            if os.path.isdir(entry_path):
                shutil.rmtree(entry_path)
            else:
                os.remove(entry_path)

    worker_count = min(len(image_files), max(1, os.cpu_count() or 1), 32)
    logger.info(f"Processing rembg outputs in parallel with {worker_count} workers")

    def process_single_file(filename):
        source_path = os.path.join(rembg_dir, filename)
        frame_id = os.path.splitext(filename)[0]
        image_target_dir = os.path.join(ims_root_dir, frame_id)
        mask_target_dir = os.path.join(seg_root_dir, frame_id)
        image_target_path = os.path.join(image_target_dir, "000000.png")
        mask_target_path = os.path.join(mask_target_dir, "000000.png")

        os.makedirs(image_target_dir, exist_ok=True)
        os.makedirs(mask_target_dir, exist_ok=True)

        with Image.open(source_path) as image:
            rgba_image = image.convert("RGBA")
            rgb_image = Image.new("RGB", rgba_image.size, (0, 0, 0))
            rgb_image.paste(rgba_image.convert("RGB"), mask=rgba_image.getchannel("A"))
            alpha_channel = np.array(rgba_image.getchannel("A"), dtype=np.uint8)
            mask_image = Image.fromarray(np.where(alpha_channel > 0, 255, 0).astype(np.uint8))
            rgb_image.save(image_target_path)
            mask_image.save(mask_target_path)

        os.remove(source_path)
        return filename

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        moved_count = sum(1 for _ in executor.map(process_single_file, image_files))

    logger.info(f"Reorganized {moved_count} rembg images into: {ims_root_dir}")
    logger.info(f"Generated {moved_count} rembg masks into: {seg_root_dir}")


def _folder_size_bytes(folder: Path) -> int:
    """Return total size in bytes for all regular files under folder."""
    total = 0
    for path in folder.rglob("*"):
        if path.is_file() and not path.is_symlink():
            try:
                total += path.stat().st_size
            except OSError:
                # Skip files that disappear or are not accessible during traversal.
                continue
    return total


def _human(num_bytes: int) -> str:
    """Format bytes in a human-readable string."""
    value = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}" if unit != "B" else f"{int(value)} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def detect_density_dropoff(points: np.ndarray, n_bins: int = 50) -> float:
    """
    Analyse point density as a function of radial distance from the origin and
    return the radius where density drops most drastically.

    Strategy:
      1. Bin points into spherical shells.
      2. Normalise counts by shell volume so density is comparable across radii.
      3. Smooth with a simple moving average to reduce noise.
      4. Find the bin whose density derivative is the most steeply negative
         (largest drop).  That bin edge becomes the auto margin.
    """
    distances = np.linalg.norm(points, axis=1)
    max_dist = float(distances.max())
    bins = np.linspace(0.0, max_dist, n_bins + 1)

    counts, _ = np.histogram(distances, bins=bins)

    # Volume of each spherical shell: (4/3)π(r_outer³ - r_inner³)
    shell_volumes = (4.0 / 3.0) * np.pi * (bins[1:] ** 3 - bins[:-1] ** 3)
    shell_volumes = np.where(shell_volumes < 1e-12, 1e-12, shell_volumes)  # avoid div-by-zero
    density = counts / shell_volumes

    # 3-bin moving average to smooth noise
    kernel = np.ones(3) / 3.0
    density_smooth = np.convolve(density, kernel, mode="same")

    # Derivative: drop at bin i means density[i+1] - density[i]
    gradient = np.diff(density_smooth)

    # Largest negative step → sharpest density collapse
    drop_idx = int(np.argmin(gradient))        # index in gradient array
    dropoff_radius = float(bins[drop_idx + 1]) # right edge of that bin

    # Print a mini density profile to the log
    logger.info("Radial density profile (bin centre → density):")
    bin_centres = 0.5 * (bins[:-1] + bins[1:])
    for i, (r, d) in enumerate(zip(bin_centres, density_smooth)):
        marker = " <-- DROP" if i == drop_idx else ""
        logger.info(f"  r={r:7.3f}  density={d:.4e}{marker}")

    logger.info(f"Auto-detected density dropoff radius: {dropoff_radius:.4f}")
    return dropoff_radius


def filter_points_around_origin(ply_path: str, output_ply_path: str, margin: float) -> None:
    """
    Load a PLY point cloud and keep only points within `margin` distance from
    the origin (0, 0, 0). Saves the filtered cloud to output_ply_path.
    """
    logger.info(f"Loading point cloud for origin filtering: {ply_path}")
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)

    if points.size == 0:
        logger.warning("Point cloud is empty — skipping origin filtering.")
        return

    distances = np.linalg.norm(points, axis=1)
    mask = distances <= margin
    kept = int(mask.sum())
    total = len(points)
    logger.info(f"Origin filtering (margin={margin:.4f}): keeping {kept}/{total} points")

    pcd_filtered = pcd.select_by_index(np.where(mask)[0].tolist())
    o3d.io.write_point_cloud(output_ply_path, pcd_filtered)
    logger.info(f"Filtered point cloud saved to: {output_ply_path}")


def _largest_subfolder(parent: Path) -> str:
    """Return the path of the largest direct subfolder under parent."""
    subdirs = [p for p in parent.iterdir() if p.is_dir() and not p.is_symlink()]
    if not subdirs:
        raise SystemExit(f"No subfolders found under: {parent}")

    sizes = [(d, _folder_size_bytes(d)) for d in subdirs]
    biggest_dir, biggest_size = max(sizes, key=lambda t: t[1])

    logger.info(f"Parent:  {parent}")
    logger.info(f"Largest: {biggest_dir.name}")
    logger.info(f"Path:    {biggest_dir}")
    logger.info(f"Size:    {_human(biggest_size)} ({biggest_size} bytes)")
    return str(biggest_dir)


def run_colmap_command(command, description):
    """
    Run a COLMAP command and handle errors.
    
    Args:
        command: Command string to execute
        description: Description of the operation
    """
    logger.info(f"Running: {description}")
    logger.debug(f"Command: {command}")
    
    exit_code = os.system(command)
    
    if exit_code != 0:
        logger.error(f"{description} failed with exit code {exit_code}")
        sys.exit(1)
    
    logger.info(f"{description} completed successfully")


def run_colmap_pipeline(source_path, image_path, colmap_executable, camera_model, use_gpu):
    """
    Run the complete COLMAP reconstruction pipeline.
    
    Args:
        source_path: Path to source directory for COLMAP outputs
        image_path: Path to images used by COLMAP
        colmap_executable: COLMAP executable path or command
        camera_model: Camera model (e.g., 'OPENCV')
        use_gpu: Whether to use GPU acceleration
    """
    colmap_cmd = '"{}"'.format(colmap_executable) if len(colmap_executable) > 0 else "colmap"
    use_gpu_flag = 1 if use_gpu else 0
    image_path_with_space = f"{image_path} "
    
    # ============================================================================
    # Feature Extraction
    # ============================================================================
    feat_extraction_cmd = (
        f'{colmap_cmd} feature_extractor '
        f'--database_path {source_path}/distorted/database.db '
        f'--image_path {image_path_with_space}'
        f'--ImageReader.single_camera 1 '
        f'--ImageReader.camera_model {camera_model} '
        f'--FeatureExtraction.type ALIKED_N16ROT '
        # f'--ImageReader.mask_path {source_path}/masks '
        f'--FeatureExtraction.num_threads 32 '
        # f'--SiftExtraction.domain_size_pooling true '
        # f'--FeatureExtraction.max_image_size 80000 '
        f'--AlikedExtraction.max_num_features 10000 '
        f'--FeatureExtraction.use_gpu {use_gpu_flag}'
    )
    
    os.makedirs(os.path.join(source_path, "distorted", "sparse"), exist_ok=True)
    run_colmap_command(feat_extraction_cmd, "Feature extraction")
    
    # ============================================================================
    # Feature Matching
    # ============================================================================
    feat_matching_cmd = (
        f'{colmap_cmd} exhaustive_matcher '
        f'--database_path {source_path}/distorted/database.db '
        f'--FeatureMatching.type ALIKED_LIGHTGLUE '
        f'--SiftMatching.cross_check 1 '
        f'--FeatureMatching.max_num_matches 1000000 '
        f'--SiftMatching.max_distance 0.7 '
        f'--TwoViewGeometry.max_error 4.0 '
        f'--FeatureMatching.guided_matching false '
        f'--FeatureMatching.use_gpu {use_gpu_flag} '
        f'--FeatureMatching.gpu_index 0'
    )
    
    run_colmap_command(feat_matching_cmd, "Feature matching")
    
    # ============================================================================
    # Bundle Adjustment (Mapper)
    # ============================================================================
    mapper_cmd = (
        f'{colmap_cmd} mapper '
        f'--database_path {source_path}/distorted/database.db '
        f'--image_path {image_path_with_space}'
        f'--output_path {source_path}/distorted/sparse '
        f'--Mapper.ba_refine_principal_point 1 '
        f'--Mapper.ba_refine_extra_params 1 '
        f'--Mapper.filter_max_reproj_error 2.0 '
        f'--Mapper.init_max_error 2.0 '
        f'--Mapper.multiple_models 1 '
        f'--Mapper.ba_global_function_tolerance=0.000001'
    )
    
    run_colmap_command(mapper_cmd, "Bundle adjustment (Mapper)")

    sparse_models_parent = Path(source_path) / "distorted" / "sparse"
    mapper_sparse_input = _largest_subfolder(sparse_models_parent)
    
    # ============================================================================
    # Model Orientation Alignment
    # ============================================================================
    os.makedirs(os.path.join(source_path, "distorted_sparse_aligned"), exist_ok=True)
    
    aligner_cmd = (
        f'{colmap_cmd} model_orientation_aligner '
        f'--method MANHATTAN-WORLD '
        f'--image_path {image_path_with_space}'
        f'--input_path {mapper_sparse_input} '
        f'--output_path {source_path}/distorted_sparse_aligned'
    )
    
    run_colmap_command(aligner_cmd, "Model orientation alignment")
    
    # ============================================================================
    # Image Undistortion
    # ============================================================================
    img_undist_cmd = (
        f'{colmap_cmd} image_undistorter '
        f'--image_path {image_path_with_space}'
        f'--input_path {source_path}/distorted_sparse_aligned '
        f'--output_path {source_path} '
        f'--output_type COLMAP'
    )
    
    run_colmap_command(img_undist_cmd, "Image undistortion")
    
    # ============================================================================
    # Point Filtering
    # ============================================================================
    os.makedirs(os.path.join(source_path, "sparse_filtered"), exist_ok=True)
    
    filtering_cmd = (
        f'{colmap_cmd} point_filtering '
        f'--input_path {source_path}/sparse '
        f'--output_path {source_path}/sparse_filtered '
        f'--min_track_len 3'
    )
    
    run_colmap_command(filtering_cmd, "Point filtering")

    # ============================================================================
    # Model Conversion to TXT
    # ============================================================================
    os.makedirs(os.path.join(source_path, "txt"), exist_ok=True)

    txt_converter_cmd = (
        f'{colmap_cmd} model_converter '
        f'--input_path {source_path}/sparse_filtered '
        f'--output_path {source_path}/txt '
        f'--output_type TXT'
    )

    run_colmap_command(txt_converter_cmd, "Model conversion to TXT")

    # ============================================================================
    # Model Conversion to PLY
    # ============================================================================
    converter_cmd = (
        f'{colmap_cmd} model_converter '
        f'--input_path {source_path}/sparse_filtered '
        f'--output_path {source_path}/points3D_filtered_sparse.ply '
        f'--output_type PLY'
    )
    
    run_colmap_command(converter_cmd, "Model conversion to PLY")

    _organize_sparse(source_path)
    _log_colmap_done(source_path, image_path)

    return os.path.join(source_path, "points3D_filtered_sparse.ply")


def run_origin_filtering(ply_path: str, margin: float, auto: bool = False,
                         density_bins: int = 50) -> None:
    """
    Run origin-based point cloud filtering.  When auto=True the margin is
    derived from the density-dropoff analysis; otherwise the supplied margin is
    used directly.
    """
    import open3d as _o3d  # already imported at module level; alias avoids shadowing
    pcd = _o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)

    if points.size == 0:
        logger.warning("Point cloud is empty — skipping origin filtering.")
        return

    if auto:
        logger.info("Auto-detecting density dropoff radius...")
        margin = detect_density_dropoff(points, n_bins=density_bins)
        logger.info(f"Using auto-detected margin: {margin:.4f}")
    else:
        logger.info(f"Using manual margin: {margin}")

    output_path = ply_path.replace(".ply", "_origin_filtered.ply")
    filter_points_around_origin(ply_path, output_path, margin)


def _organize_sparse(source_path: str) -> None:
    # ============================================================================
    # Organize sparse output
    # ============================================================================
    sparse_dir = os.path.join(source_path, "sparse")
    sparse_0_dir = os.path.join(sparse_dir, "0")
    os.makedirs(sparse_0_dir, exist_ok=True)
    
    # Copy files from sparse/ to sparse/0/
    if os.path.exists(sparse_dir):
        for file in os.listdir(sparse_dir):
            if file == "0":
                continue
            source_file = os.path.join(sparse_dir, file)
            dest_file = os.path.join(sparse_0_dir, file)
            if os.path.isfile(source_file):
                shutil.copy2(source_file, dest_file)
                logger.debug(f"Copied {file} to sparse/0/")


def _log_colmap_done(source_path: str, image_path: str) -> None:
    logger.info("=" * 70)
    logger.info("COLMAP pipeline completed successfully!")
    logger.info("=" * 70)
    logger.info(f"Results directory: {source_path}")
    logger.info(f"3D points (PLY): {source_path}/points3D_filtered_sparse.ply")
    logger.info(f"Origin-filtered PLY: {source_path}/points3D_filtered_sparse_origin_filtered.ply")
    logger.info(f"Sparse model: {source_path}/sparse_filtered/")
    logger.info(f"COLMAP source images: {image_path}")
    logger.info(f"Undistorted images: {source_path}/images/")
    logger.info(f"Background-removed undistorted images: {source_path}/images_rembg/")
    logger.info(f"Foreground frame layout: {source_path}/ims/")
    logger.info(f"Mask layout: {source_path}/seg/")


# Data structures for COLMAP reading
Camera = collections.namedtuple("Camera", ["id", "width", "height", "params"])
ColmapImage = collections.namedtuple("ColmapImage", ["id", "camera_id", "name", "cam_pose"])


def read_camposes_from_colmap(images_txt_path):
    """
    Read camera poses from COLMAP images.txt file.
    
    Args:
        images_txt_path: Path to COLMAP images.txt
        
    Returns:
        Tuple of (w2c_matrices, sorted_camera_ids)
    """
    img_dict = {}
    with open(images_txt_path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                rot = R_scipy.from_quat(qvec, scalar_first=True)
                R = rot.as_matrix()
                pose = np.eye(4)
                pose[:3, :3] = R
                pose[:3, 3] = tvec
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                img_dict[image_id] = ColmapImage(image_id, camera_id, image_name, pose)
    
    w2c = []
    sorted_img_dict = dict(sorted(img_dict.items(), key=lambda x: int(x[1].name.split(".")[0])))
    sorted_camera_ids = [val.camera_id for val in sorted_img_dict.values()]
    for val in sorted_img_dict.values():
        w2c.append(val.cam_pose.tolist())
    
    return w2c, sorted_camera_ids


def read_intrinsics_from_colmap(cameras_txt_path, sorted_camera_ids, target_width=None, target_height=None):
    """
    Read intrinsic matrices from COLMAP cameras.txt file.
    
    Args:
        cameras_txt_path: Path to COLMAP cameras.txt
        sorted_camera_ids: List of camera IDs in sorted order
        target_width: Target width for scaling (optional)
        target_height: Target height for scaling (optional)
        
    Returns:
        List of K matrices
    """
    cameras = {}
    with open(cameras_txt_path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                assert model == "PINHOLE", "Only PINHOLE camera model is supported"
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(camera_id, width, height, params)
    
    cam_intrinsics = []
    # IMPORTANT: do not deduplicate `sorted_camera_ids`.
    # With COLMAP single_camera mode, many images share the same camera_id,
    # but we still need one K matrix per image entry to align with w2c/fn lists.
    for cam_id in sorted_camera_ids:
        fx, fy, cx, cy = cameras[cam_id].params.tolist()
        
        # Apply scaling if target dimensions provided
        if target_width is not None and target_height is not None:
            scale_x = target_width / cameras[cam_id].width
            scale_y = target_height / cameras[cam_id].height
            fx = fx * scale_x
            fy = fy * scale_y
            cx = cx * scale_x
            cy = cy * scale_y
        
        K_new = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1.0]
        ])
        cam_intrinsics.append(K_new.tolist())
    
    return cam_intrinsics


def create_training_config_from_template(template_path, output_config_path, dataset_name, data_dir, output_dir, init_ply_name):
    """Create training config YAML while preserving template formatting style."""
    if not os.path.isfile(template_path):
        logger.warning(f"Training config template not found: {template_path}")
        return

    with open(template_path, "r", encoding="utf-8") as f:
        template_text = f.read()

    # Replace only selected scalar fields so list formatting from template is preserved.
    template_text = re.sub(
        r"(?m)^sequence:\s*.*$",
        f'sequence: "{dataset_name}"',
        template_text,
    )
    template_text = re.sub(
        r"(?m)^data_dir:\s*.*$",
        f'data_dir: "{data_dir}"',
        template_text,
    )
    template_text = re.sub(
        r"(?m)^output_dir:\s*.*$",
        f'output_dir: "{output_dir}"',
        template_text,
    )
    template_text = re.sub(
        r"(?m)^init_params_file:\s*.*$",
        f"init_params_file: {init_ply_name}",
        template_text,
    )
    template_text = re.sub(
        r"(?m)^exp_name_prefix:\s*.*$",
        f'exp_name_prefix: "{dataset_name}_"',
        template_text,
    )

    with open(output_config_path, "w", encoding="utf-8") as f:
        f.write(template_text)

    logger.info(f"✓ Generated training config: {output_config_path}")


def run_training(train_config_path):
    """Run training script in the currently active environment."""
    repo_root = Path(__file__).resolve().parent
    train_script = repo_root / "train.py"

    if not train_script.exists():
        logger.error(f"Training script not found: {train_script}")
        return False
    if not os.path.isfile(train_config_path):
        logger.error(f"Training config not found: {train_config_path}")
        return False

    cmd = [
        "python", "-u", str(train_script),
        "--config", train_config_path,
    ]
    logger.info("=" * 70)
    logger.info("Starting training in active environment...")
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info("=" * 70)

    try:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        subprocess.run(cmd, check=True, cwd=str(repo_root), capture_output=False, env=env)
        logger.info("✓ Training finished successfully")
        return True
    except subprocess.CalledProcessError as exc:
        logger.error(f"Training failed with return code {exc.returncode}")
    return False


def create_train_meta(output_dir, dataset_name="dataset", frame_start=0, frame_end=None,
                      train_config_template="configs/config_template.yaml",
                      train_config_name="train_config.yaml"):
    """
    Create train_meta.json from COLMAP output with symlinks to ims and seg folders.
    
    Args:
        output_dir: Main output directory containing txt/, ims/, seg/ folders
        dataset_name: Name of the dataset subfolder (default: 'dataset')
        frame_start: Start frame index (default: 0)
        frame_end: End frame index (default: all frames)
    """
    logger.info("\n" + "=" * 70)
    logger.info("Creating train_meta.json...")
    logger.info("=" * 70)
    # Create dataset subfolder
    dataset_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    logger.info(f"Dataset directory: {dataset_dir}")
    # Verify input paths
    txt_dir = os.path.join(output_dir, "txt")
    ims_dir = os.path.join(output_dir, "ims")
    seg_dir = os.path.join(output_dir, "seg")
    ply_file = os.path.join(output_dir, "points3D_filtered_sparse_origin_filtered.ply")
    cameras_txt = os.path.join(txt_dir, "cameras.txt")
    images_txt = os.path.join(txt_dir, "images.txt")
    
    # Check paths
    if not os.path.isdir(txt_dir):
        logger.error(f"txt directory not found: {txt_dir}")
        return None
    if not os.path.isdir(ims_dir):
        logger.error(f"ims directory not found: {ims_dir}")
        return None
    if not os.path.isdir(seg_dir):
        logger.error(f"seg directory not found: {seg_dir}")
        return None
    if not os.path.isfile(cameras_txt):
        logger.error(f"cameras.txt not found: {cameras_txt}")
        return None
    if not os.path.isfile(images_txt):
        logger.error(f"images.txt not found: {images_txt}")
        return None
    if not os.path.isfile(ply_file):
        logger.error(f"PLY file not found: {ply_file}")
        return None
    
    # Read camposes and intrinsics from COLMAP txt output
    logger.info(f"Reading camera poses from {images_txt}")
    w2c, sorted_camera_ids = read_camposes_from_colmap(images_txt)
    
    logger.info(f"Reading intrinsics from {cameras_txt}")
    # Read original dimensions from cameras.txt
    cameras = {}
    with open(cameras_txt, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                width = int(elems[2])
                height = int(elems[3])
                cameras[camera_id] = (width, height)
                break  # Get dimensions from first camera
    
    # Get width/height from first camera (all should be same)
    first_cam_id = min(cameras.keys())
    cam_width, cam_height = cameras[first_cam_id]
    logger.info(f"Camera dimensions from cameras.txt: {cam_width} x {cam_height}")
    
    k_matrices = read_intrinsics_from_colmap(cameras_txt, sorted_camera_ids, None, None)
    
    # Build camera folder list from actual ims layout: ims/<camera_folder>/000000.png
    camera_folders = [
        d for d in os.listdir(ims_dir)
        if os.path.isdir(os.path.join(ims_dir, d))
    ]
    camera_folders.sort(key=lambda x: int(x) if x.isdigit() else x)

    if not camera_folders:
        logger.error(f"No camera folders found in ims directory: {ims_dir}")
        return None

    # Keep only folders that actually contain the expected image file.
    valid_camera_folders = [
        d for d in camera_folders
        if os.path.isfile(os.path.join(ims_dir, d, "000000.png"))
    ]
    if not valid_camera_folders:
        logger.error(f"No valid camera images found under {ims_dir}/*/000000.png")
        return None

    # Align metadata list lengths to avoid out-of-range indexing.
    num_cams = min(len(valid_camera_folders), len(w2c), len(k_matrices))
    if num_cams == 0:
        logger.error("No valid cameras available after alignment")
        return None
    if num_cams != len(valid_camera_folders) or num_cams != len(w2c):
        logger.warning(
            f"Camera count mismatch: ims={len(valid_camera_folders)}, poses={len(w2c)}, intrinsics={len(k_matrices)}. "
            f"Using first {num_cams} cameras."
        )

    valid_camera_folders = valid_camera_folders[:num_cams]
    w2c = w2c[:num_cams]
    k_matrices = k_matrices[:num_cams]

    # Current pipeline output has a single timestep image (000000.png) per camera folder.
    # Keep CLI compatibility for frame_start/frame_end but clamp to one timestep.
    timestep_start = 0 if frame_start is None else max(0, frame_start)
    timestep_end = 1 if frame_end is None else max(timestep_start, min(frame_end, 1))
    logger.info(f"Found {num_cams} camera folders in ims directory")
    logger.info(f"Aggregating timesteps {timestep_start} to {timestep_end} (exclusive)")

    # Reuse the canonical aggregation helper used by make_data_for_dynamicgs.py.
    # It builds cam_id/k/w2c structure for the requested interval.

    fn_all, cam_id_all, k_all, w2c_all = aggregate_cam_params_and_images(
        k_matrices, w2c, (timestep_start, timestep_end)
    )

    # Remap helper-generated image paths to actual folder names from ims/.
    # Current pipeline stores one image per camera folder: <folder>/000000.png.
    remapped_fn_all = []
    for image_list in fn_all:
        remapped_image_list = []
        for image_path in image_list:
            cam_idx = int(image_path.split("/", 1)[0])
            remapped_image_list.append(f"{valid_camera_folders[cam_idx]}/000000.png")
        remapped_fn_all.append(remapped_image_list)

    # Create data structure for train_meta.json
    data = {
        'w': cam_width,
        'h': cam_height,
        'k': k_all,
        'w2c': w2c_all,
        'fn': remapped_fn_all,
        'cam_id': cam_id_all,
    }
    
    # Write train_meta.json to dataset folder
    meta_json_path = os.path.join(dataset_dir, 'train_meta.json')
    with open(meta_json_path, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"✓ Created train_meta.json: {meta_json_path}")
    
    # Copy PLY file to dataset folder
    ply_dest = os.path.join(dataset_dir, os.path.basename(ply_file))
    if ply_dest != ply_file and os.path.isfile(ply_file):
        shutil.copy2(ply_file, ply_dest)
        logger.info(f"✓ Copied PLY file: {ply_dest}")

    # Generate a ready-to-run training config in the dataset folder.
    train_config_out = os.path.join(dataset_dir, train_config_name)
    create_training_config_from_template(
        train_config_template,
        train_config_out,
        dataset_name=dataset_name,
        data_dir=output_dir,
        output_dir=os.path.join(output_dir, "training_output"),
        init_ply_name=os.path.basename(ply_dest),
    )
    
    # Create symlinks for ims and seg directories in dataset folder
    ims_link = os.path.join(dataset_dir, "ims")
    seg_link = os.path.join(dataset_dir, "seg")
    
    for link_path, src_path, name in [(ims_link, ims_dir, "ims"), (seg_link, seg_dir, "seg")]:
        if os.path.islink(link_path):
            os.unlink(link_path)
        elif os.path.isdir(link_path) and link_path not in [ims_dir, seg_dir]:
            # Only remove if it's not the original directory
            shutil.rmtree(link_path)
        
        if link_path != src_path:
            os.symlink(os.path.abspath(src_path), link_path)
            logger.info(f"✓ Created symlink: {link_path} -> {src_path}")
    
    logger.info("=" * 70)
    logger.info(f"✓ Dataset preparation complete: {dataset_dir}")
    logger.info("=" * 70)
    return train_config_out


def _copy_splat_plys_to_root(training_output_dir: str, dataset_name: str, root_dir: str) -> None:
    """Copy the trained splat PLY file into root_dir as splat.ply."""
    search_root = os.path.join(training_output_dir, dataset_name)
    ply_files = sorted(Path(search_root).rglob("params_*.ply")) if os.path.isdir(search_root) else []
    if not ply_files:
        logger.warning(f"No splat PLY files found under {search_root}")
        return
    src = ply_files[-1]  # take the last (highest timestep index)
    dest = os.path.join(root_dir, "splat.ply")
    shutil.copy2(str(src), dest)
    logger.info(f"Splat PLY saved: {dest}")


def main():
    """Main entry point."""
    args = parse_arguments()
    
    logger.info("=" * 70)
    logger.info("MP4 to COLMAP Pipeline")
    logger.info("=" * 70)
    logger.info(f"Input video: {args.input}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Frame interval: {args.frame_interval}")
    logger.info(f"Target frames: {args.target_frames}")
    logger.info(f"Camera model: {args.camera}")
    logger.info(f"GPU enabled: {not args.no_gpu}")
    logger.info(f"Background removal enabled: {args.background_removal}")
    logger.info(f"rembg model: {args.rembg_model}")
    logger.info("=" * 70)
    
    undistorted_images_dir = os.path.join(args.output, "images")
    rembg_dir = os.path.join(args.output, "images_rembg")
    ims_dir = os.path.join(args.output, "ims")
    seg_dir = os.path.join(args.output, "seg")

    if args.skip_to_meta_training:
        logger.info("Skipping extraction/COLMAP/rembg and jumping to create_meta + training")
    else:
        # Verify input
        verify_input_video(args.input)

        # Setup output directories
        input_dir = setup_directories(args.output)

        # Extract frames

        if not args.skip_frame_extraction:
            extract_around_n_frames(
                args.input,
                input_dir,
                target_frames=args.target_frames,
                fallback_interval=args.frame_interval,
            )
        else:
            logger.info("Skipping frame extraction (using existing frames)")
            if not os.path.exists(input_dir) or not os.listdir(input_dir):
                logger.warning(f"Input directory is empty: {input_dir}")

        # Run COLMAP pipeline
        ply_path = os.path.join(args.output, "points3D_filtered_sparse.ply")
        if args.colmap:
            logger.info("\nStarting COLMAP pipeline...")
            logger.info("=" * 70)
            ply_path = run_colmap_pipeline(
                args.output,
                input_dir,
                args.colmap_executable,
                args.camera,
                not args.no_gpu
            )
       
        # ============================================================================
        # Origin-based point cloud filtering
        # ============================================================================
        if args.origin_filter:
            logger.info("\nFiltering 3-D points around origin...")
            if args.auto_margin:
                logger.info(f"Mode: auto density-dropoff detection ({args.density_bins} bins)")
            else:
                logger.info(f"Mode: manual margin={args.origin_margin}")
            logger.info("=" * 70)
            if os.path.exists(ply_path):
                run_origin_filtering(ply_path, args.origin_margin,
                                     auto=args.auto_margin, density_bins=args.density_bins)
            else:
                logger.warning(f"PLY file not found for origin filtering: {ply_path}")

        if args.background_removal:
            logger.info("\nStarting background removal on undistorted COLMAP images...")
            logger.info(f"rembg input directory: {undistorted_images_dir}")
            logger.info(f"rembg output directory: {rembg_dir}")
            logger.info("=" * 70)
            remove_backgrounds(undistorted_images_dir, rembg_dir, args.rembg_model)
            reorganize_rembg_outputs(rembg_dir, ims_dir, seg_dir)
    
    # Create train_meta.json if requested
    if args.create_meta:
        frame_end = args.frame_end
        if frame_end is None:
            # Count frames from ims directory
            if os.path.exists(ims_dir):
                frame_end = len([d for d in os.listdir(ims_dir) if os.path.isdir(os.path.join(ims_dir, d))])
        
        train_config_path = create_train_meta(
            args.output,
            dataset_name=args.dataset_name,
            frame_start=args.frame_start,
            frame_end=frame_end,
            train_config_template=args.train_config_template,
            train_config_name=args.train_config_name,
        )

        if args.run_training:
            if train_config_path is None:
                logger.error("Skipping training because metadata/config generation failed")
            else:
                success = run_training(train_config_path)
                if success:
                    _copy_splat_plys_to_root(
                        os.path.join(args.output, "training_output"),
                        args.dataset_name,
                        args.output,
                    )
    
    logger.info("\nPipeline completed!")
    logger.info(f"Check output directory for results: {args.output}")
  

if __name__ == "__main__":
    main()
