# MP4 to Splat Workflow Guide

This guide describes the pipeline implemented in [mp4_to_splat.py](mp4_to_splat.py) in five stages:
1. Extract frames from MP4
2. Run COLMAP reconstruction
3. Remove background with rembg
4. Rebin/filter the generated point cloud
5. Run training

## Prerequisites — Set Up and Activate Environment First

The script itself (not just training) depends on packages such as `numpy`, `open3d`, `scipy`, `PIL` and `rembg` that live inside the conda environment.

**Run these steps once before anything else:**

```bash
git clone --branch training --recurse-submodules https://github.com/Spaceport-Project/Dynamic3DGaussians.git
cd Dynamic3DGaussians
conda env create --file environment.yml
conda activate dynamic_gaussians
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
pip install submodules/*
```




## 1) Extraction of MP4 File

The script extracts frames from the input video into an `input/` folder under your output directory.

- Primary mode: auto-select frame interval to target approximately `--target_frames`.
- Fallback mode: use fixed `--frame_interval` when frame count metadata is unavailable.

Key CLI arguments:
- `--input`: input video path
- `--output`: output root path
- `--target_frames`: desired extracted frame count (default: 100)
- `--frame_interval`: fallback interval (default: 30)
- `--skip_frame_extraction`: reuse existing frames in `output/input`

Example:

```bash
conda activate dynamic_gaussians

python mp4_to_splat.py \
  --input /path/to/video.mp4 \
  --output /path/to/run_output \
  --target_frames 120
```

Expected output from this stage:
- `output/input/0.png`, `output/input/1.png`, ...

## 2) Running COLMAP

After frame extraction, the script runs COLMAP end-to-end:
- Feature extraction
- Feature matching
- Mapper / bundle adjustment
- Orientation alignment
- Image undistortion
- Point filtering
- Model conversion to TXT and PLY

Key CLI arguments:
- `--camera`: COLMAP camera model (default: `OPENCV`)
- `--colmap_executable`: custom COLMAP binary path
- `--no_gpu`: disable GPU acceleration

Important output folders/files:
- `output/distorted/` (intermediate COLMAP DB and sparse models)
- `output/images/` (undistorted images)
- `output/txt/` (`cameras.txt`, `images.txt`, etc.)
- `output/points3D_filtered_sparse.ply`

## 3) Background Removal with rembg

After COLMAP undistortion, the script removes image backgrounds using `rembg`.

What happens:
- Input images are read from `output/images/`
- `rembg` processes each image and writes outputs to `output/images_rembg/`
- The script reorganizes outputs into:
  - `output/ims/<frame_id>/000000.png` (foreground RGB)
  - `output/seg/<frame_id>/000000.png` (binary mask)

Key CLI arguments:
- `--rembg_model` (default: `birefnet-general`)
- `--skip_background_removal` (reuse existing rembg outputs)

Execution behavior:
- rembg is executed directly in the currently active environment: `rembg p -m <model> <input_dir> <output_dir>`

Stage output:
- `output/images_rembg/`
- `output/ims/`
- `output/seg/`

## 4) Rebinning / Filtering the COLMAP Point Cloud

In this script, the "rebinning" stage is implemented as origin-centered filtering of the COLMAP point cloud.

What happens:
- Load `points3D_filtered_sparse.ply`
- Compute point distance to origin `(0,0,0)`
- Keep points inside a radius (margin)
- Save filtered cloud as `points3D_filtered_sparse_origin_filtered.ply`

Modes:
- Manual radius: `--origin_margin` (default: `5.0`)
- Auto radius: `--auto_margin` with radial density dropoff detection
  - Controlled by `--density_bins` (default: `50`)

Key CLI arguments:
- `--origin_margin`
- `--auto_margin`
- `--density_bins`

Output of this stage:
- `output/points3D_filtered_sparse_origin_filtered.ply`

## 5) Running Training

Training is optional and happens after metadata generation.

The script can:
- Build dataset metadata (`train_meta.json`) via `--create_meta`
- Generate a training config from template
- Launch training in the currently active environment

Key CLI arguments:
- `--create_meta`
- `--dataset_name`
- `--train_config_template`
- `--train_config_name`
- `--run_training`

Training command internally executed:
- `python -u train.py --config <generated_config>`

Important generated dataset artifacts:
- `output/<dataset_name>/train_meta.json`
- `output/<dataset_name>/<train_config_name>`
- `output/<dataset_name>/points3D_filtered_sparse_origin_filtered.ply`
- Symlinks:
  - `output/<dataset_name>/ims -> output/ims`
  - `output/<dataset_name>/seg -> output/seg`

## Full Example (All 5 Stages)

```bash
conda activate dynamic_gaussians

python mp4_to_splat.py \
  --input /path/to/video.mp4 \
  --output /path/to/run_output \
  --target_frames 80 \
  --camera OPENCV \
  --auto_margin \
  --create_meta \
  --dataset_name my_dataset \
  --run_training
```

## Useful Variants

Run only metadata + optional training from existing outputs:

```bash
conda activate dynamic_gaussians

python mp4_to_splat.py \
  --input /path/to/video.mp4 \
  --output /path/to/run_output \
  --skip_to_meta_training \
  --create_meta \
  --dataset_name my_dataset \
  --run_training
```

Skip background removal:

```bash
conda activate dynamic_gaussians

python mp4_to_splat.py \
  --input /path/to/video.mp4 \
  --output /path/to/run_output \
  --skip_background_removal
```
