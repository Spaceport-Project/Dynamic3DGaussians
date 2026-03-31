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

### Step 1 — Install CUDA 11.8 and 12.6 on ubuntu 22.04 based systems

```bash
# Add NVIDIA package repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install both CUDA versions
sudo apt-get -y install cuda-toolkit-11-8
sudo apt-get -y install cuda-toolkit-12-6
```

### Step 1b — Install cuDNN Libraries

#### cuDNN for CUDA 11.8

```bash
wget https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-8.7.0.84_cuda11-archive.tar.xz
tar -xf cudnn-linux-x86_64-8.7.0.84_cuda11-archive.tar.xz
cd cudnn-linux-x86_64-8.7.0.84_cuda11-archive
sudo cp include/cudnn*.h /usr/local/cuda-11.8/include
sudo cp lib/libcudnn* /usr/local/cuda-11.8/lib64
sudo chmod a+r /usr/local/cuda-11.8/include/cudnn*.h /usr/local/cuda-11.8/lib64/libcudnn*
cd ..
rm cudnn-linux-x86_64-8.7.0.84_cuda11-archive.tar.xz  cudnn-linux-x86_64-8.7.0.84_cuda11-archive/ -rf
```

#### cuDNN for CUDA 12.6

```bash
wget https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.1.1.17_cuda12-archive.tar.xz
tar -xf cudnn-linux-x86_64-9.1.1.17_cuda12-archive.tar.xz
cd cudnn-linux-x86_64-9.1.1.17_cuda12-archive
sudo cp include/cudnn*.h /usr/local/cuda-12.6/include
sudo cp lib/libcudnn* /usr/local/cuda-12.6/lib64
sudo chmod a+r /usr/local/cuda-12.6/include/cudnn*.h /usr/local/cuda-12.6/lib64/libcudnn*
cd ..
rm cudnn-linux-x86_64-9.1.1.17_cuda12-archive.tar.xz  cudnn-linux-x86_64-9.1.1.17_cuda12-archive/ -rf

```

### Step 2 — Set Up Easy CUDA Switching

```bash
# Create CUDA switcher script
cat > ~/.cuda_switcher.sh << 'EOF'
#!/bin/bash
cuda_switch() {
    if [ -z "$1" ]; then
        echo "Current CUDA version:"
        nvcc --version 2>/dev/null | grep "release" || echo "CUDA not found in PATH"
        echo ""
        echo "Available CUDA installations:"
        ls -d /usr/local/cuda-* 2>/dev/null | xargs -I {} basename {} || echo "No CUDA versions found"
        echo ""
        echo "Usage: cuda_switch <version>"
        echo "Example: cuda_switch 11.8"
        return 0
    fi

    local version=$1
    local cuda_path="/usr/local/cuda-${version}"

    if [ ! -d "$cuda_path" ]; then
        echo "Error: CUDA $version not found at $cuda_path"
        echo "Available versions:"
        ls -d /usr/local/cuda-* 2>/dev/null | xargs -I {} basename {}
        return 1
    fi

    export CUDA_HOME="$cuda_path"
    export PATH="$cuda_path/bin:$(echo $PATH | sed "s|/usr/local/cuda-[^/]*/bin:||g")"
    export LD_LIBRARY_PATH="$cuda_path/lib64:$(echo $LD_LIBRARY_PATH | sed "s|/usr/local/cuda-[^/]*/lib64:||g")"

    echo "Switched to CUDA $version"
    echo "CUDA_HOME: $CUDA_HOME"
    nvcc --version | grep "release"
}

alias cuda11='cuda_switch 11.8'
alias cuda12='cuda_switch 12.6'
alias cuda_check='nvcc --version && echo "" && echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"'
EOF

# Add to bashrc and reload
echo "source ~/.cuda_switcher.sh" >> ~/.bashrc
source ~/.bashrc
cuda12
```

Quick commands available after this:
- `cuda11` — Switch to CUDA 11.8
- `cuda12` — Switch to CUDA 12.6
- `cuda_switch` — See available versions and current setup
- `cuda_check` — Verify current CUDA and library paths

### Step 3 — Build and Install COLMAP 4 from Source

COLMAP must be compiled from source to get CUDA support. Switch to CUDA 12.6 first, then build.

```bash
cuda12

# Install system dependencies
sudo apt-get install -y \
    git \
    cmake \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libopenimageio-dev \
    openimageio-tools \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libgmock-dev \
    libsqlite3-dev \
    libglew-dev \
    qt6-base-dev \
    libqt6opengl6-dev \
    libqt6openglwidgets6 \
    libcgal-dev \
    libceres-dev \
    libsuitesparse-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libmkl-full-dev



# Download and build COLMAP 4.0.2
wget https://github.com/colmap/colmap/archive/refs/tags/4.0.2.tar.gz
tar -xzf 4.0.2.tar.gz
cd colmap-4.0.2
mkdir build
cd build
cmake .. -GNinja \
    -DCMAKE_CUDA_ARCHITECTURES=86
ninja
sudo ninja install
cd ../..

# Verify
colmap -h
```


### Step 4 — Clone and Set Up the Environment

```bash
git clone --branch mp42splat --recursive https://github.com/Spaceport-Project/Dynamic3DGaussians.git
cd Dynamic3DGaussians
conda env create --file environment.yml
conda activate dynamic_gs
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install kornia
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
conda activate dynamic_gs

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
- `--colmap`: run the COLMAP reconstruction pipeline (omit to skip)

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
- `--background_removal`: run background removal (omit to skip)

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
- `--origin_filter`: run origin-based point cloud filtering (omit to skip)
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
conda activate dynamic_gs

python mp4_to_splat.py \
  --input /path/to/video.mp4 \
  --output /path/to/run_output \
  --target_frames 70 \
  --camera OPENCV \
  --colmap \
  --origin_filter \
  --auto_margin \
  --background_removal \
  --create_meta \
  --dataset_name my_dataset \
  --run_training
```

## Useful Variants

Run only metadata + optional training from existing outputs:

```bash
conda activate dynamic_gs

python mp4_to_splat.py \
  --input /path/to/video.mp4 \
  --output /path/to/run_output \
  --skip_to_meta_training \
  --create_meta \
  --dataset_name my_dataset \
  --run_training
```

Run COLMAP + origin filter, skip background removal:

```bash
conda activate dynamic_gs

python mp4_to_splat.py \
  --input /path/to/video.mp4 \
  --output /path/to/run_output \
  --colmap \
  --origin_filter
```

Extract frames only (skip COLMAP and everything after):

```bash
conda activate dynamic_gs

python mp4_to_splat.py \
  --input /path/to/video.mp4 \
  --output /path/to/run_output \
  --target_frames 120
```

Run background removal + metadata on an existing COLMAP reconstruction:

```bash
conda activate dynamic_gs

python mp4_to_splat.py \
  --input /path/to/video.mp4 \
  --output /path/to/run_output \
  --skip_frame_extraction \
  --origin_filter \
  --auto_margin \
  --background_removal \
  --create_meta \
  --dataset_name my_dataset
```
