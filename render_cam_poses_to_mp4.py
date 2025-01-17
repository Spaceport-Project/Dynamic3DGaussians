
import torch
import numpy as np
import open3d as o3d
import time
import os
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from helpers import setup_camera, quat_mult, searchForMaxIteration
from external import build_rotation
from colormap import colormap
from copy import deepcopy
from scipy.spatial.transform import Rotation as Rot
import ffmpeg
import sys
from PIL import Image
import gc



REMOVE_BACKGROUND = False  # False or True
# REMOVE_BACKGROUND = True  # False or True

w, h = 1920, 1080
near, far = 0.01, 100.0

def_pix = torch.tensor(
    np.stack(np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5, 1), -1).reshape(-1, 3)).cuda().float()
pix_ones = torch.ones(h * w, 1).cuda().float()
cnt=0

to8b = lambda x : (255*np.clip(x.permute(1,2,0).cpu().numpy(),0,1)).astype(np.uint8)

def read_camera_poses_from_file(file_path):
    # Initialize an empty list to store the camera poses
    camera_poses = []
    # Open the file and read all lines
    with open(file_path, 'r') as file:
        lines = file.readlines()
        # Process every set of 4 lines at a time since each camera pose is represented by a 4x4 matrix
        for i in range(0, len(lines), 4):  # Adjusted to 5 assuming there's a blank line between matrices
            # Extract four lines and create a 4x4 matrix
            pose_lines = lines[i:i+4]
            pose_matrix = np.array([[float(value) for value in line.strip().split()] for line in pose_lines])
            # Append the constructed matrix to the camera_poses list
            camera_poses.append(pose_matrix)

    return camera_poses



def load_scene_data2(params, low_upper_limit, seg_as_col=False):
    
    
    is_fg = params['seg_colors'][:, 0] > 0.5
    scene_data = []
    for t in range(*low_upper_limit):
        
        rendervar = {
            'means3D': params['means3D'][t].cuda(),
            'colors_precomp': params['rgb_colors'][t].cuda() if not seg_as_col else params['seg_colors'].cuda(),
            'rotations': torch.nn.functional.normalize(params['unnorm_rotations'][t].cuda()),
            'opacities': torch.sigmoid(params['logit_opacities']).cuda(),
            'scales': torch.exp(params['log_scales']).cuda(),
            'means2D': torch.zeros_like(params['means3D'][0], device="cuda")
        }
        # rendervar = {k: v.cuda() for k, v in rendervar.items()}
        if REMOVE_BACKGROUND:
            rendervar = {k: v[is_fg] for k, v in rendervar.items()}
        scene_data.append(rendervar)
    if REMOVE_BACKGROUND:
        is_fg = is_fg[is_fg]
    return scene_data, is_fg

def load_params_data(seq, exp):
    max_iter = searchForMaxIteration(f"./output/{exp}/{seq}")
    if max_iter is None:
        params = dict(np.load(f"./output/{exp}/{seq}/params.npz"))
    else:
        params = dict(np.load(f"./output/{exp}/{seq}/params_{max_iter}.npz"))
        print(f"Loading params_{max_iter}.npz file")
    params = {k: torch.tensor(v).float() for k, v in params.items()}
    return params

   

def load_scene_data(seq, exp, seg_as_col=False):
    
    max_iter = searchForMaxIteration(f"./output/{exp}/{seq}")
    if max_iter is None:
        params = dict(np.load(f"./output/{exp}/{seq}/params.npz"))
    else:
        params = dict(np.load(f"./output/{exp}/{seq}/params_{max_iter}.npz"))
        print(f"Loading params_{max_iter}.npz file")

    # params = {k: torch.tensor(v).float() for k, v in params.items()}

    params = {k: torch.tensor(v).cuda().float() for k, v in params.items()}
    is_fg = params['seg_colors'][:, 0] > 0.5
    scene_data = []
    for t in range(len(params['means3D'])):
        if t > 200:
            break
        rendervar = {
            'means3D': params['means3D'][t],
            'colors_precomp': params['rgb_colors'][t] if not seg_as_col else params['seg_colors'],
            'rotations': torch.nn.functional.normalize(params['unnorm_rotations'][t]),
            'opacities': torch.sigmoid(params['logit_opacities']),
            'scales': torch.exp(params['log_scales']),
            'means2D': torch.zeros_like(params['means3D'][0], device="cuda")
        }
        if REMOVE_BACKGROUND:
            rendervar = {k: v[is_fg] for k, v in rendervar.items()}
        scene_data.append(rendervar)
    if REMOVE_BACKGROUND:
        is_fg = is_fg[is_fg]
    
    return scene_data, is_fg






def render(w2c, k, timestep_data, bg=[0,0,0]):
    global cnt
    with torch.no_grad():
        cam = setup_camera(w, h, k, w2c, near, far, bg=torch.tensor(bg)) #[0, 177./255, 64.0/255]
        im, _, depth, = Renderer(raster_settings=cam)(**timestep_data)
        # im[~is_fg] =  torch.tensor([0, 177./255, 64.0/255], dtype=torch.float32, device="cuda")
        # torchvision.utils.save_image(im, '{0:05d}'.format(cnt) + ".png")
        cnt+=1

        return im, depth


def generate_mp4(out_filename, res=f'{w}x{h}'):
    bitrate = '2M' 
    gop_size = 5
    os.makedirs(os.path.dirname(out_filename), exist_ok=True)    
    try:
       process = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=res)
        .output(out_filename, pix_fmt='yuv420p', vcodec='libx264', r=30,video_bitrate=bitrate, g=gop_size )
        .overwrite_output()
        .global_args('-hide_banner')
        .run_async(pipe_stdin=True,pipe_stdout=True, pipe_stderr=True)
    )
    except ffmpeg.Error as e:
        print(e.stderr.decode(), file=sys.stderr)
        sys.exit(1)
    return process

def render_poses(seq, exp, cam_poses_file, f_ratio=0.82):
    cam_poses = read_camera_poses_from_file(cam_poses_file)
    k = np.array([[f_ratio * w, 0, w / 2], [0, f_ratio * w, h / 2], [0, 0, 1]])
   
    scene_data, _ = load_scene_data(seq, exp, seg_as_col=False)

    num_timesteps = len(scene_data)
      
    for id, w2c in enumerate(cam_poses):
        imgs = []   
        for t in range(num_timesteps):
            start = time.time()
            img, _ = render(w2c, k, scene_data[t], bg=[0, 0, 0])
            end = time.time()
            print("Render fps:", 1/(end-start))
            img_num = to8b(img)
            
            imgs.append(img_num)
            # if t == 0:
            #     image = Image.fromarray(img_num)
            #     image.save("img_trial.png")
            #     exit()
           

           
        process = generate_mp4(f'grid_videos_pose_1_4/output_green_{id}.mp4')
        for im in imgs:
            process.stdin.write(
                im
                .astype(np.uint8)
                .tobytes()
            )
        # Close the pipe
        process.stdin.close()
        process.wait()

def render_poses2(seq, exp, output_folder, cam_poses_file, f_ratio=0.82):
    cam_poses = read_camera_poses_from_file(cam_poses_file)
    k = np.array([[f_ratio * w, 0, w / 2], [0, f_ratio * w, h / 2], [0, 0, 1]])
    params = load_params_data(seq, exp)
    num_timesteps = len(params['means3D'])
    os.makedirs(output_folder,exist_ok=True)
    for id, w2c in enumerate(cam_poses):
        output_mp4 = os.path.join(output_folder,f"output_black_{id}.mp4")
        process = generate_mp4(output_mp4) 
      
        for step in range(num_timesteps//limit_timestep + 1):
            if  step == num_timesteps//limit_timestep:
                low_up_limit = (limit_timestep*step, num_timesteps)
            else:
                low_up_limit = (limit_timestep*step, limit_timestep*(step+1))

            print(f"Processing {low_up_limit} frame interval")
            scene_data, _ = load_scene_data2(params, low_up_limit, seg_as_col=False)
                        

            num_timestamps = len(scene_data)

            imgs = []
            for t in range(num_timestamps):
                start = time.time()
                img, _ = render(w2c, k, scene_data[t], bg=[0, 0, 0])
                end = time.time()
                # print("Render fps:", 1/(end-start))
                img_num = to8b(img)
                imgs.append(img_num)
                if t == 0:
                    image = Image.fromarray(img_num)
                    image.save("img_trial.png")
                    # exit()
            
            for im in imgs:
                process.stdin.write(
                    im
                    .astype(np.uint8)
                    .tobytes()
                )

            # Check memory usage
            # print(f"Memory allocated: {torch.cuda.memory_allocated()} bytes")
            # print(f"Memory reserved: {torch.cuda.memory_reserved()} bytes")
            gc.collect()      
            for sc in scene_data:
                for k1, v in sc.items():
                    del v
            torch.cuda.empty_cache()
            
            # Check memory usage again
            # print(f"Memory allocated after deletion: {torch.cuda.memory_allocated()} bytes")
            # print(f"Memory reserved after deletion: {torch.cuda.memory_reserved()} bytes")
         # Redirect stdout and stderr to DEVNULL
        # process.stdout = subprocess.DEVNULL
        # process.stderr = subprocess.DEVNULL   
        print(f"{output_mp4} has been generated!") 
        process.stdin.close()
        process.wait()   
  



if __name__ == "__main__":
    limit_timestep = 25

    # exp_name = "exp_black_onlyoguz_scl_2_full"
    # exp_name = "exp_only_oguz_2_scl_4_it_500_20cams"
    # for sequence in ["oguz_2"]:

    # exp_name = "exp_only_oguz_2_scl_4_it_500_green_test2"
    # exp_name = "exp_witback_oguz_2_scl_4_it_500_green_test"
    # exp_name = "yoga_wo_background_onlypt_scale_4_it_500_pose_1_4_all"
    # exp_name = "yoga_wo_background_onlypt_scale_4_it_500"

    # exp_name = "exp_withbck_test"
    # exp_name = "exp_withbck_scl_2_reduced"
    exp_name = "exp_withbck_scl_4_it_500"
    for sequence in ["10-09-2024_data/pose_1_3"]:
    
        output_folder="grid_video_pose_1_3_new_test"
        render_poses2(sequence, exp_name, output_folder, "/home/hamit/Softwares/Dynamic3DGaussians/cam_poses_new5.txt")
