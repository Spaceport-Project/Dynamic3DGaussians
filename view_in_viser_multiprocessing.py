
# import multiprocessing
# from multiprocessing import Event, Lock
import copy
import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Lock

import threading
from typing import Dict
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
# import ffmpeg
import sys
from PIL import Image
import gc
import viser
from viser import transforms as tf
import imageio.v3 as iio
import signal


REMOVE_BACKGROUND = False  # False or True
# REMOVE_BACKGROUND = True  # False or True

w, h = 1920, 1080
near, far = 0.01, 100.0

def_pix = torch.tensor(
    np.stack(np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5, 1), -1).reshape(-1, 3)).cuda().float()
pix_ones = torch.ones(h * w, 1).cuda().float()
cnt=0

to8b = lambda x : (255*np.clip(x.permute(1,2,0).cpu().detach().numpy(),0,1)).astype(np.uint8)




def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )




  
        
class Viewer():

    def __init__(self, seq, exp, f_ratio=0.8, w=1920, h=1080, near=0.01, far=100.0):
        self.seq = seq
        self.exp = exp
        self.viser_server = viser.ViserServer()
        self.viser_server.scene.world_axes.visible = False
        self.viser_server.on_client_connect(self.handle_new_client)
        self.viser_server.on_client_disconnect(self.handle_disconnect_client)
        self.clients_num = 0
        self.k = np.array([[f_ratio * w, 0, w / 2], [0, f_ratio * w, h / 2], [0, 0, 1]])
        self.w = w
        self.h = h
        self.near = near
        self.far = far
        self.model_path=f"./output/{exp}/{seq}/params.npz"
        # self.scene_data, _ = self._load_scene_data2(self.seq, self.exp, seg_as_col=False)
        self.lock = Lock()
        self.render_viewers: Dict[int, RenderViewers] = {}
        # self.render_processes = dict()
        
        signal.signal(signal.SIGINT, self.signal_handler)



    def handle_disconnect_client(self, client:viser.ClientHandle):
        # self.cnt[client.client_id] = False
        print(f"{client.client_id} client disconnected!")
        self.render_viewers[client.client_id].running = False
        self.render_viewers.pop(client.client_id)



    
    def handle_new_client(self, client:viser.ClientHandle):
        print("new client!")
        # This will run whenever we get a new camera!
        # @client.camera.on_update
        # def _(_: viser.CameraHandle) -> None:
        #     self.cnt[self.clients_num]=0
        #     print(f"New camera on client {client.client_id}!")
            
        self.clients_num +=1 
        # Show the client ID in the GUI.
        gui_info = client.gui.add_text("Client ID", initial_value=str(client.client_id))
        gui_info.disabled = False
        print(client.client_id)
        # self.render_processes[client.client_id] = mp.Process(target=self.image_renderer, args=(client,))
        self.render_viewers[client.client_id] = RenderViewers(self, client)
        # self.render_viewers[client.client_id].start()

    # def load_model_and_start_viewer(self):
    #     self.scene_data, _ = self.load_scene_data2(self.seq, self.exp, seg_as_col=False)

    
    # def image_renderer(self, client, model_path):

    #     for t in range(num_timestamps):   
                
    #         if not self.first_enter: 
    #             self.client.camera.wxyz = self.init_camera()[0]
    #             self.client.camera.position = self.init_camera()[1] 
    #             # self.client.camera.look_at= np.array([0, 1, 3.5]) #for oguz
    #             self.client.camera.look_at= np.array([1, 1, 3.5]) # for yoga 
    #             self.first_enter = True

    #         # print(self.client.camera.look_at)
    #         # print(self.client.camera.wxyz, self.client.camera.position)
    #         # self.cnt[self.client.self.client_id] = True
            
    #         R = tf.SO3(np.asarray(self.client.camera.wxyz)).as_matrix()
    #         T = self.client.camera.position
    #         # c2w[:3,:3] = R
    #         # c2w[:3,3] = T

    #         c2w = np.vstack((np.concatenate((R, T[:,None]), axis=1),[0,0,0,1]))
    #         w2c = np.linalg.inv(c2w)
    #         # if id == 0 and flag : 
    #         #     flag=False
    #         #     start = time.time()
    #         start = time.time()


    #         # print(c2w, w2c)
    #         # print(R, T)
    #         with self.lock:
    #             img = self._render(w2c, self.scene_data[t], bg=[0, 0, 0])
    #         # img = render(self.w, self.h, self.k, self.near, self.far, self.scene_data[t], bg=[0, 0, 0])
            
    #         img_num = to8b(img)
            
    #         self.client.scene.set_background_image(
    #             img_num,
    #             format="jpeg",
    #             # jpeg_quality=80
    #         )
    #         # if id == 1 and not flag:
    #         #     flag = True
    #         #     end = time.time()
    #         #     print("Render fps:", 1/(end-start))
    #         end = time.time()
    #         print("Render fps:", 1/(end-start))




    def _load_scene_data(self, params, low_upper_limit, seg_as_col=False):
        
        
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
    
    def _load_scene_data2(self, seq, exp, seg_as_col=False):
    
      
        params = dict(np.load(f"./output/{exp}/{seq}/params.npz"))
    

        params = {k: torch.tensor(v).cuda().float() for k, v in params.items()}
        is_fg = params['seg_colors'][:, 0] > 0.5
        scene_data = []
        for t in range(len(params['means3D'])):
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

    def start_viewer(self):
        while True:
            print("Total numer of clients connected:", len(self.render_viewers))
            time.sleep(3)
    def signal_handler(self,sig, frame):

        print('You pressed Ctrl+C!')

        for key, val in self.render_viewers.items():
            val.running = False
            val.process.join()
        
        viewer.viser_server.stop()

        sys.exit(0)
    
    # def load_params_data(self, seq, exp):
    #     print(f"Loading params.npz file")
    #     params = dict(np.load(f"./output/{exp}/{seq}/params.npz"))
    #     params = {k: torch.tensor(v).float() for k, v in params.items()}
    #     print(f"Loaded params.npz file")

    #     return params
    
    # def _render(self, w2c, timestep_data, bg=[0,0,0]):
    #     with torch.no_grad():
    #         cam = setup_camera(self.w, self.h, self.k, w2c, self.near, self.far, bg=torch.tensor(bg)) #[0, 177./255, 64.0/255]
    #         im, _, depth, = Renderer(raster_settings=cam)(**timestep_data)
    #         # im[~is_fg] =  torch.tensor([0, 177./255, 64.0/255], dtype=torch.float32, device="cuda")
    #         # torchvision.utils.save_image(im, '{0:05d}'.format(cnt) + ".png")

    #         return im, depth
    # @classmethod
    # def init_camera(cls, y_angle=0., center_dist=5.5, cam_height= -1, f_ratio=0.82):
    #     ry = y_angle * np.pi / 180
    #     # w2c = np.array([[np.cos(ry), 0., -np.sin(ry), -0.0],
    #     #                 [0.,         1., 0.,          cam_height],
    #     #                 [np.sin(ry), 0., np.cos(ry),  center_dist],
    #     #                 [0.,         0., 0.,          1.]])
    #     w2c = np.array([[np.cos(ry), 0., -np.sin(ry), -2.0],
    #                     [0.,         1., 0.,          cam_height],
    #                     [np.sin(ry), 0., np.cos(ry),  center_dist],
    #                     [0.,         0., 0.,          1.]])
    #     c2w = np.linalg.inv(w2c)
    #     wxyz = tf.SO3.from_matrix(c2w[:3,:3]).wxyz
    #     return wxyz, c2w[:3,3]  
        
# def render(w, h, k, near, far, w2c, timestep_data, bg=[0,0,0]):
#         with torch.no_grad():
#             cam = setup_camera(w, h, k, w2c, near, far, bg=torch.tensor(bg)) #[0, 177./255, 64.0/255]
#             im, _, _, = Renderer(raster_settings=cam)(**timestep_data)
#             # im[~is_fg] =  torch.tensor([0, 177./255, 64.0/255], dtype=torch.float32, device="cuda")
#             # torchvision.utils.save_image(im, '{0:05d}'.format(cnt) + ".png")

#             return im

def test(arg):
        # num_timestamps = len(self.scene_data)
        print(arg)

class RenderViewers():
    def __init__(self, viewer, client=None ):
      
        self.viewer = viewer
        self.client = client
        self.running = True
        # self.scene_data = viewer.scene_data
        self.scene_data, _ = self._load_scene_data2(viewer.model_path, seg_as_col=False)
        self.w = viewer.w
        self.h = viewer.h
        self.far = viewer.far
        self.near = viewer.near
        self.k = viewer.k
        self.first_enter = False
        self.lock = viewer.lock
        # self.copy_client = client.__copy__()
        self.process = mp.Process(target=test, args=(self.client,))
        self.process.start()
    def test(self):
        # num_timestamps = len(self.scene_data)
        print("num_timestamps")

    def run(self):
        num_timestamps = len(self.scene_data)

        while self.running:
            # self._render_images(num_timestamps)
            

    # def _render_images(self,num_timestamps):
            for t in range(num_timestamps):   
                
                if not self.first_enter: 
                    self.client.camera.wxyz = self.init_camera()[0]
                    self.client.camera.position = self.init_camera()[1] 
                    # self.client.camera.look_at= np.array([0, 1, 3.5]) #for oguz
                    self.client.camera.look_at= np.array([1, 1, 3.5]) # for yoga 
                    self.first_enter = True

                # print(self.client.camera.look_at)
                # print(self.client.camera.wxyz, self.client.camera.position)
                # self.cnt[self.client.self.client_id] = True
                
                R = tf.SO3(np.asarray(self.client.camera.wxyz)).as_matrix()
                T = self.client.camera.position
                # c2w[:3,:3] = R
                # c2w[:3,3] = T

                c2w = np.vstack((np.concatenate((R, T[:,None]), axis=1),[0,0,0,1]))
                w2c = np.linalg.inv(c2w)
                # if id == 0 and flag : 
                #     flag=False
                #     start = time.time()
                start = time.time()


                # print(c2w, w2c)
                # print(R, T)
                with self.lock:
                    img = self._render(w2c, self.scene_data[t], bg=[0, 0, 0])
                # img = render(self.w, self.h, self.k, self.near, self.far, self.scene_data[t], bg=[0, 0, 0])
                
                img_num = to8b(img)
                
                self.client.scene.set_background_image(
                    img_num,
                    format="jpeg",
                    # jpeg_quality=80
                )
                # if id == 1 and not flag:
                #     flag = True
                #     end = time.time()
                #     print("Render fps:", 1/(end-start))
                end = time.time()
                print("Render fps:", 1/(end-start))



    def _render(self, w2c, timestep_data, bg=[0,0,0]):
        with torch.no_grad():
            cam = setup_camera(self.w, self.h, self.k, w2c, self.near, self.far, bg=torch.tensor(bg)) #[0, 177./255, 64.0/255]
            im, _, _, = Renderer(raster_settings=cam)(**timestep_data)
            # im[~is_fg] =  torch.tensor([0, 177./255, 64.0/255], dtype=torch.float32, device="cuda")
            # torchvision.utils.save_image(im, '{0:05d}'.format(cnt) + ".png")

            return im
    def _load_scene_data2(self, model_path, seg_as_col=False):
    
      
        params = dict(np.load(model_path))
    

        params = {k: torch.tensor(v).cuda().float() for k, v in params.items()}
        is_fg = params['seg_colors'][:, 0] > 0.5
        scene_data = []
        for t in range(len(params['means3D'])):
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
    @classmethod
    def init_camera(cls, y_angle=0., center_dist=5.5, cam_height= -1, f_ratio=0.82):
        ry = y_angle * np.pi / 180
        # w2c = np.array([[np.cos(ry), 0., -np.sin(ry), -0.0],
        #                 [0.,         1., 0.,          cam_height],
        #                 [np.sin(ry), 0., np.cos(ry),  center_dist],
        #                 [0.,         0., 0.,          1.]])
        w2c = np.array([[np.cos(ry), 0., -np.sin(ry), -2.0],
                        [0.,         1., 0.,          cam_height],
                        [np.sin(ry), 0., np.cos(ry),  center_dist],
                        [0.,         0., 0.,          1.]])
        c2w = np.linalg.inv(w2c)
        wxyz = tf.SO3.from_matrix(c2w[:3,:3]).wxyz
        return wxyz, c2w[:3,3]  

# def signal_handler(sig, frame):

#     print('You pressed Ctrl+C!')

#     for key, val in viewer.render_viewers.items():
        
#         val.join()
#     viewer.viser_server.stop()

#     sys.exit(0)

if __name__ == "__main__":

    limit_timestep = 100

    # exp_name = "exp_black_onlyoguz_scl_2_full"
    # for sequence in ["oguz_2"]:

    # exp_name = "exp_only_oguz_2_scl_4_it_500_green_test2"
    # exp_name = "exp_witback_oguz_2_scl_4_it_500_green_test"
    
    

    # exp_name = "exp_withbck_test"
    # exp_name = "exp_withbck_scl_2_reduced"
    
    # exp_name = "exp_withbck_scl_2_it_500"
    # sequence = "10-09-2024_data/pose_1_3"
    
    # exp_name = "yoga_wo_background_onlypt_scale_4_it_500_pose_1_4_all"
    # sequence = "10-09-2024_data/pose_1_4_all"


    exp_name = "exp_only_oguz_2_scl_4_it_500_20cams"
    sequence = "oguz_2"

    mp.set_start_method("fork", force=True)    
    viewer = Viewer(seq=sequence, exp=exp_name,w=1700, h=956)
    time.sleep(0.2)
    viewer.start_viewer()
   

