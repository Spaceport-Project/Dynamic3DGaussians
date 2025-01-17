
import threading
from typing import Dict
import cv2
import torch
import numpy as np
import open3d as o3d
import time
import os
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
import torchvision
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
from nogil_sample import encoder
import ctypes
from ctypes import string_at
from ctypes import *
libc = CDLL("libc.so.6") 


REMOVE_BACKGROUND = False  # False or True
# REMOVE_BACKGROUND = True  # False or True

w, h = 1920, 1080
near, far = 0.01, 100.0

def_pix = torch.tensor(
    np.stack(np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5, 1), -1).reshape(-1, 3)).cuda().float()
pix_ones = torch.ones(h * w, 1).cuda().float()
cnt=0

to8b = lambda x : (255*np.clip(x.permute(1,2,0).contiguous().cpu().detach().numpy(),0,1)).astype(np.uint8)




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
        self.viser_server = viser.ViserServer(port=8080)
        self.viser_server.scene.world_axes.visible = False
        self.clients_num = 0
        self.k = np.array([[f_ratio * w, 0, w / 2], [0, f_ratio * w, h / 2], [0, 0, 1]])
        self.w = w
        self.h = h
        self.near = near
        self.far = far
        self.scene_data, _ = self._load_scene_data2(self.seq, self.exp, seg_as_col=False)
        self.render_viewers: Dict[int, RenderViewers] = {}
        signal.signal(signal.SIGINT, self.signal_handler)
        self.viser_server.on_client_connect(self.handle_new_client)
        self.viser_server.on_client_disconnect(self.handle_disconnect_client)

        



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
        self.render_viewers[client.client_id] = RenderViewers(self, client)
        self.render_viewers[client.client_id].start()

    # def load_model_and_start_viewer(self):
    #     self.scene_data, _ = self.load_scene_data2(self.seq, self.exp, seg_as_col=False)

    
   

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
            time.sleep(1)
    def signal_handler(self,sig, frame):

        print('You pressed Ctrl+C!')

        for key, val in self.render_viewers.items():
            val.running = False
            # val.ready = False
            # val.condition.notify_all()
            val.thread_cuda.join()
            val.thread_encode.join()
          
            
        
        viewer.viser_server.stop()

        sys.exit(0)
    
   

class RenderViewers():
    # look_at = np.array([0, 1, 3.5]) # for oguz
    look_at = np.array([1, 1, 3.5]) # for yoga 
    roll_limit = (0.4, -1.0)
    pitch_limit = (1.4, -1.3)
    # distance = 6 #for oguz
    distance = 7 #for yoga

    def __init__(self, viewer, client ):
        self.viewer = viewer
        self.client = client
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.encode_event = threading.Event()
        self.cuda_event = threading.Event()
        self.thread_cuda = threading.Thread(target=self.render_images)
        self.img_num = np.array([])
        self.format = "jpeg"
        self.quality = 75
        self.thread_encode = threading.Thread(target=self.encode_image)
        self.running = True
        self.scene_data = viewer.scene_data
        self.w = viewer.w
        self.h = viewer.h
        self.far = viewer.far
        self.near = viewer.near
        self.k = viewer.k
        self.first_enter = False
        self.cnt=0
        self.data_ready = False
        self.interval = 1.0/(30)


    def start(self):
        self.thread_cuda.start()
        # time.sleep(0.1)
        self.thread_encode.start()
    def encode_image(self):
        while self.running:
            # with self.condition:
            #     while not self.ready:
            #         self.condition.wait()
                
                self.encode_event.wait()
                self.encode_event.clear()
                if len(self.img) > 0:
                    image = self.img #to8b(self.imag) #cv2.cvtColor(self.img_num, cv2.COLOR_RGB2BGR) 
                    size = image.numel() * image.element_size()
                    height = image.shape[0]
                    width = image.shape[1]
                  
                    # Encode the image using the C++ extension
                    try:
                        
                        
                        # media_type, size, encoded_image = encoder.encode_image_binary(image, self.format, self.quality)
                        media_type, size, encoded_image = encoder.encode_image_binary2(image.data_ptr(), size, height, width, self.format, self.quality)
                        # media_type, size, encoded_image = encoder.encode_image_binaryturbojpeg(image.data_ptr(), size, height, width, self.format, self.quality)

                        # self.g = (ctypes.c_char*size).from_address(encoded_image)
                        # self.g = np.ctypeslib.as_array(encoded_image, shape=(size,)).tobytes()
                        self.g = string_at(encoded_image, size) 
                        # libc.free.argtypes = [ctypes.c_void_p]
                        ptr = ctypes.cast(encoded_image, ctypes.POINTER(ctypes.c_uint8))
                        libc.free(ptr)
                        
                        self.cnt += 1


                    except Exception as e:
                        print(f"Error encoding {self.img_num}: {e}")
                    # self.ready = False
                    # self.condition.notify()
                    self.cuda_event.set()

                
            
    def render_images(self):
        num_timestamps =  len(self.scene_data)
        self.t0 = time.time() + self.interval*3
        c2w = np.eye(4)
        while self.running:
              
            for t in range(num_timestamps):   
                
                if not self.first_enter: 
                    self.client.camera.wxyz = self.init_camera()[0]
                    self.client.camera.position = self.init_camera()[1] 
                    # self.client.camera.look_at= np.array([0, 1, 3.5]) #for oguz
                    self.client.camera.look_at= self.look_at # for yoga 
                    self.first_enter = True

                R_S03 = tf.SO3(np.asarray(self.client.camera.wxyz))
                R = R_S03.as_matrix()
                T = self.client.camera.position
                start = time.time()

                if (R_S03.compute_roll_radians() <  self.roll_limit[0]  and R_S03.compute_roll_radians() >  self.roll_limit[1] ) \
                    and (R_S03.compute_pitch_radians() <  self.pitch_limit[0]  and R_S03.compute_pitch_radians() >  self.pitch_limit[1]) \
                    and np.linalg.norm(self.look_at - T)  > self.distance       :
                   
                
                
                    # c2w[:3,:3] = R
                    # c2w[:3,3] = T
                    c2w = np.vstack((np.concatenate((R, T[:,None]), axis=1),[0,0,0,1]))
                    w2c = np.linalg.inv(c2w)
                    # w2c_prev = w2c.copy()
                else:
                            
                    # w2c = w2c_prev
                    c2w = np.linalg.inv(w2c)

                    self.client.camera.position = c2w[:3,3] 
                    self.client.camera.wxyz = tf.SO3.from_matrix(c2w[:3,:3]).wxyz
                    self.client.camera.look_at = self.look_at 
                    # print("inside", tf.SO3.from_matrix(c2w[:3,:3]).as_rpy_radians())

                
                
                    # with self.condition:
                    #     while self.ready:
                    #         self.condition.wait()

                self.img = self._render(w2c, self.scene_data[t], bg=[0, 0, 0])

                # if t == num_timestamps-1:
                #     return
                
              


                    # self.ready = True
                    # self.condition.notify()
                
                
                self.encode_event.set()
                
                self.cuda_event.wait()
                if not self.running:
                    break
                delta = self.t0 - time.time()
                if delta > 0:
                    time.sleep(delta)
                self.t0 = time.time() + self.interval
                self.client.scene.set_background_image2(
                    self.g,
                    "image/jpeg",
                )

                self.cuda_event.clear()






                    
                # with self.lock:
                #     self.condition.notify()
                #     while not self.ready:
                #         self.condition.wait()
                #     self.ready = False
                    # self.condition.wait()
                

               
                end = time.time()
                if self.cnt % 20 ==0:
                    print(f"{self.client.client_id}. Client Render fps:", 1/(end-start))
                



    def _render(self, w2c, timestep_data, bg=[0,0,0]):
        with torch.no_grad():
            cam = setup_camera(self.w, self.h, self.k, w2c, self.near, self.far, bg=torch.tensor(bg)) #[0, 177./255, 64.0/255]
            im, _, _, = Renderer(raster_settings=cam)(**timestep_data)
            # im[~is_fg] =  torch.tensor([0, 177./255, 64.0/255], dtype=torch.float32, device="cuda")
            # torchvision.utils.save_image(im, '{0:05d}'.format(self.cnt) + ".png")
            # cnt +=1
            im = torch.flip(im, dims=[0])
            im =  im.permute(1,2,0).contiguous()
            im = (im.clamp(0,1)*255).to(torch.uint8)
            return im
    @classmethod
    def init_camera(cls, y_angle=0., center_dist=5.5, cam_height= -1, f_ratio=0.82):
        ry = y_angle * np.pi / 180
        w2c = np.array([[np.cos(ry), 0., -np.sin(ry), -0.0],
                        [0.,         1., 0.,          cam_height],
                        [np.sin(ry), 0., np.cos(ry),  center_dist],
                        [0.,         0., 0.,          1.]])
        # w2c = np.array([[np.cos(ry), 0., -np.sin(ry), -2.0],
        #                 [0.,         1., 0.,          cam_height],
        #                 [np.sin(ry), 0., np.cos(ry),  center_dist],
        #                 [0.,         0., 0.,          1.]])
        c2w = np.linalg.inv(w2c)
        wxyz = tf.SO3.from_matrix(c2w[:3,:3]).wxyz
        return wxyz, c2w[:3,3]  



if __name__ == "__main__":

    limit_timestep = 100

    # exp_name = "exp_black_onlyoguz_scl_2_full"
    # sequence = "oguz_2"

    # exp_name = "exp_only_oguz_2_scl_4_it_500_green_test2"
    # exp_name = "exp_witback_oguz_2_scl_4_it_500_green_test"
    
    

    # exp_name = "exp_withbck_test"
    # exp_name = "exp_withbck_scl_2_reduced"
    
    # exp_name = "exp_withbck_scl_2_it_500"
    # sequence = "10-09-2024_data/pose_1_3"
    


    exp_name = "yoga_wo_background_onlypt_scale_4_it_500_pose_1_4_all"
    sequence = "10-09-2024_data/pose_1_4_all"

    # exp_name = "exp_only_oguz_2_scl_4_it_500_20cams"
    # sequence = "oguz_2"


        
    viewer = Viewer(seq=sequence, exp=exp_name,w=1500, h=840)
    time.sleep(0.2)
    viewer.start_viewer()
   

