
# from collections import namedtuple
from collections import namedtuple
import os
import random
import threading
from typing import Dict
import cv2
import torch
import numpy as np
import time
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from diff_gaussian_rasterization_org import GaussianRasterizer as Renderer_org
import torchvision
from gaussian_renderer import GaussianModel
import torchvision.transforms as T

from helpers import setup_camera, quat_mult, searchForMaxIteration
from external import build_rotation
from colormap import colormap
from copy import deepcopy
from scipy.spatial.transform import Rotation as Rot
import sys
from PIL import Image
import viser
from viser import transforms as tf
import signal
from encoders import gst_h264_endoder_pipeline
import ctypes
from ctypes import string_at
from ctypes import *
from scipy.io.wavfile import write
from pydub import AudioSegment




libc = CDLL("libc.so.6") 
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')
gi.require_version('GstVideo', '1.0')
# gi.require_version('GstCuda', '1.0')
from gi.repository import Gst, GstApp, GLib, GObject
Gst.init(None)


REMOVE_BACKGROUND = False  # False or True
# REMOVE_BACKGROUND = True  # False or True

w, h = 1920, 1080
near, far = 0.01, 100.0

def_pix = torch.tensor(
    np.stack(np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5, 1), -1).reshape(-1, 3)).cuda().float()
pix_ones = torch.ones(h * w, 1).cuda().float()

to8b = lambda x : (255*np.clip(x.permute(1,2,0).contiguous().cpu().detach().numpy(),0,1)).astype(np.uint8)






  
        
class Viewer():

    def __init__(self, seq, exp, pc_path, port = 8080, f_ratio=0.8, w=1920, h=1080, near=0.01, far=100.0):
        self.seq = seq
        self.exp = exp
        self.pc_path = pc_path
        self.viser_server = viser.ViserServer(port=port)
        self.viser_server.scene.world_axes.visible = False
        self.clients_num = 0
        self.k = np.array([[f_ratio * w, 0, w / 2], [0, f_ratio * w, h / 2], [0, 0, 1]])
        self.w = w
        self.h = h
        self.near = near
        self.far = far
        self.scene_data, _ = self._load_scene_data3(self.seq, self.exp, seg_as_col=False)
        self.bg_scene_data = self.load_bg_scene_data(self.pc_path)
        self.render_viewers: Dict[int, RenderViewers] = {}
        signal.signal(signal.SIGINT, self.signal_handler)
        self.viser_server.on_client_connect(self.handle_new_client)
        self.viser_server.on_client_disconnect(self.handle_disconnect_client)
        self.running = True
   
        



    def handle_disconnect_client(self, client:viser.ClientHandle):
        print(f"{client.client_id} client disconnected!")
        self.render_viewers[client.client_id].running = False
        # self.render_viewers[client.client_id].thread_cuda.join()
        # self.render_viewers[client.client_id].thread_encode.join()
        # self.render_viewers[client.client_id].thread_process_video_buffers.join()
        self.render_viewers.pop(client.client_id)



    
    def handle_new_client(self, client:viser.ClientHandle):
        
            
        self.clients_num +=1 
        # Show the client ID in the GUI.
        # gui_info = client.gui.add_text("Client ID", initial_value= str(client.client_id))
        
        # gui_info.disabled = False
        # button = client.gui.add_button("Start/Pause Sound")
        # button.disabled = False
        print("new client!", client.client_id)
        print("Total number of clients connected to Hamit's demo:", len(self.render_viewers))
        self.render_viewers[client.client_id] = RenderViewers(self, client)
        
        self.render_viewers[client.client_id].start()

    def load_bg_scene_data(self, pc_path):
        gaussians = GaussianModel(3)
        gaussians.load_ply(pc_path)
        screenspace_points = torch.zeros_like(gaussians.get_xyz, dtype=gaussians.get_xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        scales = gaussians.get_scaling
        rotations = gaussians.get_rotation
        means3D = gaussians.get_xyz
        means2D = screenspace_points
        opacity = gaussians.get_opacity
        shs = gaussians.get_features
        colors_precomp = None
        cov3D_precomp = None

        return {
            "means3D" : means3D,
            "means2D" : means2D,
            "shs" : shs,
            "colors_precomp" : colors_precomp,
            "opacities" : opacity,
            "scales" : scales,
            "rotations" : rotations,
            "cov3D_precomp" : cov3D_precomp
        }, gaussians.active_sh_degree


   

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
        length = len(params['means3D'])
        for t in range(length): #len(params['means3D'])):
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
    
    def _load_scene_data3(self, seq, exp, seg_as_col=False):
        params_file =[ os.path.join(f"./output/{exp}/{seq}/", file) for file in  os.listdir(f"./output/{exp}/{seq}/") if file.startswith("params")]
        params_file = sorted(params_file, key= lambda x : int(os.path.basename(x).split("_")[1].split(".")[0]) if len(os.path.basename(x).split("_")) > 1 else os.path.basename(x).split("_")[0].split(".")[0]) 
        # print(params_file)
        scene_data = []
        total = 0
        for l, param_file  in enumerate(params_file):
            # if l != 1:
            #     continue
            
            params = dict(np.load(param_file))  
            print(f"{param_file} loaded!")

            params = {k: torch.tensor(v).cuda().float() for k, v in params.items()}
            is_fg = params['seg_colors'][:, 0] > 0.5
            if l == 69:
                length=80
            elif l ==75:
                length = 5
            else:
                length = len(params['means3D'])
            # length = len(params['means3D'])
            total = total + length
            print(f"total timesteps:", total, l)
            for t in range(length): #len(params['means3D'])):
                rendervar = {
                    'means3D': params['means3D'][t],
                    'colors_precomp': params['rgb_colors'][t] if not seg_as_col else params['seg_colors'],
                    'rotations': torch.nn.functional.normalize(params['unnorm_rotations'][t]),
                    'opacities': torch.sigmoid(params['logit_opacities']),
                    'scales': torch.exp(params['log_scales'] ),
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
            if not self.running:
                # time.sleep(1)
                break
            print("Total number of clients connected to the demo:", len(self.render_viewers))
            time.sleep(3600)
    def signal_handler(self,sig, frame):

        print('You pressed Ctrl+C!')

        self.running = False
        for key, val in self.render_viewers.items():
            # val.video_audio_event.set()

            val.running = self.running
            val.thread_cuda.join()
            val.thread_encode.join()
            val.thread_process_video_buffers.join()
            # val.thread_process_audio_buffers.join()

            
        
        viewer.viser_server.stop()

        sys.exit(0)
    
   

class RenderViewers():
  

    # look_at = np.array([-0.28, 1.65, 0.09]) 
    # look_at =  np.array([-0.12150903, -0.63690853, -0.01141978])
    look_at =  np.array([-3.41289393e-01,  1.79894763e+00,  3.27396714e-01])
    roll_limit = (np.pi, -np.pi) 
    distance_in = 0 #2 
    distance_out = 3 #4.5

  
    

    def __init__(self, viewer, client ):
        self.viewer = viewer
        self.client = client
        self.encode_event = threading.Event()
        self.cuda_event = threading.Event()
        # self.video_audio_event = threading.Event()
        # self.audio_event = threading.Event()
        self.thread_cuda = threading.Thread(target=self.render_images)    
        self.thread_encode = threading.Thread(target=self.encode_image)
        self.thread_process_audio_buffers = threading.Thread(target=self.process_audio_buffers)

        self.thread_process_video_buffers =  threading.Thread(target=self.process_video_buffers)
        self.running = True
        self.triggered = False
        self.scene_data = viewer.scene_data
        self.bg_scene_data = viewer.bg_scene_data
        # with client.gui.add_folder("Playback"):
        self.gui_play_button  = client.gui.add_button(" Play", icon=viser.Icon.PLAYER_PLAY)
        self.gui_pause_button  = client.gui.add_button(" Pause", icon=viser.Icon.PLAYER_PAUSE)
        self.gui_pause_button.disabled = True
        self.gui_play_button.disabled = False
        # self.gui_start_button = viewer.gui_start_button
        self.gui_play_button.on_click(self.handle_on_play_click)
        self.gui_pause_button.on_click(self.handle_on_pause_click)

        self.w = viewer.w
        self.h = viewer.h
        self.far = viewer.far   
        self.near = viewer.near
        self.k = viewer.k
        self.first_enter = False
        self.frame_number=0
        self.data_ready = False
        self.frame_rate = 30
        self.interval = 1.0/self.frame_rate
        self.isPaused = True
    
    def handle_on_play_click(self, _):
        
       
        if not self.gui_play_button.disabled:
            self.gui_play_button.disabled = True
            self.gui_pause_button.disabled = False
            self.isPaused = False


       
  
    def handle_on_pause_click(self, _):
        if not self.gui_pause_button.disabled:
            self.gui_play_button.disabled = False
            self.gui_pause_button.disabled = True
            self.isPaused = True

        

        
    def start(self):
       

        self.thread_encode.start()
        time.sleep(0.5)

        self.thread_cuda.start()
       
        self.thread_process_video_buffers.start()
        # time.sleep(1)
        # self.thread_process_audio_buffers.start()

    def process_audio_buffers(self, audio_file="/home/hamit/Downloads/yoga1.wav", segment_duration_ms=34):
    # def process_audio_buffers(self, audio_file="/home/hamit/Softwares/Dynamic3DGaussians/output/hamit_2024-12-04_17-14-42_scl_4_it_600_test1/2024-12-04_17-14-42/output.aac", segment_duration_ms=33):
        """
        Extracts audio segments corresponding to image timestamps.
        
        :param audio_file: Path to the audio file.
      
        :param segment_duration_ms: Duration of each audio segment in milliseconds.
        
        """

        # Load the audio file
        audio = AudioSegment.from_file(audio_file)
        audio = audio.set_channels(1)
        # audio.export("/home/hamit/Downloads/sample4.wav", format="wav")
        duration_ms = int(len(audio))
        print("duration in ms:", duration_ms)
        image_timestamps = [i for i in range(segment_duration_ms, duration_ms, segment_duration_ms)]  # Example timestamps in milliseconds
        
        frame_number = 0
       
       


        while self.running :

                t0 = time.time() + self.interval
                start_time = 0
                for i, timestamp in enumerate(image_timestamps):
                  

                    if not self.running:
                        break
                    # self.video_audio_event.wait()

                    start = time.time()
                    start_time = timestamp
                    end_time = start_time + segment_duration_ms
                    audio_segment = audio[start_time:end_time]
                    try:
                        self.client.scene.set_background_audio_pckt (
                                audio_segment.raw_data,
                                frame_number
                    )
                
            
                    except Exception as e:
                        print(f"Error sending audio packet: {e}")
                    
                    delta = t0 - time.time()
                    if delta > 0 :
                        time.sleep(delta)
                    t0 = time.time() + self.interval

                    end = time.time()

                    # print("Audio frame number", frame_number)
                    if frame_number % 50 ==0:      
                        print(f"{self.client.client_id} Audio Process Buffer fps:", 1.0/(end-start))  
                    frame_number += 1
                  

       
    
    
    def process_video_buffers(self):
        frame_number=0

        try:
            while self.running:
               
                start = time.time()
                size, data_ptr = gst_h264_endoder_pipeline.get_next_video_buffer_data(self.client_cnt)
                
                self.h264_pck = ctypes.string_at(data_ptr, size) 
                

                self.client.scene.set_background_h264_pckt(
                    self.h264_pck,
                    frame_number%300
                )
                
                end = time.time()

                if frame_number % 200 ==0:
                    print(f"{self.client.client_id} Sending encoded packet  fps:", 1/(end-start))  
                frame_number += 1
                 
        except KeyboardInterrupt:
            print("Stopped processing buffers.")
    
    def encode_image(self):
        self.client_cnt = gst_h264_endoder_pipeline.main_fun()
        print("Client num:",self.client_cnt)
        while self.running:
           
            self.encode_event.wait()
            self.encode_event.clear()
            
            if len(self.img) > 0:
                fore_size = self.img.numel() * self.img.element_size()
                # bg_size = self.img_bg.numel() * self.img_bg.element_size()

                
                try:
                    # gst_h264_endoder_pipeline.push_tensor_fore_and_back_ground(self.img.data_ptr(), fore_size, self.img_bg.data_ptr(), bg_size, 1080, 1920, self.frame_number, self.frame_rate, self.client_cnt)
                    gst_h264_endoder_pipeline.push_tensor_frame(self.img.data_ptr(), fore_size, self.frame_number, self.frame_rate, self.client_cnt)
                    
                    self.frame_number += 1


                except Exception as e:
                    print(f"Error encoding {self.img_num}: {e}")
              
            self.cuda_event.set()
        self.cuda_event.set()    

        gst_h264_endoder_pipeline.close_pipeline(self.client_cnt)
      
            
    def render_images(self):
        
        num_timestamps = len(self.scene_data)
        
        t0 = time.time() + self.interval
        c2w = np.eye(4)
        w2c = np.eye(4)

       
        
        while self.running:
            self.frame_number = 0 
            current_ts = 0
            previous_ts = 0
            # is_cam_pose_changed = True
            T_old = np.array([1, 1, 1])
            for t in range(num_timestamps): 
                # if t < 600:
                #     continue  
                if current_ts >= num_timestamps or not self.running:
                    break
                
                while True:
                    if not self.running:
                        break
                    if not self.first_enter: 
                        self.client.camera.wxyz = self.init_camera()[0]
                        self.client.camera.position = self.init_camera()[1] 
                        self.client.camera.look_at= self.look_at # for yoga 
                        self.first_enter = True

                    R_S03 = tf.SO3(np.asarray(self.client.camera.wxyz))
                    R = R_S03.as_matrix()
                    T = self.client.camera.position
                    # print(T, T_old)
                    if abs(T[0] - T_old[0]) > 0.01 or abs(T[1] - T_old[1]) > 0.01 or abs(T[2] - T_old[2]) > 0.01:
                        is_cam_pose_changed = True
                    else:
                        is_cam_pose_changed = False
                    # start = time.time()

                    if  np.linalg.norm(self.look_at - T)  > self.distance_in and  np.linalg.norm(self.look_at - T) < self.distance_out:

                        c2w = np.vstack((np.concatenate((R, T[:,None]), axis=1),[0,0,0,1]))
                        w2c = np.linalg.inv(c2w)
                        # print(repr(c2w))
                    else:
                                
                        c2w = np.linalg.inv(w2c)

                        self.client.camera.position = c2w[:3,3] 
                        self.client.camera.wxyz = tf.SO3.from_matrix(c2w[:3,:3]).wxyz
                        self.client.camera.look_at = self.look_at 
                    # print(f"current ts:{current_ts}")
                    # print(is_cam_pose_changed)
                    self.img = self._render(c2w, self.scene_data[current_ts], self.bg_scene_data, bg=[0, 177./255, 64.0/255], is_cam_pose_changed=is_cam_pose_changed)
                
                    
                    self.encode_event.set()
                    
                    self.cuda_event.wait()
                    
                    
                    delta = t0 - time.time()
                    if delta > 0:
                        time.sleep(delta)
                    t0 = time.time() + self.interval

                    T_old = T.copy()

                    self.cuda_event.clear()

                    if not self.isPaused:
                        previous_ts =  t
                        current_ts = t + 1
                        break
                    else:
                        current_ts = previous_ts



        self.encode_event.set()
   
                

    # def _render(self, w2c, timestep_data, bg=[0,0,0]):
    #     with torch.no_grad():
    #         cam = setup_camera(self.w, self.h, self.k, w2c, self.near, self.far, bg=torch.tensor(bg)) #[0, 177./255, 64.0/255]
    #         im, _, _, = Renderer(raster_settings=cam)(**timestep_data)
    #         # im[~is_fg] =  torch.tensor([0, 177./255, 64.0/255], dtype=torch.float32, device="cuda")
    #         # torchvision.utils.save_image(im, '{0:05d}'.format(cnt) + ".png")
    #         # im = torch.flip(im, dims=[0])
    #         im =  im.permute(1,2,0).contiguous()
    #         im = (im.clamp(0,1)*255).to(torch.uint8)
    #         return im
    
    @staticmethod
    def mask_person(image_tensor, green_rgb, threshold=0.1):
        green_tensor = torch.tensor(green_rgb, device=image_tensor.device).view(1, 3, 1, 1)
        diff = torch.abs(image_tensor - green_tensor)
        mask = torch.sum(diff, dim=1) > threshold
        return mask
    
    @staticmethod
    def crop_person(image_tensor, mask):
        non_zero_coords = torch.nonzero(mask.squeeze())

        # Check if any non-zero coordinates were found
        if non_zero_coords.numel() == 0:
            return
           # raise ValueError("No person detected in the image. Check the green background color and threshold.")

        # Safely extract coordinates
        y_coords = non_zero_coords[:, 0]
        x_coords = non_zero_coords[:, 1]

        # Use item() to get scalar values
        y_min = y_coords.min().item()
        y_max = y_coords.max().item()
        x_min = x_coords.min().item()
        x_max = x_coords.max().item()

        # Add small padding to avoid empty crops
        y_min = max(0, y_min)
        x_min = max(0, x_min)
        y_max = min(image_tensor.shape[2] - 1, y_max)
        x_max = min(image_tensor.shape[3] - 1, x_max)

        # Ensure we have a valid crop region
        if y_max <= y_min or x_max <= x_min:
            raise ValueError("Invalid crop region detected")

        cropped_image = image_tensor[:, :, y_min:y_max+1, x_min:x_max+1]
        return cropped_image, (y_min, y_max, x_min, x_max)
    @staticmethod
    def crop_person2(image_tensor, mask):
        # Get indices where mask is True (non-zero)
        non_zero_coords = torch.nonzero(mask.squeeze())  # Remove batch dimension

        # Extract min and max coordinates correctly
        y_coords = non_zero_coords[:, 0]  # First column contains y coordinates
        x_coords = non_zero_coords[:, 1]  # Second column contains x coordinates

        y_min = torch.min(y_coords)
        y_max = torch.max(y_coords)
        x_min = torch.min(x_coords)
        x_max = torch.max(x_coords)

        # Crop the image using the coordinates
        cropped_image = image_tensor[:, :, y_min:y_max+1, x_min:x_max+1]
        return cropped_image, (y_min, y_max, x_min, x_max)
    @staticmethod
    def resize_and_place2(image_tensor, cropped_tensor, bg, bbox, scale=2):
        y_min, y_max, x_min, x_max = bbox

        # Resize the cropped person
        resize_transform = T.Resize(
            (int(cropped_tensor.shape[2] * scale),
            int(cropped_tensor.shape[3] * scale)),
            antialias=True
        )
        resized_person = resize_transform(cropped_tensor)

        new_height, new_width = resized_person.shape[2], resized_person.shape[3]

        # Calculate x position (centered horizontally)
        center_x = (x_min + x_max) // 2
        x_start = max(0, center_x - new_width // 2)

        # Calculate y position (bottom-aligned with original crop)
        y_start = y_max - new_height  # This will place the bottom of resized person at y_max

        # Adjust if the resized person would go outside the image bounds
        if y_start < 0:
            y_start = 0
        if x_start + new_width > image_tensor.shape[3]:
            x_start = image_tensor.shape[3] - new_width
        x_start = max(0, x_start)

        y_end = min(y_start + new_height, image_tensor.shape[2])
        x_end = min(x_start + new_width, image_tensor.shape[3])

        # Create output image
        output_image = image_tensor.clone()

        # Calculate the actual heights and widths to copy
        height_to_copy = min(new_height, y_end - y_start)
        width_to_copy = min(new_width, x_end - x_start)
        # green_tensor = torch.tensor([bg], device=image_tensor.device).view(1, 3, 1, 1)
        # output_image[:, :, :] = green_tensor
        # Only copy the portion that fits within the image bounds
        output_image[:, :, y_start:y_start+height_to_copy, x_start:x_start+width_to_copy] = \
            resized_person[:, :, :height_to_copy, :width_to_copy]

        return output_image

    @staticmethod
    def resize_and_place(image_tensor, cropped_tensor, bg, bbox, scale=2):
        y_min, y_max, x_min, x_max = bbox

        # Resize the cropped person
        resize_transform = T.Resize(
            (int(cropped_tensor.shape[2] * scale),
            int(cropped_tensor.shape[3] * scale)),
            antialias=True
        )
        resized_person = resize_transform(cropped_tensor)

        new_height, new_width = resized_person.shape[2], resized_person.shape[3]

        # Calculate x position (centered horizontally)
        center_x = (x_min + x_max) // 2
        x_start = max(0, center_x - new_width // 2)

        # Calculate y position (bottom-aligned with original crop)
        y_start = y_max - new_height  # This will place the bottom of resized person at y_max

        # Adjust if the resized person would go outside the image bounds
        if y_start < 0:
            y_start = 0
        if x_start + new_width > image_tensor.shape[3]:
            x_start = image_tensor.shape[3] - new_width
        x_start = max(0, x_start)

        y_end = min(y_start + new_height, image_tensor.shape[2])
        x_end = min(x_start + new_width, image_tensor.shape[3])

        # Create output image
        output_image = image_tensor.clone()

        # Calculate the actual heights and widths to copy
        height_to_copy = min(new_height, y_end - y_start)
        width_to_copy = min(new_width, x_end - x_start)

        # Only copy the portion that fits within the image bounds
        green_tensor = torch.tensor([bg], device=image_tensor.device).view(1, 3, 1, 1)
        output_image[:, :, :] = green_tensor
        output_image[:, :, y_start:y_start+height_to_copy, x_start:x_start+width_to_copy] = \
            resized_person[:, :, :height_to_copy, :width_to_copy]

        return output_image
    
   
    
    @staticmethod
    def resize_and_place3(image_tensor, cropped_tensor, bg, bbox, scale=2):
        y_min, y_max, x_min, x_max = bbox

        # Calculate center of original cropped tensor
        original_center_y = (y_min + y_max) / 2
        original_center_x = (x_min + x_max) / 2

        # Resize the cropped person
        resize_transform = T.Resize(
            (int(cropped_tensor.shape[2] * scale),
            int(cropped_tensor.shape[3] * scale)),
            antialias=True
        )
        resized_person = resize_transform(cropped_tensor)

        new_height, new_width = resized_person.shape[2], resized_person.shape[3]

        # Calculate starting positions to maintain the same center
        y_start = int(original_center_y - new_height / 2)
        x_start = int(original_center_x - new_width / 2)

        # Adjust if the resized person would go outside the image bounds
        if y_start < 0:
            y_start = 0
        if x_start < 0:
            x_start = 0
        if y_start + new_height > image_tensor.shape[2]:
            y_start = image_tensor.shape[2] - new_height
        if x_start + new_width > image_tensor.shape[3]:
            x_start = image_tensor.shape[3] - new_width

        # Ensure we have valid positive coordinates
        y_start = max(0, y_start)
        x_start = max(0, x_start)

        y_end = min(y_start + new_height, image_tensor.shape[2])
        x_end = min(x_start + new_width, image_tensor.shape[3])

        # Create output image
        output_image = image_tensor.clone()

        # Calculate the actual heights and widths to copy
        height_to_copy = min(new_height, y_end - y_start)
        width_to_copy = min(new_width, x_end - x_start)

        green_tensor = torch.tensor([bg], device=image_tensor.device).view(1, 3, 1, 1)
        output_image[:, :, :] = green_tensor
        # Only copy the portion that fits within the image bounds
        output_image[:, :, y_start:y_start+height_to_copy, x_start:x_start+width_to_copy] = \
            resized_person[:, :, :height_to_copy, :width_to_copy]

        return output_image

    @staticmethod
    def process_image(image_tensor, green_rgb, scale=2, threshold=0.1):
    # Load the image and move it to GPU
        
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension and move to GPU

        # Mask the person
        mask = RenderViewers.mask_person(image_tensor, green_rgb, threshold)

        # Crop the person
        cropped_tensor, bbox = RenderViewers.crop_person(image_tensor, mask)

        # Resize and place the person back
        output_tensor = RenderViewers.resize_and_place2(image_tensor, cropped_tensor, green_rgb, bbox, scale)

        
        return output_tensor

    
    def _render(self, c2w, timestep_data, gaussians, bg=[0,0,0], is_cam_pose_changed=True):
        with torch.no_grad():
            # print(timestep_data["means3D"].shape)
            # timestep_data["scales"] = torch.mul(timestep_data["scales"], 1.2)
            # c2w[0,3] += -1
            # c2w_clone = c2w.copy()
            # c2w_clone[:3,3] *=0.8
            w2c = np.linalg.inv(c2w)
            cam = setup_camera(self.w, self.h, self.k, w2c, self.near, self.far, scale= 1/0.3, bg=torch.tensor(bg)) #[0, 177./255, 64.0/255]
            
            im, _, _, = Renderer(raster_settings=cam)(**timestep_data)
            # torchvision.utils.save_image(im, 'scaled_{0:05d}'.format(2) + ".png")
            # cam = setup_camera(self.w, self.h, self.k, w2c, self.near, self.far, scale= 1, bg=torch.tensor(bg)) #[0, 177./255, 64.0/255]
            # im, _, _, = Renderer(raster_settings=cam)(**timestep_data)
            # torchvision.utils.save_image(im, '{0:05d}'.format(2) + ".png")


            # im = RenderViewers.process_image(im, bg, scale=0.75)
            # torchvision.utils.save_image(im, '{0:05d}'.format(2) + ".png")
            # print("before",im.shape)
            # im =  im.permute(1,2,0).contiguous()
            # im = (im.clamp(0,1)*255).to(torch.uint8)
            # print(im.shape)

            # green_color = torch.tensor(bg, device=im.device)
            # color_distance = torch.norm(im - green_color.view(3, 1, 1), dim=0)
            # threshold = 0.9  # You might need to adjust this value  
            # mask = (color_distance > threshold).float() 
            # mask = mask.expand_as(im)  
            # img_bg_prev = torch.tensor([0], device=im.device)
            if is_cam_pose_changed:
                # c2w[0,3] += 0.3
                c2w[1,3] += -0.45  # -0.53 #0.27 #-0.4
                c2w[0,3] += -0.34*0.3 #-0.34 # -3
                c2w[2,3] += -3.27396714e-01
                w2c_bg = np.linalg.inv(c2w)
                cam_bg = setup_camera(self.w, self.h, self.k, w2c_bg, self.near, self.far, bg=torch.tensor(bg))
                cam_dict = cam_bg._asdict()
                cam_dict["sh_degree"] = gaussians[1]
                cam_dict["debug"] = False
                cam_dict["antialiasing"] = False
                cam_dict["bg"] = torch.tensor([0,0,0]).cuda().float()
                # cam_dict["scale_modifier"] = 0.5
                extented_cam = namedtuple('extented_cam', cam_dict.keys())  
                cam_bg = extented_cam(**cam_dict) 
            
                im_bg,_,_ = Renderer_org(raster_settings=cam_bg)(**gaussians[0])
                self.img_bg_prev = im_bg
            else:
                im_bg = self.img_bg_prev
            # im_bg =  im_bg.permute(1,2,0).contiguous()
            # im_bg = (im_bg.clamp(0,1)*255).to(torch.uint8)

            # im[~is_fg] =  torch.tensor([0, 177./255, 64.0/255], dtype=torch.float32, device="cuda")
            # torchvision.utils.save_image(im_bg, '{0:05d}'.format(1) + ".png")
            # im = torch.flip(im, dims=[0])


            if len(im.shape) == 3:
                im = im.unsqueeze(0)  
                im_bg = im_bg.unsqueeze(0)  

            
            green_color_threshold=0.5
            target_green = torch.tensor([0, 177/255, 64/255]).to(im.device)

            # Calculate difference from target green for each pixel
            diff = torch.abs(im - target_green.view(1, 3, 1, 1))

            # Create mask where pixels are close to target green
            mask = (diff.sum(dim=1, keepdim=True) > green_color_threshold).float()

            # Combine images using the mask
            result = im * mask + im_bg * (1 - mask)
            result = result.squeeze(0)

            result =  result.permute(1,2,0).contiguous()
            result = (result.clamp(0,1)*255).to(torch.uint8)

            return result
        
    @classmethod
    def init_camera(cls, y_angle=0., center_dist=4., cam_height= 3., f_ratio=0.82):
        ry = y_angle * np.pi / 180
        # w2c = np.array([[np.cos(ry), 0., -np.sin(ry), -0.0],
        #                 [0.,         1., 0.,          cam_height],
        #                 [np.sin(ry), 0., np.cos(ry),  center_dist],
        #                 [0.,         0., 0.,          1.]])
        
        # c2w = np.array([[-0.94743326, -0.0982282 , -0.30450196,  2.2777 ],
        #         [-0.09894314,  0.99500657, -0.01312203 , -1.185],
        #         [ 0.3042704 ,  0.01769613, -0.95242132,  6.5],
        #         [ 0.  ,        0.    ,      0.  ,        1.        ]])
    #     c2w  = np.array([[-0.16760753, -0.20236734,  0.96486018, -7.71863478],
    #    [ 0.07101295,  0.97368453,  0.21655392, -0.11243791],
    #    [-0.98329287,  0.10481364, -0.14882615,  1.26942662],
    #    [ 0.        ,  0.        ,  0.        ,  1.        ]])
        
    #     c2w = np.array([[-0.44812752,  0.22500475, -0.86519049,  1.81380838],
    #    [-0.09567986,  0.95017668,  0.2966642 ,  1.05998857],        # for 0.70 scale
    #    [ 0.88883468,  0.21572469, -0.40427189,  1.00827238],
    #    [ 0.        ,  0.        ,  0.        ,  1.        ]])
        
        c2w = np.array([[-0.71431047,  0.03423488, -0.69899107,  0.83273878],
       [-0.1549435 ,  0.96627859,  0.20566528,  1.26594328],   # for 0.3 scale
       [ 0.68246103,  0.25521298, -0.68491844,  1.4389307 ],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
        
     
      
        wxyz = tf.SO3.from_matrix(c2w[:3,:3]).wxyz
        return wxyz, c2w[:3,3]  



if __name__ == "__main__":


    # exp_name = "exp_black_onlyoguz_scl_2_full"
    # for sequence in ["oguz_2"]:

    # exp_name = "exp_only_oguz_2_scl_4_it_500_green_test2"
    # exp_name = "exp_witback_oguz_2_scl_4_it_500_green_test"
    
    

    # exp_name = "exp_withbck_test"
    # exp_name = "exp_withbck_scl_2_reduced"
    
    # exp_name = "exp_withbck_scl_2_it_500"
    # sequence = "10-09-2024_data/pose_1_3"
    

    # exp_name = "exp_only_oguz_2_scl_4_it_500_20cams"
    # sequence = "oguz_2"

    exp_name = "hamit_2024-12-04_17-14-42_scl_2_it_600_test1"
    sequence = "2024-12-04_17-14-42"

    # exp_name = "hamit_2024-12-19_19-12-14_4096_wo_bckgrnd_scl_2_it_700"
    # sequence = "2024-12-19_19-12-14_4096_wo_bckgrnd"
    
    # exp_name = "2025-02-05_14-06-52_gain_9_scl_1_it_700_2"
    # sequence = "2025-02-05_14-06-52_gain_9"
    
    # exp_name = "2025-02-05_14-16-46_gain_9_scl_1_it_700_2"
    # exp_name = "2025-02-05_14-24-53_gain_8_scl_1_test_whole_scene_start_0"
    # sequence = "2025-02-05_14-24-53_gain_8_whole_scene"
    
    # exp_name = "2025-02-05_14-24-53_gain_8_scl_1_test_start_0"
    # sequence = "2025-02-05_14-24-53_gain_8"

    # exp_name = "2025-02-05_14-29-36_aligned_scl_1_test_start_0"
    # sequence = "2025-02-05_14-29-36_aligned"

    exp_name = "2025-02-05_14-16-46_gain_9_scl_1_it_700_test_start_0-1000"
    # exp_name = "2025-02-05_14-16-46_gain_9_scl_1_it_700_test_expanded_start_0"
    sequence = "2025-02-05_14-16-46_gain_9"

    # exp_name = "2025-02-05_14-29-36_aligned_scl_1_test_green_start_0"
    # sequence = "2025-02-05_14-29-36_aligned"

    # exp_name = "2025-02-05_14-29-36_scl_1_test_start_1-56"
    # sequence = "2025-02-05_14-29-36"

    # exp_name = "bicycle_2_scl_2_test7"
    # sequence = "bicycle_2"
    point_cloud = "/home/hamit/gaussian-splatting/output/salon/point_cloud/iteration_30000/point_cloud.ply" 
        
    viewer = Viewer(seq=sequence, exp=exp_name, port =8089,  pc_path= point_cloud, w=1920, h=1080)
    time.sleep(0.2)
    viewer.start_viewer()
   

