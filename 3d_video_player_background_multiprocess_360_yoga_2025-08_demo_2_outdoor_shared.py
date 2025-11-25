
from collections import namedtuple
from multiprocessing import shared_memory
import os
import random
import threading
from typing import Dict
import torch
import numpy as np
import time
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
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
# from queue import Queue, Empty  
import torch.multiprocessing as torch_mp  
from io import BytesIO  
import queue
import gc 




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

# def_pix = torch.tensor(
#     np.stack(np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5, 1), -1).reshape(-1, 3)).cuda().float()
# pix_ones = torch.ones(h * w, 1).cuda().float()

# to8b = lambda x : (255*np.clip(x.permute(1,2,0).contiguous().cpu().detach().numpy(),0,1)).astype(np.uint8)



class AudioPacketizer:
    def __init__(self, mp3_path: str, packet_duration_ms: int = 50, force_mono: bool = False, format: str = 'opus'):
        """
        Initialize the audio packetizer
        Args:
            mp3_path: Path to MP3 file
            packet_duration_ms: Duration of each packet in milliseconds (default: 50ms)
            force_mono: If True, converts stereo to mono. If False, maintains original channels (default: False)
            format: Audio format to use ('opus' or 'aac')
        """
        self.audio = AudioSegment.from_mp3(mp3_path)
        self.packet_duration_ms = packet_duration_ms
        self.position = 0
        self.overlap_ms = 15
        self.format = format.lower()
        self.exit = False

        # Convert to mono if requested
        if force_mono and self.audio.channels > 1:
            self.audio = self.audio.set_channels(1)

        self.channels = self.audio.channels

        # Set sample rate based on format
        if self.format == 'opus':
            self.audio = self.audio.set_frame_rate(48000)  # Opus works best with 48kHz
        elif self.format == 'aac': # AAC
            self.audio = self.audio.set_frame_rate(44100)  # AAC standard
        else:
            raise(f"{self.format} audio format not supported")

        # Set consistent format
        self.audio = self.audio.set_sample_width(2)  # 16-bit

    def get_next_packet(self) -> tuple[bytes, int] | None:
        """
        Get the next audio packet
        Returns:
            Tuple of (packet_data, timestamp) or None if end of audio
        """
        if self.position >= len(self.audio) or self.exit:
            return None

        # Extract packet
        end_pos = min(self.position + self.packet_duration_ms, len(self.audio))
        packet = self.audio[self.position:end_pos]
        packet = packet.fade_in(self.overlap_ms).fade_out(self.overlap_ms)

        # Export packet
        buffer = BytesIO()

        if self.format == 'opus':
            # Opus-specific export parameters
            export_params = [
                "-ar", "48000",          # Sample rate (48kHz optimal for Opus)
                "-ac", str(self.channels),  # Number of channels
                "-acodec", "libopus",    # Use Opus codec
                "-b:a", "64k",           # Bitrate
                "-vbr", "on",            # Variable bitrate
                "-compression_level", "10",  # Compression level
                "-frame_duration", "20",  # Frame size in ms
                "-application", "audio"   # Application type
            ]

            # Add channel-specific settings for Opus
            if self.channels == 1:
                export_params.extend(["-cutoff", "12000"])
            else:
                export_params.extend(["-cutoff", "20000"])

            packet.export(buffer, format='opus', parameters=export_params)
        else:
            # AAC-specific export parameters
            export_params = [
                "-ar", "44100",      # Sample rate
                "-ac", str(self.channels),  # Number of channels
                "-acodec", "aac",    # Use AAC codec
                "-b:a", "192k",      # Higher bitrate for AAC
                "-q:a", "2",         # Quality setting
                "-cutoff", "18000"   # Frequency cutoff
            ]

            packet.export(buffer, format='adts', parameters=export_params)

        packet_data = buffer.getvalue()

        # Calculate timestamp
        timestamp = int(time.time() * 1000)  # Current timestamp in milliseconds

        # Update position with overlap
        self.position = end_pos - self.overlap_ms

        return packet_data, timestamp
    def reset(self):
        """Reset the position to the beginning of the audio"""
        self.position = 0


class ZeroCopyTensorManager:
    def __init__(self, tensor_dict):
        self.tensor_dict = tensor_dict
        self.shared_memories = {}
        self.tensor_info = {}
        self._create_shared_memory_tensors()
    
    def _create_shared_memory_tensors(self):
        print("Creating shared memory tensors...")
        
        for name, tensor in self.tensor_dict.items():
            if tensor is None:
               
                self.shared_memories[name] = None
            else:
                if tensor.is_cuda:
                    np_array = tensor.cpu().numpy()
                else:
                    np_array = tensor.detach().numpy()
            
                shm = shared_memory.SharedMemory(create=True, size=np_array.nbytes)
                shared_array = np.ndarray(np_array.shape, dtype=np_array.dtype, buffer=shm.buf)
                shared_array[:] = np_array[:]
            
            
                self.shared_memories[name] = shm
                self.tensor_info[name] = {
                    'shm_name': shm.name,
                    'shape': np_array.shape,
                    'dtype': str(np_array.dtype),
                    'size': np_array.nbytes
                }
            
                print(f"Created shared memory tensor '{name}' {np_array.shape}")
        total_bytes = sum(tensor.element_size() * tensor.numel()  for tensor in self.tensor_dict.values() if tensor is not None )
        total_mb = total_bytes / (1024 * 1024)
        print(f"Dict size: {total_mb:.2f} MB ({total_bytes:,} bytes)")
    
    def get_tensor_info(self):
        return self.tensor_info.copy()
    
    def cleanup(self):
        print("Cleaning up shared memory...")
        for name, shm in self.shared_memories.items():
            try:
                shm.close()
                shm.unlink()
                print(f"Cleaned up shared memory for '{name}'")
            except Exception as e:
                print(f"Error cleaning up '{name}': {e}")


def load_bg_scene_data( pc_path, max_sh_degree=3):
        gaussians = GaussianModel(max_sh_degree)
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
            "means3D" : means3D.detach_(),
            "means2D" : means2D.detach_(),
            "shs" : shs.detach_(),
            "colors_precomp" : colors_precomp,
            "opacities" : opacity.detach_(),
            "scales" : scales.detach_(),
            "rotations" : rotations.detach_(),
            "cov3D_precomp" : cov3D_precomp
        }, gaussians.active_sh_degree
  
        
class Viewer():

    def __init__(self, seq, exp, pc_path, port=8082, title="", f_ratio=0.8, w=1920, h=1080, near=0.01, far=100.0, max_sh_degree=3):
        self.seq = seq
        self.exp = exp
        self.viser_server = viser.ViserServer(port=port, title=title )
        self.viser_server.scene.world_axes.visible = False
        self.clients_num = 0
        self.k = np.array([[f_ratio * w, 0, w / 2], [0, f_ratio * w, h / 2], [0, 0, 1]])
        self.w = w
        self.h = h
        self.near = near
        self.far = far
        self.params_files, self.num_timesteps, self.look_at, self.params_files_dict = self.load_list_params_files(self.seq, self.exp)
        
        bg_scene_data, self.max_sh_degree = self.load_bg_scene_data(pc_path, max_sh_degree=max_sh_degree)
        self.bg_scene_data_manager = ZeroCopyTensorManager(bg_scene_data)
        self.bg_scene_data_info = self.bg_scene_data_manager.get_tensor_info()
        self.max_sh_degree = max_sh_degree
        # bg_scene_data, self.max_sh_degree = self.load_bg_scene_data(pc_path, max_sh_degree=max_sh_degree)
        
        # def share_cuda_tensor(tensor):
        #     """Share CUDA tensor between processes"""
        #     if torch.is_tensor(tensor) and tensor.is_cuda:
        #         return tensor.share_memory_()
        #     return tensor

        # def share_scene_data(scene_data):
        #     """Share all CUDA tensors in scene data"""
        #     shared_data = {}
        #     for k, v in scene_data.items():
        #         shared_data[k] = share_cuda_tensor(v)
        #     return shared_data
        
        # self.bg_scene_data_shared = share_scene_data(bg_scene_data)
        
       
        self.render_viewers: Dict[int, RenderViewers] = {}
        signal.signal(signal.SIGINT, self.signal_handler)
        self.viser_server.on_client_connect(self.handle_new_client)
        self.viser_server.on_client_disconnect(self.handle_disconnect_client)
        self.running = True
   
        




    def handle_disconnect_client(self, client:viser.ClientHandle):
        
        
        print(f"{client.client_id} client disconnected!")

        # Get the viewer before removing it
        viewer = self.render_viewers[client.client_id]

        # Cleanup the specific viewer
        viewer.clean_up()
       

        # Remove from dictionary
        del self.render_viewers[client.client_id]

        # Force garbage collection
        gc.collect()

        print("Total number of clients connected to the demo:", len(self.render_viewers))



    
    def handle_new_client(self, client:viser.ClientHandle):
        
            
        self.clients_num +=1 
       
        print("new client!", client.client_id)
        print("Total number of clients connected to the demo:", len(self.render_viewers))
        self.render_viewers[client.client_id] = RenderViewers(self, client)
        
        self.render_viewers[client.client_id].start()

   
   
    
    def load_list_params_files(self, seq, exp):
        params_files =[ os.path.join(f"./output/{exp}/{seq}/", file) for file in  os.listdir(f"./output/{exp}/{seq}/") if file.startswith("params")]
        params_files = sorted(params_files, key= lambda x : int(os.path.basename(x).split("_")[1].split(".")[0]) if len(os.path.basename(x).split("_")) > 1 else os.path.basename(x).split("_")[0].split(".")[0]) 
        pc = np.load(os.path.join(f"./output/{exp}/{seq}/", "init_pt_cld.npz"))
        xyz = [vert[:3] for vert in pc['data']]
        xyz = np.asarray(xyz)
        center = np.mean(xyz[:], axis=0)
        print("Foreground center:",center)
        total_length = 0
        param_files_dict = dict()
        for l, params_file  in enumerate(params_files):
          
            
            with np.load(params_file, allow_pickle=True, mmap_mode="r") as params:
                for param in params:  
                    if param == "allow_pickle":
                        continue
                    data = params[param]
                    length= 0
                    for pr in data:
                        length += len(pr)
                    param_files_dict[(total_length, total_length +  length)] = params_file
                    total_length += length
                  

        return params_files, total_length, center, param_files_dict
    
    def load_bg_scene_data(self, pc_path, max_sh_degree=3, device="cpu"):
        gaussians = GaussianModel(max_sh_degree)
        gaussians.load_ply(pc_path)
        screenspace_points = torch.zeros_like(gaussians.get_xyz, dtype=gaussians.get_xyz.dtype, requires_grad=True, device=device) + 0
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
            "means3D" : means3D.detach(),
            "means2D" : means2D.detach(),
            "shs" : shs.detach(),
            "colors_precomp" : colors_precomp,
            "opacities" : opacity.detach(),
            "scales" : scales.detach(),
            "rotations" : rotations.detach(),
            "cov3D_precomp" : cov3D_precomp
        }, gaussians.active_sh_degree


   
    def start_viewer(self):
    
        while True:
            if not self.running:
                # time.sleep(1)
                break
            time.sleep(360)
            print("Total number of clients connected to the demo:", len(self.render_viewers))
           
    def signal_handler(self,sig, frame):

        print('You pressed Ctrl+C!')

        self.running = False
      
        for key, val in self.render_viewers.items():
            # val.video_audio_event.set()
            # val.packetizer.exit = True
            print("killing render_viwer",key)
            val.running.value = self.running

            # time.sleep(1)

            # Cleanup the specific viewer
            val.clean_up()
        

            # Remove from dictionary
            del self.render_viewers[key]

            
            
        RenderViewers.clear_gpu_memory(True)
        viewer.viser_server.stop()

        sys.exit(0)
    
   


class RenderViewers():
  

  

    distance_in = 2 #2 
    distance_out = 11 #4.5
    theta_limits = (40, 90)
    phi_limits = (-np.inf, np.inf)

  
    

    def __init__(self, viewer, client ):
        if not torch.cuda.is_available():  
            raise RuntimeError("CUDA is not available")  
        self.viewer = viewer
        self.client = client
        self.encode_event = threading.Event()
        self.cuda_event = threading.Event()
        # self.cuda_event_2 = threading.Event()

        self.load_event = threading.Event()
        self.thread_cuda = threading.Thread(target=self.render_images)    
        self.thread_encode = threading.Thread(target=self.encode_image)
        self.thread_load_scenes = threading.Thread(target=self.load_scenes)
        self.thread_process_audio_buffers = threading.Thread(target=self.process_audio_buffers)
        self.thread_process_video_buffers =  threading.Thread(target=self.process_video_buffers)
        self.bg_scene_data_info = viewer.bg_scene_data_info
        self.max_sh_degree = viewer.max_sh_degree
     # Create multiprocessing context
        # torch_mp = torch_mp.get_context('spawn')

        # Create queues using the multiprocessing context
        self.render_input_queue = torch_mp.Queue(maxsize=3)
        self.render_output_queue = torch_mp.Queue(maxsize=3)
        self.render_event = torch_mp.Event()  # Create the event  
        self.render_ready_event = torch_mp.Event()  # Create the event  
        self.running = torch_mp.Value('b', True) 
        self.running_render = torch_mp.Value('b', True) 

        # Create render process
        self.render_process = torch_mp.Process(
            target=RenderViewers.render_process_function,
            args=(
                self.render_input_queue,
                self.render_output_queue,
                self.bg_scene_data_info,
                self.max_sh_degree,
                viewer.w,
                viewer.h,
                viewer.k,
                viewer.near,
                viewer.far,
                self.render_event ,           # Pass the event to the process  
                self.render_ready_event,
                self.running

            )
        )

        self.render_process.daemon = False  




        self.jump_timestep_flag = False
        self.triggered = False
        self.scene_data = torch_mp.Queue()
        self.params_files = viewer.params_files
        self.num_timesteps = viewer.num_timesteps
        
        self.params_files_dict = viewer.params_files_dict

        self.frame_rate = 30
        self.interval = 1.0/self.frame_rate
        self.look_at = viewer.look_at
        length = self.num_timesteps * self.interval

        self.gui_slider_bar_text = client.gui.add_slider_bar_text(0, length=length, color="#228be6")
       
        self.gui_play_button  = client.gui.add_button("Start_player", visible=True, icon=viser.Icon.PLAYER_PLAY, color="white", label_second="Pause_player",
                                                      icon_second=viser.Icon.PLAYER_PAUSE, color_second="white", visible_second=False)
        self.gui_player_skip_back_button= client.gui.add_button("Player_skip_back", visible=True,icon=viser.Icon.PLAYER_SKIP_BACK,  
                                                                color="white")
        self.gui_player_sound_button= client.gui.add_button("Player_sound_off", visible=True,icon=viser.Icon.VOLUME_OFF, color="white",
                                                            label_second="Player_sound_on",icon_second=viser.Icon.VOLUME, color_second="white",
                                                            visible_second=False)
        self.gui_repeat_button  = client.gui.add_button("Repeat", visible=False, icon=viser.Icon.REPEAT, color="white", label_second="Repeatoff",
                                                      icon_second=viser.Icon.REPEAT_OFF, color_second="white", visible_second=True)
        
        
        self.gui_play_button.on_click(self.handle_on_play_click)
        self.gui_player_skip_back_button.on_click(self.handle_on_play_skip_back_click)
        self.gui_player_sound_button.on_click(self.handle_on_play_sound_click)

        self.gui_repeat_button.on_click(self.handle_on_repeat_button_click)

        
        self.w = viewer.w
        self.h = viewer.h
        self.far = viewer.far   
        self.near = viewer.near
        self.k = viewer.k
        self.first_enter = True
        self.frame_number=0
        self.data_ready = False
     
        self.isPaused = True
        self.skip_back = False

    
    def handle_on_play_click(self, _):
        self.gui_play_button.visible= not self.gui_play_button.visible
        self.gui_play_button.visible_second= not self.gui_play_button.visible_second

        
        self.isPaused = not self.isPaused 
    # def handle_on_play_click(self, _):
        
    #     if not self.gui_play_button.disabled:
    #         self.gui_play_button.disabled = True
    #         self.gui_pause_button.disabled = False
    #         self.isPaused = False


    def handle_on_play_skip_back_click(self, _):
        self.gui_play_button.visible=True
        self.gui_play_button.visible_second=False
        
       
        while not self.scene_data.empty():  
            self.scene_data.get()
        self.load_event.set()
       
        self.isPaused = True
        self.jump_timestep_flag = True
        self.first_timestep = True
        self.start_timestep = 0
        # self.first_enter = True
        self.gui_slider_bar_text.value=0.0  
  
    def handle_on_play_sound_click(self, _):
        self.gui_player_sound_button.visible= not self.gui_player_sound_button.visible
        self.gui_player_sound_button.visible_second= not self.gui_player_sound_button.visible_second
   
    def handle_on_repeat_button_click(self,_):
        self.gui_repeat_button.visible= not self.gui_repeat_button.visible
        self.gui_repeat_button.visible_second= not self.gui_repeat_button.visible_second

    
  

    def load_bg_scene_data(self):
        gaussians = GaussianModel(self.max_sh_degree)
        gaussians.load_ply(self.pc_path)
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
            "means3D" : means3D.detach_(),
            "means2D" : means2D.detach_(),
            "shs" : shs.detach_(),
            "colors_precomp" : colors_precomp,
            "opacities" : opacity.detach_(),
            "scales" : scales.detach_(),
            "rotations" : rotations.detach_(),
            "cov3D_precomp" : cov3D_precomp
        }, gaussians.active_sh_degree
  

        
    def start(self):
       
        # self.thread_load_scenes.start()
        # # time.sleep(0.5)
        # self.thread_encode.start()
        # # time.sleep(0.5)

        # self.thread_cuda.start()
        # print(f"Thread_cuda started with PID:")  

        # self.render_process.start()
        # print(f"Render process started with PID: {self.render_process.pid}")  


        # self.thread_process_video_buffers.start()

        try:

            self.thread_load_scenes.start()
            # time.sleep(5)
            self.render_process.start()
            print(f"Render process started with PID: {self.render_process.pid}")
            self.thread_encode.start()

            
            self.thread_cuda.start()
            time.sleep(1)
            self.thread_process_video_buffers.start()

          


            # self.thread_process_audio_buffers.start()


        except Exception as e:
            print(f"Error starting components: {e}")
            # Clean up any started threads/processes
            self.running.value = False
            if self.thread_load_scenes.is_alive():
                self.thread_load_scenes.join()
            if self.thread_encode.is_alive():
                self.thread_encode.join()
            if self.thread_cuda.is_alive():
                self.thread_cuda.join()
            if self.render_process.is_alive():
                self.render_process.terminate()
                self.render_process.join(timeout=2)  
            if self.thread_process_video_buffers.is_alive():
                self.thread_process_video_buffers.join()
            # if self.thread_process_audio_buffers.is_alive():
            #    self.thread_process_audio_buffers.join() 
            raise




        # time.sleep(1)
        # self.thread_process_audio_buffers.start()

    def process_audio_buffers(self, audio_file="/home/hamit/Downloads/ocean.mp3"):
    # Replace this with your actual audio packet generation logic
        frame_number = 0
        self.packetizer = AudioPacketizer(audio_file, packet_duration_ms= self.interval * 1000*5 , force_mono=False, format="opus") 
        # interval = self.packetizer.packet_duration_ms / 1000.0  # Convert to seconds  


        while self.running :

            t0 = time.time() 
            # packet  = self.packetizer.get_next_packet() 
            self.audio_event.wait()
            self.audio_event.clear()
            if not self.isPaused:
                packet  = self.packetizer.get_next_packet() 
            else:
                # elapsed = time.time() - t0
                # sleep_time = interval - elapsed
                # if sleep_time > 0:
                #     time.sleep(sleep_time)
                continue
            if packet is None:  
                self.packetizer.reset()  
                continue  
            packet_data, time_stamp = packet
            # audio_packets = self.generate_audio_packets(audio_file)
            # for pack in audio_packets:
            # print(packet_data)
            try:
                self.client.scene.set_background_audio_pckt (
                    packet_data,
                    frame_number
                )
            except Exception as e:
                print(f"Error sending audio packet: {e}")

          


            frame_number += 1

           
   
    
    
    def process_video_buffers(self):
        frame_number=0

        try:
            while self.running.value:
               
                # start = time.time()
                size, data_ptr = gst_h264_endoder_pipeline.get_next_video_buffer_data(self.client_cnt)
                
                self.h264_pck = ctypes.string_at(data_ptr, size) 
                

                self.client.scene.set_background_h264_pckt(
                    self.h264_pck,
                    frame_number%300
                )
              
                
                # end = time.time()

                # if frame_number % 200 ==0:
                #     print(f"{self.client.client_id} Sending encoded packet  fps:", 1/(end-start))  
                # frame_number += 1
                 
        except KeyboardInterrupt:
            print("Stopped processing buffers.")
    
    def encode_image(self):
        self.client_cnt = gst_h264_endoder_pipeline.main_fun()
        print("Client num:",self.client_cnt)
        while self.running.value:
           
            self.encode_event.wait()
            self.encode_event.clear()
            
            
            try:
                if len(self.img) > 0:
                    size = self.img.numel() * self.img.element_size()
                    gst_h264_endoder_pipeline.push_tensor_frame(self.img.data_ptr(), size, self.frame_number, self.frame_rate, self.client_cnt)
                    self.frame_number += 1


            except Exception as e:
                # print(f"Error encoding {self.img_num}: {e}")
                pass
              
            self.cuda_event.set()
        self.cuda_event.set()    

        gst_h264_endoder_pipeline.close_pipeline(self.client_cnt)
      
    def load_scenes(self):
        

        while self.running.value:

         
            t = 0
           
            for interval, params_file in self.params_files_dict.items():
                
                
                if  self.jump_timestep_flag :
                    start, end = interval
                    
                    if start > self.start_timestep  or self.start_timestep >= end:
                        continue
                    

                    
                params = np.load(params_file, allow_pickle=True)
                # print(f"{params_file} loaded!")
    

                for param in params:
                    
                    if param == "allow_pickle":
                        continue
                    data = params[param]
                    
                    for pr in data:
                        cnt = 0
                        while cnt < len(pr):
                        # for  cnt, p in enumerate(pr):
                            if self.jump_timestep_flag:
                                try:
                                    start
                                    # print("start , cnt", start, cnt, self.start_timestep)
                                    if cnt + start != self.start_timestep:
                                        cnt+=1
                                        continue
                                    else:
                                        self.jump_timestep_flag = False
                                        t = self.start_timestep
                                        

                                except NameError:
                                    break
                            p = pr[cnt]
                            
                            if not self.isPaused:    
                                cnt += 1
                                t += 1
                          
                               

                            with torch.no_grad():

                                rendervar = {
                                    'means3D': torch.tensor(p['means3D'], dtype=torch.float32),
                                    'colors_precomp': torch.sigmoid(torch.tensor(p['rgb_colors'], dtype=torch.float32)),
                                    'rotations': torch.nn.functional.normalize(
                                        torch.tensor(p['unnorm_rotations'], dtype=torch.float32)
                                    ),
                                    'opacities': torch.sigmoid(
                                        torch.tensor(p['logit_opacities'], dtype=torch.float32)
                                    ),
                                    'scales': torch.exp(
                                        torch.tensor(p['log_scales'], dtype=torch.float32)
                                    ),
                                    'means2D': torch.zeros_like(
                                        torch.tensor(pr[0]['means3D'], dtype=torch.float32)
                                    )
                                }

                            # print("timestep:",t )
                            self.scene_data.put((rendervar, t))
                           
                           
                            if self.scene_data.qsize() > 3 :
                                self.load_event.wait()
                                self.load_event.clear()
                            if self.jump_timestep_flag:
                                # t = self.start_timestep
                                break
                if self.jump_timestep_flag:
                    break
            
               
                   
                          
           
            
        self.load_event.clear()

    

    
    def render_images(self):
        # t0 = time.time() + self.interval
        T_old = np.array([1, 1, 1])
        t = 0
        previous_ts = 0
        # bg_scened_data = {k: v.detach_() if v is not None else None for k,v in self.bg_scene_data[0].items()  }
        # for key, val in  self.bg_scene_data[0].items():
        #     scene_data[key] = val.cuda()
        # gaussians = self.load_bg_scene_data()
        # gaussians = bg_scene_data
        self.start_timestep = 0
        self.first_timestep = False
        while self.running.value:
            t0 = time.time()
            if self.first_enter:
                
                self.client.camera.up_direction = np.array([0, -1, 0])
                self.client.camera.position = self.init_camera()[1] 
                self.client.camera.look_at= self.look_at # for yoga 
                look_at_shift_up = deepcopy(self.look_at)
                look_at_shift_up[1] = 2.16
                self.client.camera.constraints = np.array([np.deg2rad(self.theta_limits[0]),np.deg2rad(self.theta_limits[1]), 
                                                            np.deg2rad(self.phi_limits[0]), np.deg2rad(self.phi_limits[1]),
                                                            self.distance_in, self.distance_out]
                                                            )
                
                self.first_enter = False

            # Calculate camera matrices
            R_S03 = tf.SO3(np.asarray(self.client.camera.wxyz))
            R = R_S03.as_matrix()
            T = self.client.camera.position
            if abs(T[0] - T_old[0]) > 0.01 or abs(T[1] - T_old[1]) > 0.01 or abs(T[2] - T_old[2]) > 0.01:
                is_cam_pose_changed = True
            else:
                is_cam_pose_changed = False

            
            
            relative_position =  look_at_shift_up - T
           
            c2w = np.vstack((np.concatenate((R, T[:,None]), axis=1),[0,0,0,1]))
            # w2c = np.linalg.inv(c2w)
            theta, phi, length_rel = self.get_theta_phi_angles_from_cam_pos(relative_position)
            
            try:
                scene_data, t = self.scene_data.get(timeout=2)
            except queue.Empty:
                print("scene empty")
                continue
            

            old_proggress_bar_value = float((t / self.num_timesteps)*100)
            # Send data to render process
            
            try:
                
                self.render_input_queue.put((c2w, theta, is_cam_pose_changed, scene_data))
                
                
            except Exception as e:
                # If queue is full, skip this frame
                print(f"Error putting data in render queue: {e}")  

                continue
            
            self.render_event.set() 
            # Wait for rendering to complete  
            
            self.render_ready_event.wait()
                     
            self.render_ready_event.clear()  
            
            try:
                self.img = self.render_output_queue.get_nowait()
                
                if old_proggress_bar_value + 2 < self.gui_slider_bar_text.value  or old_proggress_bar_value - 2 > self.gui_slider_bar_text.value:
                    if int(old_proggress_bar_value) == 0 and not self.isPaused:
                        t = 0
                        self.jump_timestep_flag = False
                        if self.gui_repeat_button.visible_second==True:
                                self.first_timestep = True 
                                self.start_timestep = t 
                                self.jump_timestep_flag = True
                                while not self.scene_data.empty():  
                                    self.scene_data.get()
                                self.load_event.set()
                                self.isPaused = not self.isPaused
                                self.gui_play_button.visible= not self.gui_play_button.visible
                                self.gui_play_button.visible_second= not self.gui_play_button.visible_second
             
                    else:
                        t = int(self.gui_slider_bar_text.value * self.num_timesteps/100)
                        self.jump_timestep_flag = True

                        
                    self.gui_slider_bar_text.value = float((t / self.num_timesteps)*100)
                    old_proggress_bar_value = self.gui_slider_bar_text.value
                  

                else:
                    self.first_timestep = False 
                    self.gui_slider_bar_text.value = old_proggress_bar_value
                        
                # img_num = self.img.numpy()

                # image = Image.fromarray(img_num)
                # image.save("img_trial.png")

                self.encode_event.set()
                
            except queue.Empty:
                print("Render timeout - frame dropped")
                # self.encode_event.set()

                continue
        

            self.cuda_event.wait()
            

            elapsed = time.time() - t0
            sleep_time = self.interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            T_old = T.copy()
            
            if self.scene_data.qsize() < 2 :
                
                self.load_event.set()
            if self.jump_timestep_flag and not self.first_timestep:
                self.start_timestep = t
                while not self.scene_data.empty():  
                    self.scene_data.get()
                self.load_event.set()

            self.cuda_event.clear()
        
        # im = self.img.cpu()
        # del im


        # try:
        #     while True:
        #         _, _, _, scene_data= self.render_input_queue.get_nowait()
        #         print("deleting inputs")
        #         for tensor in scene_data.values():
        #             if torch.is_tensor(tensor) and tensor.is_cuda:
        #                 ten = tensor.cpu()
                        
        #                 del ten
        #             else:
        #                 del tensor
        #             print("deleting inputs")
        # except queue.Empty:
        #     pass
    
    
        
        # try:
        #     while True:
        #         tensor = self.render_output_queue.get_nowait()
        #         print("deleting outputs")

        #         if torch.is_tensor(tensor) and tensor.is_cuda:
        #             # tensor.detach_()

        #             ten = tensor.cpu()
        #             del ten
        #         else:
        #             del tensor
        #         print("deleting outputs")
        # except queue.Empty:
        #     pass

        # try:
        #     while True:
        #         tensor, t = self.scene_data.get_nowait()
        #         print("deleting outputs")

        #         if torch.is_tensor(tensor) and tensor.is_cuda:
        #             # tensor.detach_()

        #             ten = tensor.cpu()
        #             del ten
        #         else:
        #             del tensor
        #         print("deleting outputs")
        # except queue.Empty:
        #     pass


        self.render_event.set()
        self.load_event.set()
        self.encode_event.set()
        

    
    def clean_up(self):
        """Thorough cleanup when a client disconnects"""
        # self.running_render.value = False
        # self.render_event.set()
        # self.render_ready_event.set()

        
        
        self.running.value = False
        # time.sleep(2)

        # Set all events to unblock threads
       
        self.encode_event.set()
        self.cuda_event.set()
        
        self.render_event.set()
        self.render_ready_event.set()
        self.load_event.set()
      
        if  self.render_process.is_alive():
            print("joining")
            self.render_process.join(timeout=2)  
           
            if self.render_process.is_alive(): 
                print("terminating")
                self.render_process.terminate() 
                self.render_process.join()

            if self.render_process.is_alive(): 
                print("killing")
                self.render_process.kill()  

        
       

        # Wait for threads to finish
        if hasattr(self, 'thread_load_scenes') and self.thread_load_scenes.is_alive():
            self.thread_load_scenes.join(timeout=2)
        # time.sleep(1)
        
        if hasattr(self, 'thread_cuda') and self.thread_cuda.is_alive():
            self.thread_cuda.join(timeout=2)

        if hasattr(self, 'thread_encode') and self.thread_encode.is_alive():
            self.thread_encode.join(timeout=2)

        if hasattr(self, 'thread_process_video_buffers') and self.thread_process_video_buffers.is_alive():
            self.thread_process_video_buffers.join(timeout=2)
        
       
        # while not self.render_input_queue.empty():
        #     try:
        #         _, _, _, scene_data= self.render_input_queue.get()
        #         for tensor in scene_data.values():
        #             if torch.is_tensor(tensor) and tensor.is_cuda:
        #                 ten = tensor.cpu()
                        
        #                 del ten
        #             else:
        #                 del tensor
        #             print("deleting inputs")
        #     except queue.Empty:
        #         break
        # self.render_input_queue.close()
   
        # while not self.render_output_queue.empty():
        #     try:
        #         tensor = self.render_output_queue.get()

        #         if torch.is_tensor(tensor) and tensor.is_cuda:
        #             # tensor.detach_()

        #             ten = tensor.cpu()
        #             del ten
        #         else:
        #             del tensor
        #     except queue.Empty:
        #         break
        # self.render_output_queue.close()
        
        #  # Terminate render process
        # try:
        #     while True:
        #         _, _, _, scene_data= self.render_input_queue.get_nowait()
        #         print("deleting inputs")
        #         for tensor in scene_data.values():
        #             if torch.is_tensor(tensor) and tensor.is_cuda:
        #                 ten = tensor.cpu()
                        
        #                 del ten
        #             else:
        #                 del tensor
        #             print("deleting inputs")
        # except queue.Empty:
        #     pass
    
    
        
        # try:
        #     while True:
        #         tensor = self.render_output_queue.get_nowait()
        #         print("deleting outputs")

        #         if torch.is_tensor(tensor) and tensor.is_cuda:
        #             # tensor.detach_()

        #             ten = tensor.cpu()
        #             del ten
        #         else:
        #             del tensor
        #         print("deleting outputs")
        # except queue.Empty:
        #     pass

        # try:
        #     while True:
        #         tensor, t = self.scene_data.get_nowait()
        #         print("deleting outputs")

        #         if torch.is_tensor(tensor) and tensor.is_cuda:
        #             # tensor.detach_()

        #             ten = tensor.cpu()
        #             del ten
        #         else:
        #             del tensor
        #         print("deleting outputs")
        # except queue.Empty:
        #     pass
        
        

        
        
        for attr_name in list(self.__dict__.keys()):
            attr_value = getattr(self, attr_name)
            
            if isinstance(attr_value, torch.Tensor) and attr_value.is_cuda:
                attr_value.cpu()  # Move to CPU first
                print(attr_name)
                delattr(self, attr_name)  # Then delete
        
        # gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("Clean-up Done!")
                
      
       



        
      
       

       
        

     

        # torch.cuda.empty_cache()
        # torch.cuda.ipc_collect()
        # torch.cuda.synchronize()


       
        # self.show_live_cuda_tensors()
        # RenderViewers.clear_gpu_memory(True)


   
    def show_live_cuda_tensors(self):
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) and obj.is_cuda:
                    print(f"Type: {type(obj)}, Size: {obj.size()}, Device: {obj.device}, Dtype: {obj.dtype}, RefCount: {sys.getrefcount(obj)}")
                elif hasattr(obj, 'data') and torch.is_tensor(obj.data) and obj.data.is_cuda:
                    print(f"Data Type: {type(obj)}, Size: {obj.data.size()}, Device: {obj.data.device}, Dtype: {obj.data.dtype, }RefCount: {sys.getrefcount(obj)} ")
            except Exception as e:
                pass
        if torch.cuda.is_available():
            print(f"Current GPU memory usage:")
            print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB") 
    
    @staticmethod
    def render_process_function(input_queue, output_queue, bg_scene_data_info, max_sh, w, h, k, near, far, render_event, render_ready_event, running):
        # global bg_scene_data
        # from diff_gaussian_rasterization import GaussianRasterizer as Renderer
        cuda_bg_scene_data = {} 
        # shared_tensors = {}
        shared_memories = {}
        for tensor_name, info in bg_scene_data_info.items():
            try:
                shm = shared_memory.SharedMemory(name=info['shm_name'])
                np_array = np.ndarray(
                    info['shape'], 
                    dtype=np.dtype(info['dtype']), 
                    buffer=shm.buf
                )
                tensor_cpu = torch.from_numpy(np_array)
                shared_memories[tensor_name] = shm
                cuda_bg_scene_data[tensor_name] = tensor_cpu.to("cuda", non_blocking=True)
                del tensor_cpu 

            except Exception as e:
                print(f"❌ Error connecting to shared memory '{tensor_name}': {e}")
                return
        

        print("Render process started")
        cnt = 0
        bg= [0, 177./255, 64.0/255]
        try:
            while running.value:
                try:
                    # print(running.value)
                    render_event.wait()
                    if not running.value:
                        break
                    # if not render_event.wait(timeout=0.1):  
                    #     continue 
                    render_event.clear()

                    try:
                        item = input_queue.get(timeout=0.1)
                    except queue.Empty:
                        render_ready_event.set()
                        continue

                    c2w, theta, is_cam_pose_changed, scene_data = item
                    
                    w2c = np.linalg.inv(c2w)
                    for key, val in scene_data.items():
                        scene_data[key] = val.cuda()
                    # scene_data = {key: torch.tensor(val).cuda().float() for key, val in scene_data.items()}
                    with torch.no_grad():
                        cam = setup_camera(w, h, k, w2c, near, far,
                                        bg=torch.tensor(bg, device='cuda'))
                        im, _, _, = Renderer(raster_settings=cam)(**scene_data)

                        if is_cam_pose_changed:
                            
                            c2w[0,3] += 0.70 #0.85
                            c2w[1,3] += -0.1 # -0.53 #0.27 #-0.4
                            c2w[2,3] += -3.5 #-0.34 # -3
                            # c2w[0,3] += 0.5
                            w2c_bg = np.linalg.inv(c2w)
                            cam_bg = setup_camera(w, h, k, w2c_bg, near, far, bg=torch.tensor(bg))
                            cam_dict = cam_bg._asdict()
                            cam_dict["sh_degree"] = max_sh
                            cam_dict["debug"] = False
                            cam_dict["antialiasing"] = False
                            cam_dict["bg"] = torch.tensor([0,0,0]).cuda().float()
                            # cam_dict["scale_modifier"] = 0.5
                            extented_cam = namedtuple('extented_cam', cam_dict.keys())  
                            cam_bg = extented_cam(**cam_dict) 
                        
                            im_bg,_,_ = Renderer(raster_settings=cam_bg)(**cuda_bg_scene_data)
                            img_bg_prev = im_bg
                        else:
                            im_bg = img_bg_prev
                        
                        if len(im.shape) == 3:
                            im = im.unsqueeze(0)  
                            im_bg = im_bg.unsqueeze(0)  
                        
                        green_color_threshold=0.72
                        target_green = torch.tensor([0, 177/255, 64/255]).to(im.device)

                        # Calculate difference from target green for each pixel
                        diff = torch.abs(im - target_green.view(1, 3, 1, 1))

                        # Create mask where pixels are close to target green
                        mask = (diff.sum(dim=1, keepdim=True) > green_color_threshold).float()
                        
                        # Step 2: Create mask for non-black areas of background  
                        if theta > 90:
                            black_threshold = 0.3
                            bg_nonblack_mask = (im_bg.abs().sum(dim=1, keepdim=True) > black_threshold).float() 
                            result = im_bg * bg_nonblack_mask + im * mask * (1 - bg_nonblack_mask) 
                            #Set background green
                            # bg_diff = torch.abs(im_bg - target_green.view(1, 3, 1, 1))
                            # bg_green_mask = (bg_diff.sum(dim=1, keepdim=True) < green_color_threshold).float()
                            # result = im_bg * (1 - bg_green_mask) + im  * bg_green_mask

                        else:  

                            # Combine images using the mask
                            result = im * mask + im_bg * (1 - mask)

                        result = result.squeeze(0)

                        result =  result.permute(1,2,0).contiguous()
                        result = (result.clamp(0,1)*255).to(torch.uint8)
                        
                        try:  
                            output_queue.put(result, timeout=0.1)  
                        except queue.Full:  
                        # Queue is full, skip or handle appropriately  
                            pass 
                        render_ready_event.set()

                        # im = im.permute(1,2,0).contiguous()
                        # im = (im.clamp(0,1)*255).to(torch.uint8)
                
                        # output_queue.put(im)
                        # render_ready_event.set()

                    # Periodic cleanup
                    if cnt % 30 == 0:
                        torch.cuda.empty_cache()
                    cnt += 1

                except Exception as e:
                    print(f"Error in render process: {e}")
                    render_ready_event.set()
                    continue
                # print("begin")
            
            
        
        except Exception as e:  
            print(f"Render process error: {e}")  
        finally:
            # try:
            #     while True:
            #         _, _, _, scene_data= input_queue.get_nowait()
            #         print("deleting inputs")
            #         for tensor in scene_data.values():
            #             if torch.is_tensor(tensor) and tensor.is_cuda:
            #                 ten = tensor.cpu()
                            
            #                 del ten
            #             else:
            #                 del tensor
            #             print("deleting inputs")
            # except queue.Empty:
            #     pass
        
        
            
            # try:
            #     while True:
            #         tensor = output_queue.get_nowait()
            #         print("deleting outputs")

            #         if torch.is_tensor(tensor) and tensor.is_cuda:
            #             # tensor.detach_()

            #             ten = tensor.cpu()
            #             del ten
            #         else:
            #             del tensor
            #         print("deleting outputs")
            # except queue.Empty:
            #     pass
        
            # print("begin deleting")
            result = result.cpu()
            del result
            for shm in shared_memories.values():
                try:
                    shm.close()
                except:
                    pass
            for gaus in cuda_bg_scene_data.values():
                if torch.is_tensor(gaus) :
                    g = gaus.cpu()
                    # print("deleting gs")
                    del g
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()
            #     torch.cuda.synchronize()

            print("Cleaned in Render Function")
        
        
    

    
    @classmethod
    def init_camera(cls, y_angle=0., center_dist=4., cam_height= 3., f_ratio=0.82):
        ry = y_angle * np.pi / 180
        # w2c = np.array([[np.cos(ry), 0., -np.sin(ry), -0.0],
        #                 [0.,         1., 0.,          cam_height],
        #                 [np.sin(ry), 0., np.cos(ry),  center_dist],
        #                 [0.,         0., 0.,          1.]])
        
        c2w = np.array([[-0.94743326, -0.0982282 , -0.30450196,  2.2777 ],
                [-0.09894314,  0.99500657, -0.01312203 , 0.185],
                [ 0.3042704 ,  0.01769613, -0.95242132,  3.5],
                [ 0.  ,        0.    ,      0.  ,        1.        ]])
        
        # c2w = np.array([[-0.95418331, -0.12057518 , -0.27385366 , 2.13062697],
        #     [-0.09921866 , 0.9909332,  -0.09059276 , -1.56769998],
        #     [ 0.28229393, -0.05927071 ,-0.95749523 , 8.50743482],
        #     [ 0. ,         0.  ,        0.  ,        1.        ]])
        # c2w = np.linalg.inv(w2c)
        wxyz = tf.SO3.from_matrix(c2w[:3,:3]).wxyz
        return wxyz, c2w[:3,3]  
    @classmethod
    def get_theta_phi_angles_from_cam_pos(cls, position):

        # Calculate the length (magnitude) of the position vector
        position_length = np.linalg.norm(position)

        # Calculate the polar angle (theta) relative to the y-axis
        # Theta = arccos(y / ||position||)
        theta = np.arccos(position[1] / position_length)
        phi = np.arctan2(position[2], position[0])

        # Convert theta to degrees
        theta_degrees = np.degrees(theta)
        phi_degrees = np.degrees(phi)  


        return theta_degrees, phi_degrees, position_length

    @staticmethod
    def clear_gpu_memory(aggressive=False):
        """
        Clear GPU memory with different levels of aggressiveness

        Args:
            aggressive (bool): If True, tries to clear everything possible
        """
        # Basic cleaning
        torch.cuda.empty_cache()

        # Count tensors before cleaning
        total_before = 0
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) and obj.is_cuda:
                    total_before += 1
            except:
                pass

        print(f"Found {total_before} CUDA tensors before cleaning")

        # Delete all references to tensors and torch modules
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj):
                    if obj.is_cuda:
                        print(f"Deleting tensor of size {obj.size()} on {obj.device}")
                        del obj
                elif hasattr(obj, 'data') and torch.is_tensor(obj.data):
                    if obj.data.is_cuda:
                        print(f"Deleting tensor object of size {obj.data.size()} on {obj.data.device}")
                        del obj
            except:
                pass

        if aggressive:
            # Clear cuda cache
            torch.cuda.empty_cache()

            # Run garbage collector
            gc.collect()

            # Clear any remaining cuda memory
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            # Reset peak memory stats
            torch.cuda.reset_peak_memory_stats()

            # If you have any models, you might want to move them to CPU
            for obj in gc.get_objects():
                try:
                    if isinstance(obj, torch.nn.Module):
                        obj.cpu()
                except:
                    pass

        # Count tensors after cleaning
        total_after = 0
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) and obj.is_cuda:
                    total_after += 1
            except:
                pass

        print(f"Found {total_after} CUDA tensors after cleaning")
        print(f"Cleaned {total_before - total_after} tensors")

    # Print current memory usage
        if torch.cuda.is_available():
            print(f"Current GPU memory usage:")
            print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    
if __name__ == "__main__":


    


    # exp_name = "samples"
    # sequence = "sample_1"


    # exp_name = "2025-08-06_3412x2500_combin1_all_multiprocess"
    # sequence = "2025-08-06_3412x2500_combin1"

    exp_name = "2025-08-06_3412x2500_combin2_all_multiprocess"
    sequence = "2025-08-06_3412x2500_combin2"

    # exp_name = "2025-08-06_3412x2500_combin3_all_multiprocess"
    # sequence = "2025-08-06_3412x2500_combin3"
    
    # pc_path = "/home/ubuntu/splat_plys/filtered_point_cloud_rotated_scaled144_model_yplane.ply" 
    # pc_path = "/home/ubuntu/splat_plys/Cherry_Blossom_Tree_Splat_cleaned_trans_sh_0.ply"

    


    # pc_path = "/home/hamit/Documents/splat_plys/8_25_2025_cleaned_scaled6.ply"
    # pc_path = "/home/hamit/Documents/splat_plys/bicycle_2_cleaned_trans_sh_3.ply"



    # pc_path = "/home/hamit/Documents/splat_plys/filtered_point_cloud_rotated_scaled144_model_sh_0.ply" 
    # pc_path = "/home/hamit/Documents/splat_plys/Stone_Gravel_Patch_trans_sh_3.ply"

    # bg_scene_data = load_bg_scene_data(pc_path, max_sh_degree=3)
        
    pc_path = "/home/hamit/Documents/splat_plys/bicycle_1_cleaned_trans_sh_0.ply"

    torch_mp.set_start_method('spawn', force=True)
    torch_mp.set_sharing_strategy('file_system')
    
    viewer = Viewer(seq=sequence, exp=exp_name, pc_path=pc_path, port=8089, title="360° Demo10 Spaceport", w=1920, h=1080, max_sh_degree=0)

    # viewer = Viewer(seq=sequence, exp=exp_name,w=1920, h=1080)
    time.sleep(0.2)
    viewer.start_viewer()

