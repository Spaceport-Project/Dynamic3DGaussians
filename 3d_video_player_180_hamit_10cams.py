import os
import random
import threading
from typing import Dict
import torch
import numpy as np
import time
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
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
from io import BytesIO  




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


from io import BytesIO
import time
from pydub import AudioSegment

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
        if self.position >= len(self.audio) or self.exit :
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
    #         """Reset the position to the beginning of the audio"""
        self.position = 0

# class AudioPacketizer:
#     def __init__(self, mp3_path: str, packet_duration_ms: int = 50, force_mono: bool = False):
#         """
#         Initialize the audio packetizer
#         Args:
#             mp3_path: Path to MP3 file
#             packet_duration_ms: Duration of each packet in milliseconds (default: 50ms)
#             force_mono: If True, converts stereo to mono. If False, maintains original channels (default: False)
#         """
#         self.audio = AudioSegment.from_mp3(mp3_path)
#         self.packet_duration_ms = packet_duration_ms
#         self.position = 0
#         self.overlap_ms = 15

#         # Convert to mono if requested
#         if force_mono and self.audio.channels > 1:
#             self.audio = self.audio.set_channels(1)

#         self.channels = self.audio.channels

#         # Ensure consistent sample rate (44.1kHz is standard)
#         self.audio = self.audio.set_frame_rate(44100)

#         # Set consistent format
#         self.audio = self.audio.set_sample_width(2)  # 16-bit

#     def get_next_packet(self) -> tuple[bytes, int] | None:
#         """
#         Get the next audio packet
#         Returns:
#             Tuple of (packet_data, timestamp) or None if end of audio
#         """
#         if self.position >= len(self.audio):
#             return None

#         # Extract packet
#         end_pos = min(self.position + self.packet_duration_ms, len(self.audio))
#         packet = self.audio[self.position:end_pos]
#         packet = packet.fade_in(self.overlap_ms).fade_out(self.overlap_ms)

#         # Export packet as MP3
#         buffer = BytesIO()

#         # Adjust export parameters based on number of channels
#         export_params = [
#             "-ar", "44100",      # Sample rate
#             "-ac", str(self.channels),  # Number of channels (1 or 2)
#             "-b:a", "192k",      # Higher bitrate for Windows
#             "-q:a", "0",         # Highest quality
#             "-bufsize", "192k",  # Larger buffer size
#             "-compression_level", "0"  # Fastest compression
#         ]

#         # Add joint stereo parameter only for stereo audio
#         if self.channels == 2:
#             export_params.extend(["-joint_stereo", "0"])  # Disable joint stereo for stereo

#         packet.export(buffer,
#                      format='mp3',
#                      parameters=export_params)

#         packet_data = buffer.getvalue()

#         # Calculate timestamp
#         timestamp = int(time.time() * 1000)  # Current timestamp in milliseconds

#         # Update position with overlap
#         self.position = end_pos - self.overlap_ms

#         return packet_data, timestamp

# class AudioPacketizer:
#     def __init__(self, mp3_path: str, packet_duration_ms: int = 50):
#         """
#         Initialize the audio packetizer
#         Args:
#             mp3_path: Path to MP3 file
#             packet_duration_ms: Duration of each packet in milliseconds (default: 22ms)
#         """
#         self.audio = AudioSegment.from_mp3(mp3_path)
#         self.packet_duration_ms = packet_duration_ms
#         self.position = 0
#         self.overlap_ms = 15

#         # Convert to mono if stereo
#         if self.audio.channels > 1:
#             self.audio = self.audio.set_channels(1)
#         assert self.audio.channels == 1, "Audio must be mono"  
#            # Ensure consistent sample rate (44.1kHz is standard)  
#         self.audio = self.audio.set_frame_rate(44100)  

#         # Set consistent format  
#         self.audio = self.audio.set_sample_width(2)  # 16-bit


#     def get_next_packet(self) -> tuple[bytes, int] | None:
#         """
#         Get the next audio packet
#         Returns:
#             Tuple of (packet_data, timestamp) or None if end of audio
#         """
#         if self.position >= len(self.audio):
#             return None

#         # Extract packet
#         end_pos = min(self.position + self.packet_duration_ms, len(self.audio))
#         packet = self.audio[self.position:end_pos]
#         packet = packet.fade_in(self.overlap_ms).fade_out(self.overlap_ms)  

#         # Export packet as MP3
#         buffer = BytesIO()
#         packet.export(buffer, format='mp3', 
#                       parameters=[
#                         "-ar", "44100",  # Sample rate
#                         "-ac", "1",      # Mono
#                         "-b:a", "192k",     # Higher bitrate for Windows  
#                         "-joint_stereo", "0",  # Disable joint stereo
#                         "-q:a", "0",     # Highest quality
#                         "-bufsize", "192k", # Larger buffer size
#                         "-compression_level", "0"  # Fastest compression
#                         ]
#                       )
#         # packet.export(buffer,
#         #             format='ipod',
#         #             parameters=[
#         #             "-ar", "44100",
#         #             "-ac", "1",
#         #             "-b:a", "96k",
#         #             "-q:a", "0"
#         #         ]
               
                   
#         # )
#         packet_data = buffer.getvalue()

#         # Calculate timestamp
#         timestamp = int(time.time() * 1000)  # Current timestamp in milliseconds

#         # Update position
#         # self.position = end_pos
#         self.position = end_pos - self.overlap_ms  


#         return packet_data, timestamp

#     def reset(self):
#         """Reset the position to the beginning of the audio"""
#         self.position = 0


  
        
class Viewer():

    def __init__(self, seq, exp, port, title="", f_ratio=0.8, w=1920, h=1080, near=0.01, far=100.0):
        self.seq = seq
        self.exp = exp
        self.viser_server = viser.ViserServer(port=port, title=title)
        self.viser_server.scene.world_axes.visible = False
        self.clients_num = 0
        self.k = np.array([[f_ratio * w, 0, w / 2], [0, f_ratio * w, h / 2], [0, 0, 1]])
        self.w = w
        self.h = h
        self.near = near
        self.far = far
        self.scene_data, _ , self.look_at = self._load_scene_data3(self.seq, self.exp, seg_as_col=False)
        self.render_viewers: Dict[int, RenderViewers] = {}
        signal.signal(signal.SIGINT, self.signal_handler)
        self.viser_server.on_client_connect(self.handle_new_client)
        self.viser_server.on_client_disconnect(self.handle_disconnect_client)
        self.running = True
   
        



    def handle_disconnect_client(self, client:viser.ClientHandle):
        print(f"{client.client_id} client disconnected!")
        self.render_viewers[client.client_id].running = False
        self.render_viewers[client.client_id].thread_cuda.join()
        self.render_viewers[client.client_id].thread_encode.join()
        self.render_viewers[client.client_id].thread_process_video_buffers.join()
        self.render_viewers[client.client_id].thread_process_audio_buffers.join()

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
        
        params = dict(np.load(f"./output/{exp}/{seq}/params_9.npz"))
    

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
        pc = np.load(os.path.join(f"./output/{exp}/{seq}/", "init_pt_cld.npz"))

        xyz = [vert[:3] for vert in pc['data']]
        xyz = np.asarray(xyz)
        center = np.mean(xyz[:], axis=0)
        print("Foreground center:",center)
        # print(params_file)
        scene_data = []
        total = 0
        for l, param_file  in enumerate(params_file):
            # if l > 10:
            #     break
            # if  l > 0:
            #     continue
            params = dict(np.load(param_file))  
            print(f"{param_file} loaded!")

            params = {k: torch.tensor(v).cuda().float() for k, v in params.items()}
            is_fg = params['seg_colors'][:, 0] > 0.5
            # if l == 69:
            #     length=80
            # elif l ==75:
            #     length = 5
            # else:
            length = len(params['means3D'])
            total = total + length
            print(f"total timesteps:", total, l)
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
        return scene_data, is_fg, center
    def _load_scene_data4(self, seq, exp, seg_as_col=False):
        params_files =[ os.path.join(f"./output/{exp}/{seq}/", file) for file in  os.listdir(f"./output/{exp}/{seq}/") if file.startswith("params")]
        params_files = sorted(params_files, key= lambda x : int(os.path.basename(x).split("_")[1].split(".")[0]) if len(os.path.basename(x).split("_")) > 1 else os.path.basename(x).split("_")[0].split(".")[0]) 
        pc = np.load(os.path.join(f"./output/{exp}/{seq}/", "init_pt_cld.npz"))

        xyz = [vert[:3] for vert in pc['data']]
        xyz = np.asarray(xyz)
        center = np.mean(xyz[:], axis=0)
        print("Foreground center:",center)
        # print(params_file)
        scene_data = []
        total_length = 0
        for l, params_file  in enumerate(params_files):
          
            params = np.load(params_file, allow_pickle=True)
            print(f"{params_file} loaded!")
  

            for param in params:
                if param == "allow_pickle":
                    continue
                data = params[param]
                
                for pr in data:
                    total_length += len(pr)
                    for id, p in enumerate(pr):
                        
                        rendervar = {
                            'means3D': torch.tensor(p['means3D']).cuda().float(),
                            'colors_precomp':  torch.tensor(p['rgb_colors']).cuda().float() if not seg_as_col else torch.tensor(p['seg_colors']).cuda().float(),
                            'rotations': torch.nn.functional.normalize( torch.tensor(p['unnorm_rotations']).cuda().float()),
                            'opacities': torch.sigmoid(torch.tensor(pr[0]['logit_opacities']).cuda().float()),
                            'scales': torch.exp(torch.tensor(pr[0]['log_scales']).cuda().float()),
                            'means2D': torch.zeros_like( torch.tensor(pr[0]['means3D']).float(), device="cuda")
                        }

                        scene_data.append(rendervar)


        is_fg = False #params['seg_colors'][:, 0] > 0.5
        print("total length:", total_length) 
        
        return scene_data, is_fg, center

    def start_viewer(self):
    
        while True:
            if not self.running:
                break
            print("Total number of clients connected to the demo:", len(self.render_viewers))
            time.sleep(3600)
    def signal_handler(self,sig, frame):

        print('You pressed Ctrl+C!')

        self.running = False
        for key, val in self.render_viewers.items():
            # val.video_audio_event.set()
            val.packetizer.exit = True

            val.running = self.running
            val.thread_cuda.join()
            val.thread_encode.join()
            val.thread_process_video_buffers.join()
            val.thread_process_audio_buffers.join()

            
        
        viewer.viser_server.stop()

        sys.exit(0)
    
   

class RenderViewers():
  


    distance_in = 5 #2 
    distance_out = 20 #4.5

    theta_limits = (40, 110)
    # phi_limits = (-117, 108)
    phi_limits = (75, 247)


  
    

    def __init__(self, viewer, client ):
        self.viewer = viewer
        self.client = client
        self.encode_event = threading.Event()
        self.cuda_event = threading.Event()
        self.audio_event = threading.Event()
        self.thread_cuda = threading.Thread(target=self.render_images)    
        self.thread_encode = threading.Thread(target=self.encode_image)
        self.thread_process_audio_buffers = threading.Thread(target=self.process_audio_buffers)

        self.thread_process_video_buffers =  threading.Thread(target=self.process_video_buffers)
        self.running = True
        self.triggered = False
        self.scene_data = viewer.scene_data
        self.look_at = viewer.look_at
   
        self.frame_rate = 30
        self.interval = 1.0/self.frame_rate
        length=len(self.scene_data) * self.interval
       
        self.gui_slider_bar_text = client.gui.add_slider_bar_text(0, length=length, color="#228be6")
       
        self.gui_play_button  = client.gui.add_button("Start_player", visible=True, icon=viser.Icon.PLAYER_PLAY, color="white", label_second="Pause_player",
                                                      icon_second=viser.Icon.PLAYER_PAUSE, color_second="white", visible_second=False)
        self.gui_player_skip_back_button= client.gui.add_button("Player_skip_back", visible=True,icon=viser.Icon.PLAYER_SKIP_BACK,  
                                                                color="white")
        self.gui_player_sound_button= client.gui.add_button("Player_sound_off", visible=True,icon=viser.Icon.VOLUME_OFF, color="white",
                                                            label_second="Player_sound_on",icon_second=viser.Icon.VOLUME, color_second="white",
                                                            visible_second=False)
       
        
        # self.grid_plane = client.scene.add_grid(
        #             "/grid",
        #             width=25.0,
        #             height=25.0,
        #             # width_segments=50,
        #             # height_segments=50,
        #             plane="xz",
        #             cell_thickness=5.0,
        #             section_thickness=3.0,
        #             # position= (0.12362304, 2.0, 0.37942997), #(0.0, 2.5, 0.0),
        #             # wxyz= (-0.23188804, -0.04431691,  0.97151249,  0.02067442) #(0.70872991,  0.        , -0.35172332,  0.61154936) 
            
        # )


        # client.scene.add_image(
        #     "/noise",
        #     np.random.randint(0, 256, size=(400, 400, 3), dtype=np.uint8),
        #     4.0,
        #     4.0,
        #     format="jpeg",
        #     wxyz=(1.0, 0.0, 0.0, 0.0),
        #     position=(2.0, 2.0, -1e-2),
        # )
        self.gui_play_button.on_click(self.handle_on_play_click)
        self.gui_player_skip_back_button.on_click(self.handle_on_play_skip_back_click)
        self.gui_player_sound_button.on_click(self.handle_on_play_sound_click)

        self.w = viewer.w
        self.h = viewer.h
        self.far = viewer.far   
        self.near = viewer.near
        self.k = viewer.k
        self.first_enter = False
        self.frame_number=0
        self.data_ready = False
     
        self.isPaused = True
        self.skip_back = False
        # self.increment=1
        # self.mutex = threading.Lock()  

    
    def handle_on_play_click(self, _):
        self.gui_play_button.visible= not self.gui_play_button.visible
        self.gui_play_button.visible_second= not self.gui_play_button.visible_second

        
        self.isPaused = not self.isPaused 


    
    def handle_on_play_skip_back_click(self,_):
        self.gui_play_button.visible=True
        self.gui_play_button.visible_second=False
        # self.isPaused = True
        # self.gui_play_button.value=False
        # self.current_ts = len(self.scene_data)
        self.isPaused = True
        # time.sleep(0.1)

        self.skip_back = True
        self.gui_slider_bar_text.value=0.0
        # self.gui_text.value=f"{0:02d}:{0:02d}"

    def handle_on_play_sound_click(self,_):
        self.gui_player_sound_button.visible= not self.gui_player_sound_button.visible
        self.gui_player_sound_button.visible_second= not self.gui_player_sound_button.visible_second



    




        
    def start(self):
       

        self.thread_encode.start()
        time.sleep(0.5)

        self.thread_cuda.start()
       
        self.thread_process_video_buffers.start()
        self.thread_process_audio_buffers.start()

   

    def process_audio_buffers(self, audio_file="/home/hamit/Downloads/Journey.mp3"):
    # Replace this with your actual audio packet generation logic
        frame_number = 0
        self.packetizer = AudioPacketizer(audio_file, packet_duration_ms= self.interval * 1000*5 , force_mono=False, format="opus") 
        # interval = self.packetizer.packet_duration_ms / 1000.0  # Convert to seconds  


        while self.running :

            # t0 = time.time() 
            # packet  = self.packetizer.get_next_packet() 
            self.audio_event.wait()
            self.audio_event.clear()
            # print(frame_number)

            if not self.isPaused:
                packet  = self.packetizer.get_next_packet() 
            else:
                # time.sleep(self.interval)
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

            # More precise timing control
            # elapsed = time.time() - t0
            # sleep_time = interval - elapsed
            # if sleep_time > 0:
            #     time.sleep(sleep_time)
            
            

            frame_number += 1

        # Split audio into chunks
           
    
    def process_audio_buffers2(self, audio_file="/home/hamit/Downloads/yoga1.wav", segment_duration_ms=200):
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
       
        interval = segment_duration_ms/1000.0


        while self.running :

                # t0 = time.time() 
                start_time = 0
                for i, timestamp in enumerate(image_timestamps):
                    t0 = time.time() 

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
                    
                    elapsed = time.time() - t0
                    sleep_time = interval - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                   

                    end = time.time()

                    # print("Audio frame number", frame_number)
                    if frame_number % 50 ==0:      
                        print(f"{self.client.client_id} Audio Process Buffer fps:", 1.0/(end-start))  
                    frame_number += 1
                  

       
    
    
    def process_video_buffers(self):
        frame_number=0

        try:
            while self.running:
               
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
                size = self.img.numel() * self.img.element_size()
                
                try:
                    
                    gst_h264_endoder_pipeline.push_tensor_frame(self.img.data_ptr(), size, self.frame_number, self.frame_rate, self.client_cnt)
                    


                except Exception as e:
                    print(f"Error encoding {self.img_num}: {e}")
              
            self.cuda_event.set()
        self.cuda_event.set()    

        gst_h264_endoder_pipeline.close_pipeline(self.client_cnt)
      
            
    def render_images(self):
        
        num_timestamps = len(self.scene_data)
        # percent_incr = (2.0/num_timestamps)*100
    

        while self.running:
            self.frame_number = 0 
            previous_ts = 0
            t = 0
            while t < num_timestamps:
            # for t in range(num_timestamps): 
            
                
                if not self.running:
                    break
                percent = (t / num_timestamps)*100
                old_proggress_bar_value = float(percent)
                
               
                while True:
                    t0 = time.time()
                    if not self.running:
                        break
                    if not self.first_enter: 
                        self.client.camera.up_direction = np.array([0,-1,0]) #self.init_camera()[0]
                        self.client.camera.position = self.init_camera()[1] 
                        self.client.camera.look_at= self.look_at # for yoga 
                        self.client.camera.constraints = np.array([np.deg2rad(self.theta_limits[0]),np.deg2rad(self.theta_limits[1]), 
                                                                   np.deg2rad(self.phi_limits[0]), np.deg2rad(self.phi_limits[1]),
                                                                   self.distance_in, self.distance_out]
                                                                   )
                        self.first_enter = True

                    R_S03 = tf.SO3(np.asarray(self.client.camera.wxyz))
                    R = R_S03.as_matrix()
                    T = self.client.camera.position

                  

                    c2w = np.vstack((np.concatenate((R, T[:,None]), axis=1),[0,0,0,1]))
                    w2c = np.linalg.inv(c2w)
                    print(repr(c2w))
                  
                    self.img = self._render(w2c, self.scene_data[t], bg=[0, 177.0/255, 64.0/255])

                    if old_proggress_bar_value + 4 < self.gui_slider_bar_text.value  or old_proggress_bar_value - 4 > self.gui_slider_bar_text.value:
                        if old_proggress_bar_value == 0 and not self.isPaused:
                            t = 0
                        else:
                            t = int(self.gui_slider_bar_text.value * num_timestamps/100)
    
                        self.gui_slider_bar_text.value = float((t / num_timestamps)*100)
                        old_proggress_bar_value = self.gui_slider_bar_text.value

                        if self.isPaused:
                            previous_ts = t  

                    else:
                        
                        self.gui_slider_bar_text.value = old_proggress_bar_value
                        
                        
                       
                    self.encode_event.set()
                    if t % 5 == 0 :
                        self.audio_event.set()

                    self.cuda_event.wait()
                   

                    
                    
                    elapsed = time.time() - t0
                    sleep_time = self.interval - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    
                    
                    self.cuda_event.clear()

                   
                        

                    if self.skip_back:
                        # print("it is breaking")
                        break

                    if not self.isPaused:
                        previous_ts =  t 
                        self.frame_number +=1
                        break
                    else:
                        t = previous_ts

                
              

                if self.skip_back:
                    self.skip_back = False
                    # print("it is breaking")
                    break
            
                t += 1

        self.encode_event.set()
        self.audio_event.set()

                



    def _render(self, w2c, timestep_data, bg=[0,0,0]):
        with torch.no_grad():
            cam = setup_camera(self.w, self.h, self.k, w2c, self.near, self.far, bg=torch.tensor(bg)) #[0, 177./255, 64.0/255]
            im, radi, _, = Renderer(raster_settings=cam)(**timestep_data)
            # im[~is_fg] =  torch.tensor([0, 177./255, 64.0/255], dtype=torch.float32, device="cuda")
            # torchvision.utils.save_image(im, '{0:05d}'.format(cnt) + ".png")
            # im = torch.flip(im, dims=[0])
            im =  im.permute(1,2,0).contiguous()
            im = (im.clamp(0,1)*255).to(torch.uint8)
            return im
    
    
    @classmethod
    def init_camera(cls, y_angle=0., center_dist=4., cam_height= 3., f_ratio=0.82):
        ry = y_angle * np.pi / 180
      
        # c2w = np.array( [[-0.8885279, -0.07652057, -0.45239671,  3.07338111],
        #     [-0.09569717,  0.99521716,  0.01961779 , 0.69955555],
        #     [ 0.4487318 ,  0.06072404, -0.89160101,  6.19292782],
        #     [ 0.   ,       0.  ,        0.   ,       1.        ]])
        
        c2w = np.array([[ 0.84822181,  0.13134315, -0.51309721,  4.45859424],
       [ 0.03296448,  0.95379384,  0.29864805, -1.69569923],
       [ 0.52861433, -0.27023377,  0.80469907, -6.41917903],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
        wxyz = tf.SO3.from_matrix(c2w[:3,:3]).wxyz
        print(wxyz, c2w[:3,3], np.linalg.inv(c2w))
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


        return theta_degrees, phi_degrees


if __name__ == "__main__":




    
    exp_name = "2024-12-04_17-14-42_10bestcams_2_scl_2_test1"
    sequence = "2024-12-04_17-14-42_10bestcams_2"


        
    viewer = Viewer(seq=sequence, exp=exp_name, port=8084, title="10 Camera - Demo2 Spaceport", w=1920, h=1080)
    time.sleep(0.2)
    viewer.start_viewer()
   


   

