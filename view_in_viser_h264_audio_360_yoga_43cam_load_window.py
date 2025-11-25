
import inspect
import os
import random
import threading
from typing import Dict
import torch
import numpy as np
import time
from helpers import setup_camera, quat_mult, searchForMaxIteration
# from diff_gaussian_rasterization import GaussianRasterizer as Renderer

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
from queue import Queue  
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

to8b = lambda x : (255*np.clip(x.permute(1,2,0).contiguous().cpu().detach().numpy(),0,1)).astype(np.uint8)




# def optimize_memory_usage():
#     # Before optimization
#     torch.cuda.reset_peak_memory_stats()
#     # Run operation
#     peak_before = torch.cuda.max_memory_allocated()

#     # Apply optimization
#     torch.cuda.empty_cache()
#     torch.cuda.reset_peak_memory_stats()
#     # Run optimized operation
#     peak_after = torch.cuda.max_memory_allocated()

#     improvement = (peak_before - peak_after) / 1e9
#     print(f"Memory improvement: {improvement:.2f} GB")

  
        
class Viewer():

    def __init__(self, seq, exp, f_ratio=0.8, w=1920, h=1080, near=0.01, far=100.0):
        self.seq = seq
        self.exp = exp
        self.viser_server = viser.ViserServer(port=8084)
        self.viser_server.scene.world_axes.visible = False
        self.clients_num = 0
        self.k = np.array([[f_ratio * w, 0, w / 2], [0, f_ratio * w, h / 2], [0, 0, 1]])
        self.w = w
        self.h = h
        self.near = near
        self.far = far
        # self.scene_data, _ , self.look_at = self._load_scene_data3(self.seq, self.exp, seg_as_col=False)
        self.params_files, self.num_timesteps, self.look_at = self._load_list_params_files(self.seq, self.exp)
        self.render_viewers: Dict[int, RenderViewers] = {}
        signal.signal(signal.SIGINT, self.signal_handler)
        self.viser_server.on_client_connect(self.handle_new_client)
        self.viser_server.on_client_disconnect(self.handle_disconnect_client)
        self.running = True
   
        




    def handle_disconnect_client(self, client:viser.ClientHandle):
        # print("Before disconnection",torch.cuda.memory_summary())  
        print(f"{client.client_id} client disconnected!")
        self.render_viewers[client.client_id].running = False
        self.render_viewers[client.client_id].thread_cuda.join()

        # self.render_viewers[client.client_id].thread_encode.join()
        # self.render_viewers[client.client_id].thread_process_video_buffers.join()
        self.render_viewers[client.client_id].thread_load_scenes.join()




        # 2. Delete specific queues/dictionaries
        while not self.render_viewers[client.client_id].scene_data.empty():
            item = self.render_viewers[client.client_id].scene_data.get()
            # If items are dictionaries containing tensors
            for key in list(item.keys()):
                if isinstance(item[key], torch.Tensor):
                    item[key] = item[key].detach().cpu()  # Move to CPU first
                    del item[key]
            del item
        self.render_viewers[client.client_id].img = self.render_viewers[client.client_id].img.detach().cpu()
        del self.render_viewers[client.client_id].img 
        # 3. Clear CUDA memory
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Wait for all GPU operations to finish
            torch.cuda.empty_cache()

       
    
        
        
        
        print("size of ", client.client_id, " scene data size", self.render_viewers[client.client_id].scene_data.qsize())
        self.render_viewers.pop(client.client_id)
        # gc.collect() 
        deep_cleanup()
        # show_live_cuda_tensors()
        show_live_cuda_tensors()

        # clear_gpu_memory(aggressive=True)
        # print_tensor_details()  



        print("size of render_viewers", len(self.render_viewers))
        # print("After disconnection", torch.cuda.memory_summary())  



    
    def handle_new_client(self, client:viser.ClientHandle):
        
            
        self.clients_num +=1 
       
        print("new client!", client.client_id)
        print("Total number of clients connected to the demo:", len(self.render_viewers))
        self.render_viewers[client.client_id] = RenderViewers(self, client)
        
        self.render_viewers[client.client_id].start()

   
   

   
    def _load_list_params_files(self, seq, exp):
        params_files =[ os.path.join(f"./output/{exp}/{seq}/", file) for file in  os.listdir(f"./output/{exp}/{seq}/") if file.startswith("params")]
        params_files = sorted(params_files, key= lambda x : int(os.path.basename(x).split("_")[1].split(".")[0]) if len(os.path.basename(x).split("_")) > 1 else os.path.basename(x).split("_")[0].split(".")[0]) 
        pc = np.load(os.path.join(f"./output/{exp}/{seq}/", "init_pt_cld.npz"))
        xyz = [vert[:3] for vert in pc['data']]
        xyz = np.asarray(xyz)
        center = np.mean(xyz[:], axis=0)
        print("Foreground center:",center)
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

        return params_files, total_length, center


    def start_viewer(self):

        while True:
            if not self.running:
                # time.sleep(1)
                break
   
            time.sleep(10)
            print("Total number of clients connected to the demo:", len(self.render_viewers))
           
    def signal_handler(self,sig, frame):

        print('You pressed Ctrl+C!')

        self.running = False
        for key, val in self.render_viewers.items():
            # val.video_audio_event.set()

            val.running = self.running
            val.thread_load_scenes.join()
            val.thread_cuda.join()
            # val.thread_encode.join()
            # val.thread_process_video_buffers.join()
            # val.thread_process_audio_buffers.join()

            
        
        viewer.viser_server.stop()

        sys.exit(0)
def deep_cleanup():
    # 1. Clear gradients and move model to CPU
    # if 'model' in globals():
    #     model.cpu()
    #     model.zero_grad(set_to_none=True)

    # # 2. Clear optimizer states
    # if 'optimizer' in globals():
    #     optimizer.state = {}
    #     optimizer.zero_grad(set_to_none=True)

    # 3. Clear CUDA cache
    torch.cuda.empty_cache()

    # 4. Delete any variables holding tensors
    for name, value in list(globals().items()):
        try:
            if torch.is_tensor(value):
                value.cpu()
                globals()[name] = None
            elif hasattr(value, 'state_dict'):
                globals()[name] = None
        except:
            pass

    # 5. Force garbage collection
    import gc
    gc.collect()

    # 6. Clear CUDA cache again
    torch.cuda.empty_cache()   
def show_live_cuda_tensors():
    if 'model' in globals():  
        globals()['model'].cpu()  
    torch.cuda.empty_cache()  

    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                print(f"Type: {type(obj)}, Size: {obj.size()}, Device: {obj.device}, Dtype: {obj.dtype}, RefCount: {sys.getrefcount(obj)}")
                # obj.detach_().cpu()
                # del obj
            elif hasattr(obj, 'data') and torch.is_tensor(obj.data) and obj.data.is_cuda:
                print(f"Data Type: {type(obj)}, Size: {obj.data.size()}, Device: {obj.data.device}, Dtype: {obj.data.dtype, }RefCount: {sys.getrefcount(obj)} ")
                # obj.detach_().cpu()
                # del obj
        except Exception as e:
            pass
    # gc.collect()  

    if torch.cuda.is_available():
        # torch.cuda.synchronize()
        # torch.cuda.empty_cache()
        print(f"Current GPU memory usage:")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB") 

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

def print_tensor_details():
    """Print details of all CUDA tensors still in memory"""
    print("\nRemaining CUDA tensors:")
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                print(f"Size: {obj.size()}, Device: {obj.device}, Dtype: {obj.dtype}, RefCount: {sys.getrefcount(obj)}")
        except:
            pass

class RenderViewers():
  

    # look_at = np.array([ 0.08500356,  0.42318333, -0.73812744 ]) #np.array([-0.28, 1.65, 0.09]) 
    roll_limit = (np.pi, -np.pi) 
    # roll_limit = (1.4, -1.0)
    pitch_limit = (2.4, -1.3)
    distance_in = 3 #2 
    distance_out = 20 #4.5

  
    

    def __init__(self, viewer, client ):
        self.viewer = viewer
        self.client = client
        self.encode_event = threading.Event()
        self.cuda_event = threading.Event()
        self.cuda_event_2 = threading.Event()

        self.load_event = threading.Event()
        self.thread_cuda = threading.Thread(target=self.render_images)    
        self.thread_encode = threading.Thread(target=self.encode_image)
        self.thread_load_scenes = threading.Thread(target=self.load_scenes)
        self.thread_process_audio_buffers = threading.Thread(target=self.process_audio_buffers)

        self.thread_process_video_buffers =  threading.Thread(target=self.process_video_buffers)
        self.running = True
        self.triggered = False
        self.scene_data = Queue()
        self.params_files = viewer.params_files
        self.num_timestamps = viewer.num_timesteps
        self.look_at = viewer.look_at
        self.gui_play_button  = client.gui.add_button(" Play", icon=viser.Icon.PLAYER_PLAY, color="red")
        self.gui_pause_button  = client.gui.add_button(" Pause", icon=viser.Icon.PLAYER_PAUSE, color="red" )
        self.gui_pause_button.disabled = True
        self.gui_play_button.disabled = False
        self.gui_play_button.on_click(self.handle_on_play_click)
        self.gui_pause_button.on_click(self.handle_on_pause_click)

        self.w = viewer.w
        self.h = viewer.h
        self.far = viewer.far   
        self.near = viewer.near
        self.k = viewer.k
        self.first_enter = True
        self.frame_number=0
        self.data_ready = False
        self.frame_rate = 30
        self.interval = 1.0/self.frame_rate
        # self.load_interval = 1.0/self.frame_rate/3
        self.isPaused = False
    
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
       

        self.thread_load_scenes.start()
        time.sleep(0.5)
        # self.thread_encode.start()
        time.sleep(0.5)

        self.thread_cuda.start()
       
        # self.thread_process_video_buffers.start()

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
            
            try:
                size = self.img.numel() * self.img.element_size()
                
                try:
                    
                    gst_h264_endoder_pipeline.push_tensor_frame(self.img.data_ptr(), size, self.frame_number, self.frame_rate, self.client_cnt)
                    
                    self.frame_number += 1


                except Exception as e:
                    print(f"Error encoding {self.img_num}: {e}")
            except  Exception as e:
                # print(f" Image is not encoding {self.img_num}: {e}")
                pass
              
            self.cuda_event.set()
        self.cuda_event.set()    

        gst_h264_endoder_pipeline.close_pipeline(self.client_cnt)
      
    def load_scenes(self):
        seg_as_col=False
        # num_timestamps = len(self.num_timestamps)
        window = [0, 2]
        cnt = 0
        while self.running:

            if window[1] > len(self.params_files) and window[0] < len(self.params_files):
               window[1] = len(self.params_files) 

            for params_file in self.params_files[window[0]: window[1]]:
                params = np.load(params_file, allow_pickle=True)
                print(f"{params_file} loaded!")
    

                for param in params:
                    if param == "allow_pickle":
                        continue
                    data = params[param]
                    
                    for pr in data:
                        for  p in pr:
                            with torch.no_grad():
                            # with torch.inference_mode():

                                rendervar = {
                                    'means3D': torch.tensor(p['means3D']).cuda().float(),
                                    'colors_precomp':  torch.tensor(p['rgb_colors']).cuda().float() if not seg_as_col else torch.tensor(p['seg_colors']).cuda().float(),
                                    'rotations': torch.nn.functional.normalize( torch.tensor(p['unnorm_rotations']).cuda().float()),
                                    'opacities': torch.sigmoid(torch.tensor(pr[0]['logit_opacities']).cuda().float()),
                                    'scales': torch.exp( torch.tensor(pr[0]['log_scales']).cuda().float()),
                                    'means2D': torch.zeros_like( torch.tensor(pr[0]['means3D']).cuda().float())
                                }

                                self.scene_data.put(rendervar)
                                if (cnt+1) % 30 == 0:
                                    torch.cuda.synchronize()  
                                    torch.cuda.empty_cache() 
                                    print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB")  
                                    print(f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f}GB")   

                                cnt+=1
                            # print("queue size", self.scene_data.qsize())

            # self.cuda_event_2.set()
            self.load_event.wait()
            
            if window[1] == len(self.params_files):
                window = [0, 2]

            window[0] += 2
            window[1] += 2
            
            self.load_event.clear()

        # # 2. Delete specific queues/dictionaries
        # while not self.scene_data.empty():
        #     item = self.scene_data.get()
        #     # If items are dictionaries containing tensors
        #     for key in list(item.keys()):
        #         if isinstance(item[key], torch.Tensor):
        #             item[key] = item[key].detach().cpu()  # Move to CPU first
        #             del item[key]
        #     del item
        
        # # 3. Clear CUDA memory
        # gc.collect()

        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()  # Wait for all GPU operations to finish
        #     torch.cuda.empty_cache()
       
        


    # @classmethod
    # def deep_clear_cuda(self, item):
    #     if isinstance(item, dict):
    #         for key, value in item.items():
    #             if hasattr(value, 'is_cuda') and value.is_cuda:
    #                 # Move to CPU first (reduces GPU memory immediately)
    #                 value = value.cpu()
    #                 # Clear from dictionary
    #                 item[key] = None
    #                 # Delete the CPU tensor
    #                 print("refs:", sys.getrefcount(value))
    #                 # print("refs")

    #                 del value
    #             elif isinstance(value, (dict, list)):
    #                 self.deep_clear_cuda(value)
    #         # Clear the dictionary itself
    #         item.clear()
    #     elif isinstance(item, list):
    #         for i, value in enumerate(item):
    #             if hasattr(value, 'is_cuda') and value.is_cuda:
    #                 value = value.cpu()
    #                 item[i] = None
    #                 del value
    #             elif isinstance(value, (dict, list)):
    #                 self.deep_clear_cuda(value)
    #         item.clear()

    #     # Force cleanup
    #     gc.collect()
    #     torch.cuda.empty_cache()
    # # For nested structures
    @classmethod
    def deep_clear_cuda(self,item):
        if isinstance(item, dict):
            for key, value in item.items():
                if hasattr(value, 'is_cuda') and value.is_cuda:
                    # value.detach().cpu()  # Detach from computation graph
                    # print("Deleting ", key)
                    # print("refs:", sys.getrefcount(value))
                    value=None
                    del value
                    # gc.collect()  

                    # torch.cuda.empty_cache()
                elif isinstance(value, (dict, list)):
                    self.deep_clear_cuda(value)
            item.clear()
        elif isinstance(item, list):
            for value in item:
                self.deep_clear_cuda(value)
            item.clear()
        gc.collect()
        torch.cuda.empty_cache()
        # torch.cuda.ipc_collect()
        # torch.cuda.reset_peak_memory_stats()
        # For more aggressive clearing:
        # torch.cuda.reset_max_memory_allocated()
    @classmethod
    def print_refs(value, msg=""):  
        print(f"{msg} Reference count: {sys.getrefcount(value)}")  
        print(f"Object id: {id(value)}")         
    @classmethod
    def clear_gpu_dict(self, gpu_dict):
        # Iterate through all values and delete them
        keys = list(gpu_dict.keys())  

        # Then iterate over the keys list  
        for key in keys:  
            if torch.is_tensor(gpu_dict[key]) and gpu_dict[key].is_cuda: 
                it = gpu_dict[key].to(device=torch.device("cpu"))

                # print("Deleting ", key)
                del it  

        gpu_dict.clear()  
        torch.cuda.empty_cache() 
        # torch.cuda.reset_device()
       

    def render_images(self):
        from diff_gaussian_rasterization import GaussianRasterizer as Renderer
        def render(w2c, timestep_data, bg=[0,0,0]):
        # print(self.w, self.h, self.k, self.near, self.far)
            with torch.inference_mode():
            # with torch.no_grad():
                cam = setup_camera(self.w, self.h, self.k, w2c, self.near, self.far, bg=torch.tensor(bg)) #[0, 177./255, 64.0/255]
                im, radi, _ = Renderer(raster_settings=cam)(**timestep_data)
                # radi =radi.detach().cpu()
                # del radi
                # ten = _.detach().cpu()
                # del ten
                # torch.cuda.empty_cache()
                # im[~is_fg] =  torch.tensor([0, 177./255, 64.0/255], dtype=torch.float32, device="cuda")
                # torchvision.utils.save_image(im, '{0:05d}'.format(cnt) + ".png")
                # im = torch.flip(im, dims=[0])
                # im =  im.permute(1,2,0).contiguous()
                # im = (im.clamp(0,1)*255).to(torch.uint8)
                im_c =  im.permute(1,2,0).contiguous()
                im = im.detach_().cpu()
                del im
                im_c.clamp_(0,1)
                im_c.mul_(255)
                im_int= im_c.to(torch.uint8)
                im_c = im_c.detach_().cpu()
                del im_c
          
                return im_int
            # num_timestamps = len(self.num_timestamps)

        t0 = time.time() + self.interval
        c2w = np.eye(4)
        w2c = np.eye(4)

        cnt =0 
        
        
        num_wait = 0 
        # self.cuda_event_2.wait()
        # self.cuda_event_2.clear()     
        while self.running:
            

            # if not self.running:
            #     break
            if  self.first_enter: 
                self.client.camera.wxyz = self.init_camera()[0]
                self.client.camera.position = self.init_camera()[1] 
                self.client.camera.look_at= self.look_at 
                scene_data = self.scene_data.get()
                # for k, v in scene_data.items():  
                #     if torch.is_tensor(v):  
                #         scene_data[k] = v.cuda()


                self.first_enter = False

            R_S03 = tf.SO3(np.asarray(self.client.camera.wxyz))
            R = R_S03.as_matrix()
            T = self.client.camera.position

            vec_to_look_at = self.look_at - T
            theta, phi = self.get_theta_phi_angles_from_cam_pos(vec_to_look_at)
            # print("Phi in degrees:", theta, " ",  np.linalg.norm(vec_to_look_at))
            if (theta > 40 and theta < 110)  and  \
                np.linalg.norm(vec_to_look_at)  > self.distance_in and \
                np.linalg.norm(vec_to_look_at) < self.distance_out:

            # if theta > 60 and theta < 135 and  np.linalg.norm(self.look_at - T)  > self.distance_in and  np.linalg.norm(self.look_at - T) < self.distance_out:

                c2w = np.vstack((np.concatenate((R, T[:,None]), axis=1),[0,0,0,1]))
                w2c = np.linalg.inv(c2w)
                # print(w2c,c2w)
            else:
                        
                c2w = np.linalg.inv(w2c)

                self.client.camera.position = c2w[:3,3] 
                self.client.camera.wxyz = tf.SO3.from_matrix(c2w[:3,:3]).wxyz
                self.client.camera.look_at = self.look_at 
            # print(f"current ts:{current_ts}")
            # if not self.isPaused :
           
            
            self.img = render(w2c, scene_data, bg=[0, 177.0/255, 64.0/255])
            # for key in list(scene_data.keys()):
            #     if isinstance(scene_data[key], torch.Tensor):
            #         scene_data[key] = scene_data[key].detach().cpu()  # Move to CPU first
            #         del scene_data[key]
            # del scene_data

            # torch.cuda.synchronize()  
            # torch.cuda.empty_cache() 
            # print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB")  
            # print(f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f}GB")   

            
            
           
            
            self.encode_event.set()
            # if  self.isPaused:
            #     self.clear_gpu_dict(scene_data)

            if not self.isPaused:
                scene_data = self.scene_data.get()
                # for k, v in scene_data.items():  
                #     if torch.is_tensor(v):  
                #         scene_data[k] = v.cuda()
            # self.cuda_event.wait()
            
            
            delta = t0 - time.time()
            if delta > 0:
                time.sleep(delta)
            if self.scene_data.qsize() < 10: # num_wait == 70:
                self.load_event.set()

            t0 = time.time() + self.interval
        
            # self.cuda_event.clear()

            # if ( cnt+1 ) % 30 ==0:
            #     torch.cuda.empty_cache()
            # cnt +=1

        self.load_event.set()
        self.encode_event.set()
        

     
        # self.img = self.img.detach().cpu()
        # del self.img
        # # 3. Clear CUDA memory
        # gc.collect()

        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()  # Wait for all GPU operations to finish
        #     torch.cuda.empty_cache()
       
        



   
                



    def _render(self, w2c, timestep_data, bg=[0,0,0]):
        # print(self.w, self.h, self.k, self.near, self.far)
        with torch.no_grad():
            cam = setup_camera(self.w, self.h, self.k, w2c, self.near, self.far, bg=torch.tensor(bg)) #[0, 177./255, 64.0/255]
            im, radi, _ = Renderer(raster_settings=cam)(**timestep_data)
            # radi = radi.to("cpu")
            # depth= depth.to("cpu")
            # torch.cuda.empty_cache()  

            # radi =radi.detach().cpu()
            # del radi
            # ten = _.detach().cpu()
            # del ten
            # torch.cuda.empty_cache()
            # im[~is_fg] =  torch.tensor([0, 177./255, 64.0/255], dtype=torch.float32, device="cuda")
            # torchvision.utils.save_image(im, '{0:05d}'.format(cnt) + ".png")
            # im = torch.flip(im, dims=[0])
            im_c =  im.permute(1,2,0).contiguous()
            im_c.clamp_(0,1)
            im_c.mul_(255)
            im_int= im_c.to(torch.uint8)
          
          
         

            return im_int
    
    
    @classmethod
    def init_camera(cls, y_angle=0., center_dist=4., cam_height= 3., f_ratio=0.82):
        ry = y_angle * np.pi / 180
        # w2c = np.array([[np.cos(ry), 0., -np.sin(ry), -0.0],
        #                 [0.,         1., 0.,          cam_height],
        #                 [np.sin(ry), 0., np.cos(ry),  center_dist],
        #                 [0.,         0., 0.,          1.]])
        
        c2w = np.array([[-0.94743326, -0.0982282 , -0.30450196,  2.2777 ],
                [-0.09894314,  0.99500657, -0.01312203 , 0.185],
                [ 0.3042704 ,  0.01769613, -0.95242132,  6.5],
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


        return theta_degrees, phi_degrees

class MemoryTracker:
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def get_tensor_info(self):
        total_size = 0
        tensor_info = []

        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) and obj.device == self.device:
                    # Get tensor size in MB
                    size = obj.element_size() * obj.nelement() / 1024**2
                    total_size += size

                    # Get allocation site (where the tensor was created)
                    allocation_site = ''
                    for frame in inspect.stack():
                        if 'ipython' not in frame.filename.lower():
                            allocation_site = f"{frame.filename}:{frame.lineno}"
                            break

                    tensor_info.append({
                        'shape': tuple(obj.shape),
                        'dtype': str(obj.dtype),
                        'size_mb': f"{size:.2f}MB",
                        'device': str(obj.device),
                        'allocation_site': allocation_site
                    })
            except:
                pass

        return tensor_info, total_size

    def print_tensor_info(self):
        tensor_info, total_size = self.get_tensor_info()

        print("\n=== GPU Tensor Memory Usage ===")
        for idx, info in enumerate(tensor_info, 1):
            print(f"\nTensor {idx}:")
            print(f"Shape: {info['shape']}")
            print(f"Dtype: {info['dtype']}")
            print(f"Size: {info['size_mb']}")
            print(f"Device: {info['device']}")
            print(f"Allocation site: {info['allocation_site']}")

        print(f"\nTotal GPU memory used by tensors: {total_size:.2f}MB")

# Example usage
# tracker = MemoryTracker()
    
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

    # exp_name = "hamit_2024-12-04_17-14-42_scl_2_it_600_test1"
    # sequence = "2024-12-04_17-14-42"

    # exp_name = "hamit_2024-12-19_19-12-14_4096_wo_bckgrnd_scl_2_it_700"
    # sequence = "2024-12-19_19-12-14_4096_wo_bckgrnd"
    
    # exp_name = "2025-02-05_14-06-52_gain_9_scl_1_it_700_2"
    # sequence = "2025-02-05_14-06-52_gain_9"
    
    # exp_name = "2025-02-05_14-16-46_gain_9_scl_1_it_700_2"
    # exp_name = "2025-02-05_14-16-46_gain_9_scl_1_it_700_test_start_0-1000"
    # sequence = "2025-02-05_14-16-46_gain_9"

    # exp_name = "2025-03-27_15-42-48_yoga_scl_1_1000"
    # sequence = "2025-03-27_15-42-48_yoga"

    # exp_name = "2025-03-27_15-42-48_yoga_ai_scl_1_1000_test2"
    # sequence = "2025-03-27_15-42-48_yoga_ai"

    # exp_name = "2025-03-27_15-44-28_yoga_scl_1_1000"
    # sequence = "2025-03-27_15-44-28_yoga"

    # exp_name = "2025-03-27_15-46-48_yoga_ai_new_scl_1_all_green"
    # sequence = "2025-03-27_15-46-48_yoga_ai_new"

    # exp_name = "2025-03-27_15-46-48_yoga_ai_adjusted_scl_1_test"
    # sequence = "2025-03-27_15-46-48_yoga_ai_adjusted"

    # exp_name = "2025-03-27_15-42-48_yoga_ai_stest_cl_1_noseg_2"
    # sequence = "2025-03-27_15-46-48_yoga_ai_test_noseg"

    # exp_name = "2025-03-27_15-46-48_yoga_ai_test_scl_1"        
    # sequence =  "2025-03-27_15-46-48_yoga_ai_test"

    # exp_name = "2025-03-27_15-46-48_yoga_ai_test_nogreen_scl_1"        
    # sequence =  "2025-03-27_15-46-48_yoga_ai_test_nogreen"

    # exp_name = "2025-03-27_15-46-48_yoga_ai_final_scl_1"        
    # sequence =  "2025-03-27_15-46-48_yoga_ai_final"
    
    # exp_name = "2025-03-27_15-46-48_yoga_ai_final_scl_1_intrv_93_150"        
    # sequence =  "2025-03-27_15-46-48_yoga_ai_final"

    # exp_name = "2025-03-27_15-46-48_yoga_ai_final_scl_1_intrv_154"        
    # sequence =  "2025-03-27_15-46-48_yoga_ai_final"

    # exp_name = "2025-03-27_15-44-28_yoga_ai_test_scl_1"
    # sequence = "2025-03-27_15-44-28_yoga_ai_test"


    # exp_name = "2025-03-27_15-46-48_yoga_ai_final_scl_1_intrv_394"
    # sequence =  "2025-03-27_15-46-48_yoga_ai_final"

    # exp_name = "2025-03-27_15-46-48_yoga_ai_final_scl_1_intrv_0_1000"
    # sequence = "2025-03-27_15-46-48_yoga_ai_final"
    exp_name = "samples"
    sequence = "sample_1"


    # torch.cuda.set_per_process_memory_fraction(0.1)
        
    viewer = Viewer(seq=sequence, exp=exp_name,w=1920, h=1080)
    time.sleep(0.2)
    viewer.start_viewer()
   

