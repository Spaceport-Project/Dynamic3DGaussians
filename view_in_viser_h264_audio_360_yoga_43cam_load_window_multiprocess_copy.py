
import os
import random
import threading
from typing import Dict
import torch
import numpy as np
import time
# from diff_gaussian_rasterization import GaussianRasterizer as Renderer
import torchvision
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
from queue import Queue, Empty  
import torch.multiprocessing as torch_mp  

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






  
        
class Viewer():

    def __init__(self, seq, exp, f_ratio=0.8, w=1920, h=1080, near=0.01, far=100.0):
        self.seq = seq
        self.exp = exp
        self.viser_server = viser.ViserServer(port=8089)
        self.viser_server.scene.world_axes.visible = False
        self.clients_num = 0
        self.k = np.array([[f_ratio * w, 0, w / 2], [0, f_ratio * w, h / 2], [0, 0, 1]])
        self.w = w
        self.h = h
        self.near = near
        self.far = far
        self.params_files, self.num_timesteps, self.look_at = self._load_list_params_files(self.seq, self.exp)

        # self.scene_data, _ , self.look_at = self._load_scene_data3(self.seq, self.exp, seg_as_col=False)
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
        viewer.cleanup()
       

        # Remove from dictionary
        del self.render_viewers[client.client_id]

        # Force garbage collection
        gc.collect()

        print("Total number of clients:", len(self.render_viewers))



    
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
            # print(f"{params_file} loaded!")
            # if l > 0:
            #     break

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
            time.sleep(360)
            print("Total number of clients connected to the demo:", len(self.render_viewers))
           
    def signal_handler(self,sig, frame):

        print('You pressed Ctrl+C!')

        self.running = False
        for key, val in self.render_viewers.items():
            # val.video_audio_event.set()

            val.running.value = self.running
            val.thread_load_scenes.join()
            val.thread_cuda.join()
            val.thread_encode.join()
            val.thread_process_video_buffers.join()
            val.render_process.terminate()
            val.render_process.join()
            # val.thread_process_audio_buffers.join()

            
        
        viewer.viser_server.stop()

        sys.exit(0)
    
   


class RenderViewers():
  

    # look_at = np.array([ 0.08500356,  0.42318333, -0.73812744 ]) #np.array([-0.28, 1.65, 0.09]) 
    roll_limit = (np.pi, -np.pi) 
    # roll_limit = (1.4, -1.0)
    pitch_limit = (2.4, -1.3)
    distance_in = 3 #2 
    distance_out = 20 #4.5

  
    

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
       
     # Create multiprocessing context
        torch_mp = torch_mp.get_context('spawn')

        # Create queues using the multiprocessing context
        self.render_input_queue = torch_mp.Queue(maxsize=1)
        self.render_output_queue = torch_mp.Queue(maxsize=1)
        self.render_event = torch_mp.Event()  # Create the event  
        self.render_ready_event = torch_mp.Event()  # Create the event  
        self.running = torch_mp.Value('b', True) 
        # Create render process
        self.render_process = torch_mp.Process(
            target=RenderViewers.render_process_function,
            args=(
                self.render_input_queue,
                self.render_output_queue,
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

        self.render_process.daemon = True  




       
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

            self.thread_process_video_buffers.start()

          


            

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
            if self.thread_process_video_buffers.is_alive():
                self.thread_process_video_buffers.join()
            raise




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
       
       


        while self.running.value :

                t0 = time.time() + self.interval
                start_time = 0
                for i, timestamp in enumerate(image_timestamps):
                  

                    if not self.running.value:
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
            while self.running.value:
               
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
        while self.running.value:
           
            self.encode_event.wait()
            self.encode_event.clear()
            
            if len(self.img) > 0:
                size = self.img.numel() * self.img.element_size()
                try:
                   
                    gst_h264_endoder_pipeline.push_tensor_frame(self.img.data_ptr(), size, self.frame_number, self.frame_rate, self.client_cnt)
                    self.frame_number += 1


                except Exception as e:
                    print(f"Error encoding {self.img_num}: {e}")
              
            self.cuda_event.set()
        self.cuda_event.set()    

        gst_h264_endoder_pipeline.close_pipeline(self.client_cnt)
      
    def load_scenes(self):
        seg_as_col=False
        # num_timestamps = len(self.num_timestamps)
        window = [0, 1]
        while self.running.value:

            # if window[1] > len(self.params_files) and window[0] < len(self.params_files):
            #    window[1] = len(self.params_files) 
            for params_file in self.params_files:

            # for params_file in self.params_files[window[0]: window[1]]:
                params = np.load(params_file, allow_pickle=True)
                print(f"{params_file} loaded!")
    

                for param in params:
                    if param == "allow_pickle":
                        continue
                    data = params[param]
                    
                    for pr in data:
                        for  p in pr:
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

                                self.scene_data.put(rendervar)
                                if self.scene_data.qsize() > 20 :
                                    self.load_event.wait()
                                    self.load_event.clear()
                                # print("queue size", self.scene_data.qsize())

           
            
        self.load_event.clear()

    

    
    def render_images(self):
        # t0 = time.time() + self.interval

        while self.running.value:
            t0 = time.time()
            if self.first_enter:
                self.client.camera.wxyz = self.init_camera()[0]
                self.client.camera.position = self.init_camera()[1]
                self.client.camera.look_at = self.look_at
                # scene_data = self.scene_data.get()
                self.first_enter = False

            # Calculate camera matrices
            R_S03 = tf.SO3(np.asarray(self.client.camera.wxyz))
            R = R_S03.as_matrix()
            T = self.client.camera.position

            vec_to_look_at = self.look_at - T
            theta, phi = self.get_theta_phi_angles_from_cam_pos(vec_to_look_at)

            if (theta > 40 and theta < 110) and \
            np.linalg.norm(vec_to_look_at) > self.distance_in and \
            np.linalg.norm(vec_to_look_at) < self.distance_out:

                c2w = np.vstack((np.concatenate((R, T[:,None]), axis=1),[0,0,0,1]))
                w2c = np.linalg.inv(c2w)
            else:
                try:
                    c2w = np.linalg.inv(w2c)
                    self.client.camera.position = c2w[:3,3]
                    self.client.camera.wxyz = tf.SO3.from_matrix(c2w[:3,:3]).wxyz
                    self.client.camera.look_at = self.look_at
                except Exception as e:
                    continue

                

            if not self.isPaused:
                scene_data = self.scene_data.get()

            
            # Send data to render process
            
            try:
                self.render_input_queue.put((w2c, scene_data))
                
            except Exception as e:
                # If queue is full, skip this frame
                print(f"Error putting data in render queue: {e}")  

                continue
            
            self.render_event.set() 
            # Wait for rendering to complete  
            self.render_ready_event.wait()  
            self.render_ready_event.clear()  
            
            try:
                self.img = self.render_output_queue.get()
                # img_num = self.img.numpy()

                # image = Image.fromarray(img_num)
                # image.save("img_trial.png")

                self.encode_event.set()
                
            except Empty:
                print("Render timeout - frame dropped")
                raise
        

            self.cuda_event.wait()

            elapsed = time.time() - t0
            sleep_time = self.interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
           
            if self.scene_data.qsize() < 10:
                self.load_event.set()

            self.cuda_event.clear()
        
        self.render_event.set()
        self.load_event.set()
        self.encode_event.set()
        

    
    def cleanup(self):
        """Thorough cleanup when a client disconnects"""
        self.running.value = False

        # Set all events to unblock threads
        self.encode_event.set()
        self.cuda_event.set()
        self.load_event.set()
        self.render_event.set()
        self.render_ready_event.set()

        # Wait for threads to finish
        if hasattr(self, 'thread_load_scenes') and self.thread_load_scenes.is_alive():
            self.thread_load_scenes.join(timeout=1)

        if hasattr(self, 'thread_cuda') and self.thread_cuda.is_alive():
            self.thread_cuda.join(timeout=1)

        if hasattr(self, 'thread_encode') and self.thread_encode.is_alive():
            self.thread_encode.join(timeout=1)

        if hasattr(self, 'thread_process_video_buffers') and self.thread_process_video_buffers.is_alive():
            self.thread_process_video_buffers.join(timeout=1)

        # Terminate render process
        if hasattr(self, 'render_process') and self.render_process.is_alive():
            print("terminating")
            self.render_process.terminate()
            self.render_process.join(timeout=1)
        
        # self.deep_clear_cuda(self.scene_data)

        try:
            # Clear any remaining tensors
            while not self.render_input_queue.empty():
                try:
                    _, scene_data = self.render_input_queue.get_nowait()
                    for tensor in scene_data.values():
                        if torch.is_tensor(tensor) and tensor.is_cuda:
                            ten = tensor.cpu()
                            print("deleting inputs")
                            del ten
                        del tensor
                except Empty:
                    break
            self.render_input_queue.close()
        
            while not self.render_output_queue.empty():
                try:
                    tensor = self.render_output_queue.get_nowait()
                    if torch.is_tensor(tensor) and tensor.is_cuda:
                        # tensor.detach_()
                        print("deleting outputs")

                        ten = tensor.cpu()
                        del ten
                    del tensor
                except Empty:
                    break
            self.render_output_queue.close()


        except Exception as e:
            print(f"Error in render process cleanup: {e}")


        
        # Clear CUDA tensors
        if hasattr(self, 'img'):
            img = self.img.cpu()
            del img

        # Clear multiprocessing context

        # Clear all instance attributes
        for attr in list(self.__dict__.keys()):
            delattr(self, attr)

        # Force garbage collection
        # gc.collect()

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()


       
        self.show_live_cuda_tensors()


   
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
    def render_process_function(input_queue, output_queue, w, h, k, near, far, render_event, render_ready_event, running):
        from diff_gaussian_rasterization import GaussianRasterizer as Renderer
        

        print("Render process started")
        cnt = 0
        while running.value:
            try:
                # print(running.value)
                render_event.wait()
                render_event.clear()
                w2c, scene_data = input_queue.get()
                for key, val in scene_data.items():
                    scene_data[key] = val.cuda()
                # scene_data = {key: torch.tensor(val).cuda().float() for key, val in scene_data.items()}
                with torch.no_grad():
                    cam = setup_camera(w, h, k, w2c, near, far,
                                    bg=torch.tensor([0, 177.0/255, 64.0/255], device='cuda'))
                    im, radi, _ = Renderer(raster_settings=cam)(**scene_data)

                    im = im.permute(1,2,0).contiguous()
                    im = (im.clamp(0,1)*255).to(torch.uint8)
               
                    output_queue.put(im)
                    render_ready_event.set()

                # Periodic cleanup
                if cnt % 50 == 0:
                    torch.cuda.empty_cache()
                cnt += 1

            except Exception as e:
                print(f"Error in render process: {e}")
                render_ready_event.set()
                continue
        im = im.cpu()
        
        del im

        while not input_queue.empty():
            try:
                _, scene_data = input_queue.get_nowait()
                for tensor in scene_data.values():
                    if torch.is_tensor(tensor) and tensor.is_cuda:
                        ten = tensor.cpu()
                        print("deleting inputs")
                        del ten
                    del tensor
            except Empty:
                break
        input_queue.close()
        
        while not output_queue.empty():
            try:
                tensor = output_queue.get_nowait()
                if torch.is_tensor(tensor) and tensor.is_cuda:
                    # tensor.detach_()
                    print("deleting outputs")

                    ten = tensor.cpu()
                    del ten

                del tensor
            except Empty:
                break
        output_queue.close()

        print("deleted")
        
        
    

    
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


        return im
    
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


    exp_name = "2025-08-06_3412x2500_combin1_all_multiprocess"
    sequence = "2025-08-06_3412x2500_combin1"


        
    viewer = Viewer(seq=sequence, exp=exp_name,w=1920, h=1080)
    time.sleep(0.2)
    viewer.start_viewer()

