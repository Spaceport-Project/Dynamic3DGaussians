import os
import random
import threading
from typing import Dict
import torch
import numpy as np
import time
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

# Initialize GStreamer
libc = CDLL("libc.so.6")
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')
gi.require_version('GstVideo', '1.0')
from gi.repository import Gst, GstApp, GLib, GObject
Gst.init(None)

# Global settings
REMOVE_BACKGROUND = False
w, h = 1920, 1080
near, far = 0.01, 100.0

class RenderContext:
    def __init__(self):
        torch.cuda.set_device(0)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception as e:
            print(f"Error cleaning up render context: {e}")

def safe_tensor_cleanup(tensor):
    """Safely cleanup a CUDA tensor"""
    try:
        if torch.is_tensor(tensor) and tensor.is_cuda:
            tensor_cpu = tensor.detach().cpu()
            del tensor
            del tensor_cpu
    except Exception as e:
        print(f"Error cleaning up tensor: {e}")
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
        self.render_viewers: Dict[int, RenderViewers] = {}
        signal.signal(signal.SIGINT, self.signal_handler)
        self.viser_server.on_client_connect(self.handle_new_client)
        self.viser_server.on_client_disconnect(self.handle_disconnect_client)
        self.running = True

    def _load_list_params_files(self, seq, exp):
        params_files = [os.path.join(f"./output/{exp}/{seq}/", file)
                       for file in os.listdir(f"./output/{exp}/{seq}/")
                       if file.startswith("params")]
        params_files = sorted(params_files,
                            key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0])
                            if len(os.path.basename(x).split("_")) > 1
                            else os.path.basename(x).split("_")[0].split(".")[0])

        pc = np.load(os.path.join(f"./output/{exp}/{seq}/", "init_pt_cld.npz"))
        xyz = [vert[:3] for vert in pc['data']]
        xyz = np.asarray(xyz)
        center = np.mean(xyz[:], axis=0)
        print("Foreground center:", center)

        total_length = 0
        for l, params_file in enumerate(params_files):
            params = np.load(params_file, allow_pickle=True)
            print(f"{params_file} loaded!")

            for param in params:
                if param == "allow_pickle":
                    continue
                data = params[param]
                for pr in data:
                    total_length += len(pr)

        return params_files, total_length, center

    def handle_disconnect_client(self, client: viser.ClientHandle):
        print(f"Client {client.client_id} disconnecting...")
        try:
            if client.client_id in self.render_viewers:
                viewer = self.render_viewers[client.client_id]
                viewer.cleanup()
                del self.render_viewers[client.client_id]

            gc.collect()
            torch.cuda.empty_cache()
            print(f"Client {client.client_id} disconnected successfully")
            print(f"Current number of clients: {len(self.render_viewers)}")
        except Exception as e:
            print(f"Error during client disconnect cleanup: {e}")

    def handle_new_client(self, client: viser.ClientHandle):
        self.clients_num += 1
        print("New client!", client.client_id)
        print("Total number of clients connected to the demo:", len(self.render_viewers))
        self.render_viewers[client.client_id] = RenderViewers(self, client)
        self.render_viewers[client.client_id].start()

    def start_viewer(self):
        while True:
            if not self.running:
                break
            time.sleep(60)
            print("Total number of clients connected to the demo:", len(self.render_viewers))

    def cleanup(self):
        """Cleanup all viewers and resources"""
        print("Cleaning up all viewers...")
        for client_id in list(self.render_viewers.keys()):
            try:
                viewer = self.render_viewers[client_id]
                viewer.cleanup()
                del self.render_viewers[client_id]
            except Exception as e:
                print(f"Error cleaning up viewer {client_id}: {e}")

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    def signal_handler(self, sig, frame):
        print('Shutting down server...')
        self.running = False

        # Cleanup all viewers
        self.cleanup()

        # Stop the server
        self.viser_server.stop()

        sys.exit(0)
class RenderViewers():
    roll_limit = (np.pi, -np.pi)
    pitch_limit = (2.4, -1.3)
    distance_in = 3
    distance_out = 20

    def __init__(self, viewer, client):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        self.viewer = viewer
        self.client = client
        self.encode_event = threading.Event()
        self.cuda_event = threading.Event()
        self.cuda_event_2 = threading.Event()
        self.load_event = threading.Event()
        self.thread_cuda = threading.Thread(target=self.render_images)
        self.thread_encode = threading.Thread(target=self.encode_image)
        self.thread_load_scenes = threading.Thread(target=self.load_scenes)
        self.thread_process_video_buffers = threading.Thread(target=self.process_video_buffers)
        torch_mp = torch_mp.get_context('spawn')
        self.render_input_queue = torch_mp.Queue(maxsize=2)
        self.render_output_queue = torch_mp.Queue(maxsize=2)
        self.render_event = torch_mp.Event()
        self.render_ready_event = torch_mp.Event()
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
                self.render_event,
                self.render_ready_event
            )
        )
        self.render_process.daemon = True
        self.running = True
        self.triggered = False
        self.scene_data = Queue()
        self.params_files = viewer.params_files
        self.num_timestamps = viewer.num_timesteps
        self.look_at = viewer.look_at
        self.gui_play_button = client.gui.add_button(" Play", icon=viser.Icon.PLAYER_PLAY, color="red")
        self.gui_pause_button = client.gui.add_button(" Pause", icon=viser.Icon.PLAYER_PAUSE, color="red")
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
        self.frame_number = 0
        self.data_ready = False
        self.frame_rate = 30
        self.interval = 1.0 / self.frame_rate
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
        try:
            self.render_process.start()
            print(f"Render process started with PID: {self.render_process.pid}")
            self.thread_load_scenes.start()
            self.thread_encode.start()
            self.thread_cuda.start()
            print("Starting render process...")
            self.thread_process_video_buffers.start()
        except Exception as e:
            print(f"Error starting components: {e}")
            self.running = False
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

    def cleanup(self):
        print(f"Starting cleanup for client {self.client.client_id}")
        self.running = False
        # Set all events first
        for event_attr in ['encode_event', 'cuda_event', 'load_event', 'render_event', 'render_ready_event']:
            if hasattr(self, event_attr):
                event = getattr(self, event_attr)
                event.set()
        # Stop render process first
        if hasattr(self, 'render_process') and self.render_process.is_alive():
            try:
                self.render_process.terminate()
                self.render_process.join(timeout=3)
                if self.render_process.is_alive():
                    self.render_process.kill()
            except Exception as e:
                print(f"Error terminating render process: {e}")
        # Clear queues
        try:
            while not self.render_input_queue.empty():
                try:
                    _, scene_data = self.render_input_queue.get_nowait()
                    self.clear_gpu_dict(scene_data)
                except Empty:
                    break
            while not self.render_output_queue.empty():
                try:
                    tensor = self.render_output_queue.get_nowait()
                    safe_tensor_cleanup(tensor)
                except Empty:
                    break
            self.render_input_queue.close()
            self.render_output_queue.close()
        except Exception as e:
            print(f"Error clearing queues: {e}")
        # Join threads
        threads = [
            ('thread_load_scenes', self.thread_load_scenes),
            ('thread_cuda', self.thread_cuda),
            ('thread_encode', self.thread_encode),
            ('thread_process_video_buffers', self.thread_process_video_buffers)
        ]
        for thread_name, thread in threads:
            if hasattr(self, thread_name) and thread.is_alive():
                try:
                    thread.join(timeout=2)
                except Exception as e:
                    print(f"Error joining {thread_name}: {e}")
        # Clear scene data
        try:
            while hasattr(self, 'scene_data') and not self.scene_data.empty():
                item = self.scene_data.get_nowait()
                self.clear_gpu_dict(item)
        except Exception as e:
            print(f"Error clearing scene data: {e}")
        # Final CUDA cleanup
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception as e:
            print(f"Error in final CUDA cleanup: {e}")
        print(f"Cleanup completed for client {self.client.client_id}")

    @staticmethod
    def clear_gpu_dict(gpu_dict):
        keys = list(gpu_dict.keys())
        for key in keys:
            if torch.is_tensor(gpu_dict[key]) and gpu_dict[key].is_cuda:
                it = gpu_dict[key].to(device=torch.device("cpu"))
                del it
        gpu_dict.clear()
        torch.cuda.empty_cache()

    def share_scene_data(self, scene_data):
        shared_data = {}
        try:
            for k, v in scene_data.items():
                if torch.is_tensor(v) and v.is_cuda:
                    shared_tensor = v.clone().detach()
                    shared_tensor.share_memory_()
                    shared_data[k] = shared_tensor
                else:
                    shared_data[k] = v
        except Exception as e:
            print(f"Error in sharing scene data: {e}")
            self.clear_gpu_dict(shared_data)
            raise
        return shared_data

    @staticmethod
    def render_process_function(input_queue, output_queue, w, h, k, near, far, render_event, render_ready_event):
        from diff_gaussian_rasterization import GaussianRasterizer as Renderer
        import signal

        def safe_cleanup_tensors(tensors_dict):
            if not isinstance(tensors_dict, dict):
                return
            for key, tensor in tensors_dict.items():
                try:
                    if torch.is_tensor(tensor) and tensor.is_cuda:
                        tensor.detach_()
                        del tensor
                except Exception as e:
                    print(f"Error cleaning up tensor {key}: {e}")

        def cleanup_render_process():
            print("Cleaning up render process")
            try:
                while not input_queue.empty():
                    try:
                        _, scene_data = input_queue.get_nowait()
                        safe_cleanup_tensors(scene_data)
                    except Empty:
                        break
                while not output_queue.empty():
                    try:
                        tensor = output_queue.get_nowait()
                        if torch.is_tensor(tensor) and tensor.is_cuda:
                            tensor.detach_()
                            del tensor
                    except Empty:
                        break
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error in render process cleanup: {e}")

        def signal_handler(signum, frame):
            print("Render process received shutdown signal")
            cleanup_render_process()
            sys.exit(0)

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        with RenderContext():
            try:
                print("Render process started")
                while True:
                    try:
                        render_event.wait()
                        render_event.clear()
                        w2c, scene_data = input_queue.get()
                        with torch.cuda.device(0):
                            with torch.no_grad():
                                cam = setup_camera(w, h, k, w2c, near, far,
                                    bg=torch.tensor([0, 177.0/255, 64.0/255], device='cuda'))
                                im, radi, _ = Renderer(raster_settings=cam)(**scene_data)
                                safe_cleanup_tensors(scene_data)
                                im = im.permute(1,2,0).contiguous()
                                im = (im.clamp(0,1)*255).to(torch.uint8)
                                shared_im = im.share_memory_()
                                output_queue.put(shared_im)
                                render_ready_event.set()
                                del im
                                torch.cuda.empty_cache()
                    except Exception as e:
                        print(f"Error in render process: {e}")
                        render_ready_event.set()
                        continue
            finally:
                cleanup_render_process()
    def encode_image(self):
        self.client_cnt = gst_h264_endoder_pipeline.main_fun()
        print("Client num:", self.client_cnt)
        while self.running:
            self.encode_event.wait()
            self.encode_event.clear()

            if hasattr(self, 'img') and len(self.img) > 0:
                size = self.img.numel() * self.img.element_size()
                try:
                    gst_h264_endoder_pipeline.push_tensor_frame(
                        self.img.data_ptr(),
                        size,
                        self.frame_number,
                        self.frame_rate,
                        self.client_cnt
                    )
                    self.frame_number += 1
                except Exception as e:
                    print(f"Error encoding frame {self.frame_number}: {e}")
            self.cuda_event.set()
        self.cuda_event.set()
        gst_h264_endoder_pipeline.close_pipeline(self.client_cnt)

    def process_video_buffers(self):
        frame_number = 0
        try:
            while self.running:
                start = time.time()
                size, data_ptr = gst_h264_endoder_pipeline.get_next_video_buffer_data(self.client_cnt)
                self.h264_pck = ctypes.string_at(data_ptr, size)

                self.client.scene.set_background_h264_pckt(
                    self.h264_pck,
                    frame_number % 300
                )

                end = time.time()
                if frame_number % 200 == 0:
                    print(f"{self.client.client_id} Sending encoded packet fps:", 1/(end-start))
                frame_number += 1
        except KeyboardInterrupt:
            print("Stopped processing buffers.")

    def load_scenes(self):
        seg_as_col = False
        window = [0, 2]
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
                        for p in pr:
                            with torch.no_grad():
                                rendervar = {
                                    'means3D': torch.tensor(p['means3D']).cuda().float(),
                                    'colors_precomp': torch.tensor(p['rgb_colors']).cuda().float() if not seg_as_col else torch.tensor(p['seg_colors']).cuda().float(),
                                    'rotations': torch.nn.functional.normalize(torch.tensor(p['unnorm_rotations']).cuda().float()),
                                    'opacities': torch.sigmoid(torch.tensor(pr[0]['logit_opacities']).cuda().float()),
                                    'scales': torch.exp(torch.tensor(pr[0]['log_scales']).cuda().float()),
                                    'means2D': torch.zeros_like(torch.tensor(pr[0]['means3D']).float().cuda(), device="cuda")
                                }
                                self.scene_data.put(rendervar)

            self.load_event.wait()

            if window[1] == len(self.params_files):
                window = [0, 2]

            window[0] += 2
            window[1] += 2

            self.load_event.clear()

    def render_images(self):
        t0 = time.time() + self.interval

        while self.running:
            if self.first_enter:
                self.client.camera.wxyz = self.init_camera()[0]
                self.client.camera.position = self.init_camera()[1]
                self.client.camera.look_at = self.look_at
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
                c2w = np.linalg.inv(w2c)
                self.client.camera.position = c2w[:3,3]
                self.client.camera.wxyz = tf.SO3.from_matrix(c2w[:3,:3]).wxyz
                self.client.camera.look_at = self.look_at

            if not self.isPaused:
                scene_data = self.scene_data.get()

            shared_scene_data = self.share_scene_data(scene_data)

            try:
                self.render_input_queue.put((w2c, shared_scene_data), timeout=self.interval/2)
            except Exception as e:
                print(f"Error putting data in render queue: {e}")
                continue

            self.render_event.set()
            self.render_ready_event.wait()
            self.render_ready_event.clear()

            try:
                self.img = self.render_output_queue.get(timeout=self.interval/2)
                self.encode_event.set()
            except Empty:
                print("Render timeout - frame dropped")
                continue

            self.cuda_event.wait()

            delta = t0 - time.time()
            if delta > 0:
                time.sleep(delta)
            t0 = time.time() + self.interval

            if self.scene_data.qsize() < 10:
                self.load_event.set()

            self.cuda_event.clear()

        self.load_event.set()
        self.encode_event.set()

    @classmethod
    def init_camera(cls, y_angle=0., center_dist=4., cam_height=3., f_ratio=0.82):
        c2w = np.array([[-0.94743326, -0.0982282, -0.30450196,  2.2777],
                    [-0.09894314,  0.99500657, -0.01312203,  0.185],
                    [0.3042704,   0.01769613, -0.95242132,  6.5],
                    [0.,          0.,          0.,           1.]])
        wxyz = tf.SO3.from_matrix(c2w[:3,:3]).wxyz
        return wxyz, c2w[:3,3]

    @classmethod
    def get_theta_phi_angles_from_cam_pos(cls, position):
        position_length = np.linalg.norm(position)
        theta = np.arccos(position[1] / position_length)
        phi = np.arctan2(position[2], position[0])
        theta_degrees = np.degrees(theta)
        phi_degrees = np.degrees(phi)
        return theta_degrees, phi_degrees    

if __name__ == "__main__":  
    # Configuration  
    exp_name = "samples"  
    sequence = "sample_1"  

    # Create and start viewer  
    viewer = Viewer(seq=sequence, exp=exp_name, w=1920, h=1080)  
    time.sleep(0.2)  
    viewer.start_viewer()  