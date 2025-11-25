import torch
import torch.multiprocessing as mp
import numpy as np
import threading
import time
import queue
from collections import defaultdict
from multiprocessing import shared_memory, Queue, Process
import uuid
import json
import os
import signal
import psutil

class ZeroCopyTensorManager:
    def __init__(self, tensor_dict):
        self.tensor_dict = tensor_dict
        self.shared_memories = {}
        self.tensor_info = {}
        self._create_shared_memory_tensors()
    
    def _create_shared_memory_tensors(self):
        print("Creating shared memory tensors...")
        
        for name, tensor in self.tensor_dict.items():
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

def high_frequency_renderer_process(client_id, tensor_info, tensor_operations, render_config, 
                                  result_queue, control_queue, gpu_id):
    """
    Standalone rendering process - no global variables needed
    Each process gets its own copy of parameters
    """
    try:
        # Set GPU device for this process
        torch.cuda.set_device(gpu_id)
        
        # Connect to shared memory tensors (zero-copy)
        shared_tensors = {}
        shared_memories = {}
        
        for tensor_name, info in tensor_info.items():
            try:
                shm = shared_memory.SharedMemory(name=info['shm_name'])
                np_array = np.ndarray(
                    info['shape'], 
                    dtype=np.dtype(info['dtype']), 
                    buffer=shm.buf
                )
                shared_tensors[tensor_name] = torch.from_numpy(np_array)
                shared_memories[tensor_name] = shm
            except Exception as e:
                print(f"❌ Error connecting to shared memory '{tensor_name}': {e}")
                return
        
        print(f"🚀 Process started for client {client_id} on GPU {gpu_id}")
        print(f"   Connected to shared tensors: {list(shared_tensors.keys())}")
        
        # Pre-warm GPU
        dummy = torch.randn(100, 100, device=f'cuda:{gpu_id}')
        _ = dummy @ dummy  # Warm up CUDA context
        del dummy
        
        # Timing configuration
        TARGET_FRAME_TIME = 0.030  # 30ms = 33.33 FPS
        frame_count = 0
        start_time = time.time()
        last_frame_time = start_time
        
        # Pre-allocate GPU tensors for efficiency
        image_size = render_config.get('image_size', (256, 256))
        batch_size = render_config.get('batch_size', 1)
        
        # Pre-allocate output tensors to avoid repeated allocation
        output_tensors = {}
        for tensor_name, _ in tensor_operations:
            output_tensors[tensor_name] = torch.zeros(
                batch_size, 3, *image_size, 
                device=f'cuda:{gpu_id}',
                dtype=torch.float32
            )
        
        # Main rendering loop
        while True:
            frame_start = time.time()
            
            # Check for control signals (non-blocking)
            try:
                control_msg = control_queue.get_nowait()
                if control_msg == "STOP":
                    print(f"🛑 Stop signal received for client {client_id}")
                    break
                elif control_msg == "PAUSE":
                    print(f"⏸️ Pause signal received for client {client_id}")
                    while True:
                        control_msg = control_queue.get()
                        if control_msg == "RESUME":
                            print(f"▶️ Resume signal received for client {client_id}")
                            last_frame_time = time.time()
                            break
                        elif control_msg == "STOP":
                            return
            except queue.Empty:
                pass
            
            # High-frequency rendering logic
            processing_start = time.time()
            rendered_images = {}
            
            for tensor_name, slice_info in tensor_operations:
                # Access shared tensor (zero-copy)
                shared_tensor = shared_tensors[tensor_name]
                
                # Dynamic slicing based on frame count for animation
                if isinstance(slice_info, tuple):
                    start_idx, end_idx = slice_info
                    shift = (frame_count * 5) % (shared_tensor.shape[0] - (end_idx - start_idx))
                    dynamic_start = start_idx + shift
                    dynamic_end = dynamic_start + (end_idx - start_idx)
                    cpu_chunk = shared_tensor[dynamic_start:dynamic_end]
                else:
                    cpu_chunk = shared_tensor[slice_info]
                
                # Move to GPU (non-blocking for speed)
                gpu_chunk = cpu_chunk.to(f"cuda:{gpu_id}", non_blocking=True)
                
                # Fast rendering using pre-allocated tensors
                rendered_image = fast_render_frame(
                    gpu_chunk, 
                    tensor_name, 
                    render_config, 
                    client_id, 
                    frame_count,
                    output_tensors[tensor_name],
                    gpu_id
                )
                
                rendered_images[tensor_name] = rendered_image
            
            processing_end = time.time()
            processing_time = processing_end - processing_start
            
            # Create frame result
            frame_result = {
                'client_id': client_id,
                'frame_number': frame_count,
                'timestamp': frame_start,
                'gpu_id': gpu_id,
                'rendered_images': rendered_images,
                'processing_time': processing_time,
                'frame_interval': frame_start - last_frame_time if frame_count > 0 else 0,
                'process_id': os.getpid(),
                'metadata': {
                    'tensor_shapes': {name: img.shape for name, img in rendered_images.items()},
                    'memory_usage': torch.cuda.memory_allocated(gpu_id) if torch.cuda.is_available() else 0,
                    'render_config': render_config
                }
            }
            
            # Put result in queue with minimal blocking
            try:
                result_queue.put(frame_result, timeout=0.005)
                frame_count += 1
            except queue.Full:
                print(f"⚠️ Dropped frame {frame_count} for client {client_id} (queue full)")
                frame_count += 1
                continue
            
            # Precise timing control for 30ms intervals
            frame_end = time.time()
            frame_duration = frame_end - frame_start
            
            sleep_time = TARGET_FRAME_TIME - frame_duration
            if sleep_time > 0:
                time.sleep(sleep_time)
            elif sleep_time < -0.005:
                print(f"⚠️ Frame {frame_count} for client {client_id} took {frame_duration*1000:.1f}ms (target: 30ms)")
            
            last_frame_time = frame_start
    
    except Exception as e:
        print(f"❌ Error in rendering process for client {client_id}: {e}")
        error_result = {
            'client_id': client_id,
            'error': str(e),
            'timestamp': time.time(),
            'frame_number': frame_count if 'frame_count' in locals() else 0,
            'process_id': os.getpid()
        }
        try:
            result_queue.put(error_result, timeout=0.001)
        except queue.Full:
            pass
    
    finally:
        # Clean up shared memory connections
        for shm in shared_memories.values():
            try:
                shm.close()
            except:
                pass
        
        if 'frame_count' in locals():
            total_time = time.time() - start_time
            actual_fps = frame_count / total_time if total_time > 0 else 0
            print(f"🏁 Client {client_id} process completed: {frame_count} frames in {total_time:.1f}s "
                  f"(Actual FPS: {actual_fps:.1f}, Target: 33.33)")

def fast_render_frame(gpu_tensor, tensor_name, render_config, client_id, frame_number, output_tensor, gpu_id):
    """Optimized frame rendering for 30ms intervals"""
    style = render_config.get('style', 'default')
    
    # Time-based animation factor
    time_factor = frame_number * 0.05
    
    # Use pre-allocated output tensor for efficiency
    with torch.no_grad():
        if style == 'smooth_wave':
            wave = torch.sin(torch.tensor(time_factor, device=gpu_tensor.device))
            base_intensity = 0.5 + 0.3 * wave
            output_tensor.fill_(base_intensity)
            tensor_mean = gpu_tensor.mean()
            output_tensor += tensor_mean * 0.2
            
        elif style == 'pulse':
            pulse = torch.abs(torch.sin(torch.tensor(time_factor * 2, device=gpu_tensor.device)))
            output_tensor.fill_(pulse)
            tensor_std = gpu_tensor.std()
            output_tensor *= (1.0 + tensor_std * 0.5)
            
        elif style == 'rotate':
            rotation_phase = (frame_number % 120) / 120.0
            base_value = 0.3 + 0.4 * rotation_phase
            output_tensor.fill_(base_value)
            tensor_influence = torch.clamp(gpu_tensor.mean() * 0.5, 0, 0.5)
            output_tensor += tensor_influence
            
        else:
            base_noise = torch.randn_like(output_tensor) * 0.1
            output_tensor.copy_(base_noise)
            animation = torch.sin(torch.tensor(time_factor, device=gpu_tensor.device))
            output_tensor += 0.5 + 0.3 * animation
            output_tensor += gpu_tensor.mean() * 0.2
    
    torch.clamp_(output_tensor, 0, 1)
    return output_tensor

class ClientJobProcessor:
    """Handles background jobs for each client in separate threads"""
    
    def __init__(self, client_id, job_config=None):
        self.client_id = client_id
        self.job_config = job_config or {}
        
        # Threading
        self.job_thread = None
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        
        # Job state
        self.job_stats = {
            'jobs_completed': 0,
            'jobs_failed': 0,
            'start_time': 0,
            'last_job_time': 0,
            'avg_job_duration': 0,
            'total_job_time': 0
        }
        
        # Job results storage
        self.job_results = queue.Queue(maxsize=100)
        self.job_lock = threading.Lock()
        
        # Create client-specific directories
        self.client_dir = f"client_data/{client_id}"
        os.makedirs(self.client_dir, exist_ok=True)
        
    def start_background_jobs(self):
        """Start background job processing thread"""
        if self.job_thread and self.job_thread.is_alive():
            print(f"⚠️ Background jobs already running for client {self.client_id}")
            return False
        
        self.stop_event.clear()
        self.pause_event.clear()
        self.job_stats['start_time'] = time.time()
        
        self.job_thread = threading.Thread(
            target=self._job_worker_loop,
            name=f"JobWorker-{self.client_id}",
            daemon=True
        )
        self.job_thread.start()
        
        print(f"🔧 Started background jobs for client {self.client_id}")
        return True
    
    def stop_background_jobs(self):
        """Stop background job processing"""
        if not self.job_thread or not self.job_thread.is_alive():
            return False
        
        self.stop_event.set()
        self.job_thread.join(timeout=5)
        
        print(f"🛑 Stopped background jobs for client {self.client_id}")
        return True
    
    def pause_jobs(self):
        """Pause background job processing"""
        self.pause_event.set()
        print(f"⏸️ Paused background jobs for client {self.client_id}")
    
    def resume_jobs(self):
        """Resume background job processing"""
        self.pause_event.clear()
        print(f"▶️ Resumed background jobs for client {self.client_id}")
    
    def _job_worker_loop(self):
        """Main job processing loop running in background thread"""
        print(f"🔧 Job worker started for client {self.client_id}")
        
        job_interval = self.job_config.get('job_interval', 2.0)  # Default: every 2 seconds
        
        while not self.stop_event.is_set():
            try:
                # Check for pause
                if self.pause_event.is_set():
                    time.sleep(0.1)
                    continue
                
                job_start = time.time()
                
                # Execute the background job
                job_result = self._execute_client_job()
                
                job_end = time.time()
                job_duration = job_end - job_start
                
                # Update statistics
                with self.job_lock:
                    self.job_stats['jobs_completed'] += 1
                    self.job_stats['last_job_time'] = job_end
                    self.job_stats['total_job_time'] += job_duration
                    self.job_stats['avg_job_duration'] = (
                        self.job_stats['total_job_time'] / self.job_stats['jobs_completed']
                    )
                
                # Store job result
                try:
                    self.job_results.put({
                        'client_id': self.client_id,
                        'job_number': self.job_stats['jobs_completed'],
                        'timestamp': job_end,
                        'duration': job_duration,
                        'result': job_result,
                        'thread_name': threading.current_thread().name
                    }, timeout=0.1)
                except queue.Full:
                    print(f"⚠️ Job result queue full for client {self.client_id}")
                
                # Wait for next job interval
                sleep_time = max(0, job_interval - job_duration)
                if not self.stop_event.wait(sleep_time):
                    continue  # Continue if not stopped
                else:
                    break  # Stop event was set
                    
            except Exception as e:
                print(f"❌ Job error for client {self.client_id}: {e}")
                with self.job_lock:
                    self.job_stats['jobs_failed'] += 1
                
                # Wait before retrying
                if not self.stop_event.wait(1.0):
                    continue
                else:
                    break
        
        print(f"🏁 Job worker stopped for client {self.client_id}")
    
    def _execute_client_job(self):
        """Execute a background job for this client"""
        job_type = self.job_config.get('job_type', 'data_processing')
        
        if job_type == 'data_processing':
            return self._data_processing_job()
        elif job_type == 'file_operations':
            return self._file_operations_job()
        elif job_type == 'analytics':
            return self._analytics_job()
        elif job_type == 'model_inference':
            return self._model_inference_job()
        else:
            return self._default_job()
    
    def _data_processing_job(self):
        """Example: Data processing job"""
        data_size = np.random.randint(1000, 10000)
        fake_data = np.random.randn(data_size)
        
        processed_data = {
            'mean': float(np.mean(fake_data)),
            'std': float(np.std(fake_data)),
            'min': float(np.min(fake_data)),
            'max': float(np.max(fake_data)),
            'size': data_size,
            'processing_type': 'statistical_analysis'
        }
        
        time.sleep(0.1 + np.random.random() * 0.2)
        return processed_data
    
    def _file_operations_job(self):
        """Example: File operations job"""
        filename = f"{self.client_dir}/job_{int(time.time())}.json"
        
        job_data = {
            'client_id': self.client_id,
            'timestamp': time.time(),
            'job_type': 'file_operations',
            'random_data': [np.random.random() for _ in range(10)],
            'status': 'completed'
        }
        
        with open(filename, 'w') as f:
            json.dump(job_data, f, indent=2)
        
        time.sleep(0.05 + np.random.random() * 0.1)
        
        return {
            'file_created': filename,
            'file_size': os.path.getsize(filename),
            'operation': 'file_write',
            'data_points': len(job_data['random_data'])
        }
    
    def _analytics_job(self):
        """Example: Analytics job"""
        metrics = {
            'cpu_usage': np.random.random() * 100,
            'memory_usage': np.random.random() * 100,
            'network_io': np.random.randint(0, 1000),
            'disk_io': np.random.randint(0, 500),
            'active_connections': np.random.randint(1, 50),
            'processing_queue_size': np.random.randint(0, 100)
        }
        
        time.sleep(0.08 + np.random.random() * 0.15)
        
        return {
            'analytics_type': 'system_metrics',
            'metrics': metrics,
            'alert_level': 'normal' if metrics['cpu_usage'] < 80 else 'warning',
            'timestamp': time.time()
        }
    
    def _model_inference_job(self):
        """Example: Model inference job"""
        input_size = self.job_config.get('input_size', 128)
        batch_size = self.job_config.get('batch_size', 4)
        
        fake_input = np.random.randn(batch_size, input_size)
        time.sleep(0.15 + np.random.random() * 0.1)
        
        predictions = np.random.random(batch_size)
        confidence_scores = np.random.random(batch_size)
        
        return {
            'model_type': 'classification',
            'input_shape': fake_input.shape,
            'predictions': predictions.tolist(),
            'confidence_scores': confidence_scores.tolist(),
            'inference_time': 0.15 + np.random.random() * 0.1,
            'batch_size': batch_size
        }
    
    def _default_job(self):
        """Default job when no specific type is configured"""
        computation_result = sum(i**2 for i in range(1000))
        time.sleep(0.05 + np.random.random() * 0.1)
        
        return {
            'job_type': 'default_computation',
            'result': computation_result,
            'computation': 'sum_of_squares_1000',
            'execution_time': time.time()
        }
    
    def get_job_result(self, timeout=0.1):
        """Get the latest job result"""
        try:
            return self.job_results.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_all_job_results(self):
        """Get all available job results"""
        results = []
        while True:
            result = self.get_job_result(timeout=0.001)
            if result is None:
                break
            results.append(result)
        return results
    
    def get_job_stats(self):
        """Get job processing statistics"""
        with self.job_lock:
            stats = self.job_stats.copy()
            stats['is_running'] = self.job_thread and self.job_thread.is_alive()
            stats['is_paused'] = self.pause_event.is_set()
            return stats

class ProcessLoopRenderingManager:
    """Rendering manager using individual processes in a loop"""
    
    def __init__(self, tensor_manager, result_queue_size=200):
        self.tensor_manager = tensor_manager
        self.result_queue_size = result_queue_size
        
        self.num_gpus = torch.cuda.device_count()
        self.tensor_info = tensor_manager.get_tensor_info()
        
        # Process management
        self.active_clients = {}  # client_id -> client_info
        self.client_processes = {}  # client_id -> Process
        
        # Client job processors
        self.client_job_processors = {}  # client_id -> ClientJobProcessor
        
        # Performance tracking
        self.client_stats = defaultdict(lambda: {
            'frames_received': 0,
            'last_frame_time': 0,
            'avg_fps': 0,
            'connect_time': 0
        })
        
        # Threading
        self.lock = threading.RLock()
        self.next_gpu = 0
        
        print(f"Initialized process loop manager for {self.num_gpus} GPUs")
    
    def connect_client(self, client_id, tensor_operations, render_config=None, job_config=None):
        """
        Connect client for high-frequency rendering with background jobs using individual process
        """
        if render_config is None:
            render_config = {}
        if job_config is None:
            job_config = {}
        
        # Set default configs
        render_config.setdefault('image_size', (256, 256))
        render_config.setdefault('batch_size', 1)
        render_config.setdefault('style', 'smooth_wave')
        
        job_config.setdefault('job_type', 'data_processing')
        job_config.setdefault('job_interval', 2.0)
        
        with self.lock:
            if client_id in self.active_clients:
                print(f"⚠️ Client {client_id} already connected")
                return None, False
            
            # Choose GPU with least clients (round-robin)
            gpu_loads = defaultdict(int)
            for info in self.active_clients.values():
                gpu_loads[info['gpu_id']] += 1
            
            gpu_id = min(range(self.num_gpus), key=lambda g: gpu_loads[g])
            
            # Create queues
            result_queue = Queue(maxsize=self.result_queue_size)
            control_queue = Queue(maxsize=20)
            
            # Create individual process for this client
            process = Process(
                target=high_frequency_renderer_process,
                args=(
                    client_id,
                    self.tensor_info,
                    tensor_operations,
                    render_config,
                    result_queue,
                    control_queue,
                    gpu_id
                ),
                name=f"Renderer-{client_id}",
                daemon=False  # Don't make daemon so we can properly clean up
            )
            
            # Start the process
            process.start()
            
            # Create and start background job processor
            job_processor = ClientJobProcessor(client_id, job_config)
            job_processor.start_background_jobs()
            
            # Store client info
            self.active_clients[client_id] = {
                'result_queue': result_queue,
                'control_queue': control_queue,
                'gpu_id': gpu_id,
                'connect_time': time.time(),
                'render_config': render_config,
                'job_config': job_config,
                'tensor_operations': tensor_operations,
                'process_id': process.pid
            }
            
            self.client_processes[client_id] = process
            self.client_job_processors[client_id] = job_processor
            self.client_stats[client_id]['connect_time'] = time.time()
            
            print(f"🚀 Connected client {client_id} to GPU {gpu_id} with process PID {process.pid}")
            return result_queue, True
    
    def disconnect_client(self, client_id):
        """Disconnect client and stop all processing"""
        with self.lock:
            if client_id not in self.active_clients:
                print(f"⚠️ Client {client_id} not found")
                return False
            
            client_info = self.active_clients[client_id]
            process = self.client_processes[client_id]
            
            # Stop rendering process
            try:
                client_info['control_queue'].put("STOP", timeout=0.1)
            except queue.Full:
                print(f"⚠️ Control queue full for client {client_id}")
            
            # Wait for process to finish gracefully
            process.join(timeout=2.0)
            
            # Force terminate if still alive
            if process.is_alive():
                print(f"⚠️ Force terminating process for client {client_id}")
                process.terminate()
                process.join(timeout=1.0)
                
                if process.is_alive():
                    print(f"⚠️ Force killing process for client {client_id}")
                    process.kill()
                    process.join()
            
            # Stop background jobs
            if client_id in self.client_job_processors:
                self.client_job_processors[client_id].stop_background_jobs()
                del self.client_job_processors[client_id]
            
            # Remove from active clients
            del self.active_clients[client_id]
            del self.client_processes[client_id]
            
            # Print final stats
            stats = self.client_stats[client_id]
            if stats['frames_received'] > 0:
                total_time = time.time() - stats['connect_time']
                final_fps = stats['frames_received'] / total_time if total_time > 0 else 0
                print(f"📊 Client {client_id} final stats: {stats['frames_received']} frames, "
                      f"Average FPS: {final_fps:.1f}")
            
            print(f"🔌 Disconnected client {client_id}")
            return True
    
    def get_client_frame(self, client_id, timeout=0.035):
        """Get next frame from client's rendering queue"""
        with self.lock:
            if client_id not in self.active_clients:
                return None
            
            result_queue = self.active_clients[client_id]['result_queue']
        
        try:
            frame_result = result_queue.get(timeout=timeout)
            
            # Update client stats
            if 'error' not in frame_result:
                stats = self.client_stats[client_id]
                stats['frames_received'] += 1
                stats['last_frame_time'] = time.time()
                
                if stats['frames_received'] > 1:
                    total_time = stats['last_frame_time'] - stats['connect_time']
                    stats['avg_fps'] = stats['frames_received'] / total_time
            
            return frame_result
            
        except queue.Empty:
            return None
    
    def get_client_job_result(self, client_id, timeout=0.1):
        """Get latest job result from client's background thread"""
        if client_id not in self.client_job_processors:
            return None
        
        return self.client_job_processors[client_id].get_job_result(timeout=timeout)
    
    def get_all_client_job_results(self, client_id):
        """Get all available job results from client's background thread"""
        if client_id not in self.client_job_processors:
            return []
        
        return self.client_job_processors[client_id].get_all_job_results()
    
    def pause_client_rendering(self, client_id):
        """Pause rendering for a specific client"""
        with self.lock:
            if client_id not in self.active_clients:
                return False
            
            try:
                self.active_clients[client_id]['control_queue'].put("PAUSE", timeout=0.1)
                print(f"⏸️ Paused rendering for client {client_id}")
                return True
            except queue.Full:
                return False
    
    def resume_client_rendering(self, client_id):
        """Resume rendering for a specific client"""
        with self.lock:
            if client_id not in self.active_clients:
                return False
            
            try:
                self.active_clients[client_id]['control_queue'].put("RESUME", timeout=0.1)
                print(f"▶️ Resumed rendering for client {client_id}")
                return True
            except queue.Full:
                return False
    
    def pause_client_jobs(self, client_id):
        """Pause background jobs for a specific client"""
        if client_id not in self.client_job_processors:
            return False
        
        self.client_job_processors[client_id].pause_jobs()
        return True
    
    def resume_client_jobs(self, client_id):
        """Resume background jobs for a specific client"""
        if client_id not in self.client_job_processors:
            return False
        
        self.client_job_processors[client_id].resume_jobs()
        return True
    
    def get_client_job_stats(self, client_id):
        """Get job processing statistics for a client"""
        if client_id not in self.client_job_processors:
            return None
        
        return self.client_job_processors[client_id].get_job_stats()
    
    def get_all_latest_frames(self):
        """Get latest frame from each client's rendering queue"""
        latest_frames = {}
        
        with self.lock:
            client_ids = list(self.active_clients.keys())
        
        for client_id in client_ids:
            frame = self.get_client_frame(client_id, timeout=0.001)
            if frame:
                latest_frames[client_id] = frame
        
        return latest_frames
    
    def get_all_latest_job_results(self):
        """Get latest job result from each client's background thread"""
        latest_job_results = {}
        
        for client_id in list(self.client_job_processors.keys()):
            job_result = self.get_client_job_result(client_id, timeout=0.001)
            if job_result:
                latest_job_results[client_id] = job_result
        
        return latest_job_results
    
    def get_process_status(self):
        """Get status of all client processes"""
        process_status = {}
        
        with self.lock:
            for client_id, process in self.client_processes.items():
                try:
                    # Get process info using psutil for detailed stats
                    proc = psutil.Process(process.pid)
                    process_status[client_id] = {
                        'pid': process.pid,
                        'is_alive': process.is_alive(),
                        'cpu_percent': proc.cpu_percent(),
                        'memory_mb': proc.memory_info().rss / 1024 / 1024,
                        'status': proc.status(),
                        'create_time': proc.create_time(),
                        'gpu_id': self.active_clients[client_id]['gpu_id']
                    }
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    process_status[client_id] = {
                        'pid': process.pid,
                        'is_alive': process.is_alive(),
                        'status': 'unknown',
                        'gpu_id': self.active_clients[client_id]['gpu_id']
                    }
        
        return process_status
    
    def get_comprehensive_status(self):
        """Get comprehensive status including rendering, jobs, and processes"""
        with self.lock:
            status = {
                'total_clients': len(self.active_clients),
                'target_fps': 33.33,
                'frame_interval_ms': 30,
                'clients_per_gpu': defaultdict(int),
                'active_clients': list(self.active_clients.keys()),
                'client_details': {},
                'process_status': self.get_process_status()
            }
            
            for client_id, info in self.active_clients.items():
                status['clients_per_gpu'][info['gpu_id']] += 1
                
                # Get job stats
                job_stats = self.get_client_job_stats(client_id)
                render_stats = self.client_stats[client_id]
                
                status['client_details'][client_id] = {
                    'gpu_id': info['gpu_id'],
                    'process_id': info['process_id'],
                    'render_config': info['render_config'],
                    'job_config': info['job_config'],
                    'rendering_stats': render_stats,
                    'job_stats': job_stats,
                    'uptime': time.time() - info['connect_time']
                }
        
        return status
    
    def shutdown(self):
        """Shutdown all clients, jobs, and processes"""
        print("Shutting down process loop rendering manager...")
        
        # Disconnect all clients (this will stop their processes and jobs)
        with self.lock:
            client_ids = list(self.active_clients.keys())
        
        for client_id in client_ids:
            self.disconnect_client(client_id)
        
        # Wait for clean shutdown
        time.sleep(1)
        
        print("Process loop rendering manager shut down")

def main():
    """Example of process loop client processing with rendering and background jobs"""
    mp.set_start_method("spawn", force=True)
    
    # Shared tensors
    shared_tensors = {
        "scene_data": torch.randn(200000, 128),
        "lighting": torch.randn(100000, 64),
        "materials": torch.randn(50000, 256),
    }
    
    tensor_manager = ZeroCopyTensorManager(shared_tensors)
    
    try:
        # Create process loop rendering manager
        render_manager = ProcessLoopRenderingManager(
            tensor_manager,
            result_queue_size=300
        )
        
        # Connect clients with different job configurations
        clients = []
        job_types = ['data_processing', 'file_operations', 'analytics', 'model_inference']
        
        for i in range(8):
            client_id = f"client_{i}"
            
            tensor_operations = [
                ("scene_data", (i*2000, (i+1)*2000)),
                ("lighting", (i*1000, (i+1)*1000)),
            ]
            
            render_config = {
                'image_size': (128 + i*32, 128 + i*32),
                'style': ['smooth_wave', 'pulse', 'rotate', 'default'][i%4],
                'batch_size': 1
            }
            
            job_config = {
                'job_type': job_types[i%4],
                'job_interval': 1.5 + i * 0.5,  # Different intervals
                'input_size': 128 + i * 32,
                'batch_size': 2 + i
            }
            
            result_queue, success = render_manager.connect_client(
                client_id, tensor_operations, render_config, job_config
            )
            
            if success:
                clients.append(client_id)
                print(f"✅ Connected {client_id} with {job_types[i%4]} jobs")
        
        # Monitor both rendering and job processing
        print("\n🚀 Starting monitoring of rendering + background jobs...")
        start_time = time.time()
        frame_counts = defaultdict(int)
        job_counts = defaultdict(int)
        last_stats_time = start_time
        
        # Run for 15 seconds to see both rendering and job results
        while time.time() - start_time < 15:
            loop_start = time.time()
            
            # Get latest frames from all clients
            latest_frames = render_manager.get_all_latest_frames()
            for client_id, frame in latest_frames.items():
                if 'error' not in frame:
                    frame_counts[client_id] += 1
                    
                    # Print every 50th frame to avoid spam
                    if frame_counts[client_id] % 50 == 0:
                        print(f"🎞️ {client_id} frame {frame['frame_number']}: "
                              f"Processing: {frame['processing_time']*1000:.1f}ms "
                              f"(GPU {frame['gpu_id']}, PID {frame['process_id']})")
            
            # Get latest job results from all clients
            latest_job_results = render_manager.get_all_latest_job_results()
            for client_id, job_result in latest_job_results.items():
                job_counts[client_id] += 1
                print(f"🔧 {client_id} job {job_result['job_number']}: "
                      f"{job_result['result'].get('job_type', 'unknown')} "
                      f"({job_result['duration']*1000:.1f}ms) "
                      f"[Thread: {job_result['thread_name']}]")
            
            # Print comprehensive stats every 5 seconds
            current_time = time.time()
            if current_time - last_stats_time >= 5.0:
                print(f"\n📊 Comprehensive Stats (after {current_time - start_time:.1f}s):")
                
                status = render_manager.get_comprehensive_status()
                for client_id, details in status['client_details'].items():
                    render_stats = details['rendering_stats']
                    job_stats = details['job_stats']
                    process_info = status['process_status'][client_id]
                    
                    print(f"   {client_id} (GPU {details['gpu_id']}, PID {details['process_id']}):")
                    print(f"     🎞️ Rendering: {render_stats['avg_fps']:.1f} FPS "
                          f"({render_stats['frames_received']} frames)")
                    print(f"     🔧 Jobs: {job_stats['jobs_completed']} completed, "
                          f"{job_stats['jobs_failed']} failed "
                          f"(Avg: {job_stats['avg_job_duration']*1000:.1f}ms)")
                    print(f"     💻 Process: CPU {process_info.get('cpu_percent', 0):.1f}%, "
                          f"Memory {process_info.get('memory_mb', 0):.1f}MB")
                    print(f"     ⏱️ Uptime: {details['uptime']:.1f}s")
                
                last_stats_time = current_time
            
            # Test pause/resume functionality
            if 7 < (current_time - start_time) < 8:
                print("\n⏸️ Testing pause functionality...")
                render_manager.pause_client_rendering("client_0")
                render_manager.pause_client_jobs("client_1")
            elif 9 < (current_time - start_time) < 10:
                print("\n▶️ Testing resume functionality...")
                render_manager.resume_client_rendering("client_0")
                render_manager.resume_client_jobs("client_1")
            
            # Maintain loop timing
            loop_duration = time.time() - loop_start
            sleep_time = max(0, 0.1 - loop_duration)  # 100ms loop target
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        # Final statistics
        total_time = time.time() - start_time
        print(f"\n📊 Final Results after {total_time:.1f}s:")
        print("🎞️ Rendering Results:")
        for client_id, count in frame_counts.items():
            actual_fps = count / total_time
            print(f"   {client_id}: {count} frames, FPS: {actual_fps:.1f}")
        
        print("🔧 Job Processing Results:")
        for client_id, count in job_counts.items():
            jobs_per_second = count / total_time
            print(f"   {client_id}: {count} jobs, Rate: {jobs_per_second:.2f} jobs/sec")
        
        # Show final process status
        final_process_status = render_manager.get_process_status()
        print(f"\n💻 Final Process Status:")
        for client_id, proc_info in final_process_status.items():
            print(f"   {client_id}: PID {proc_info['pid']}, "
                  f"Alive: {proc_info['is_alive']}, "
                  f"Memory: {proc_info.get('memory_mb', 0):.1f}MB")
        
    finally:
        render_manager.shutdown()
        tensor_manager.cleanup()

if __name__ == "__main__":
    main()