import torch
import torch.multiprocessing as mp
import numpy as np
import concurrent.futures as cf
import threading
import time
from collections import defaultdict
from multiprocessing import shared_memory

class ZeroCopyTensorManager:
    def __init__(self, tensor_dict):
        """Initialize with your existing tensor dictionary - no file I/O!"""
        self.tensor_dict = tensor_dict
        self.shared_memories = {}
        self.tensor_info = {}
        
        # Convert to shared memory directly (no files!)
        self._create_shared_memory_tensors()
    
    def _create_shared_memory_tensors(self):
        """Create shared memory segments directly from your tensors"""
        print("Creating shared memory tensors (no file I/O)...")
        
        for name, tensor in self.tensor_dict.items():
            # Convert to numpy
            if tensor.is_cuda:
                np_array = tensor.cpu().numpy()
            else:
                np_array = tensor.detach().numpy()
            
            # Create shared memory segment
            shm = shared_memory.SharedMemory(create=True, size=np_array.nbytes)
            
            # Create numpy array that uses the shared memory
            shared_array = np.ndarray(np_array.shape, dtype=np_array.dtype, buffer=shm.buf)
            
            # Copy data directly to shared memory (in RAM, not disk!)
            shared_array[:] = np_array[:]
            
            # Store references
            self.shared_memories[name] = shm
            self.tensor_info[name] = {
                'shm_name': shm.name,
                'shape': np_array.shape,
                'dtype': str(np_array.dtype),
                'size': np_array.nbytes
            }
            
            print(f"Created shared memory tensor '{name}' {np_array.shape} (no file)")
    
    def get_tensor(self, name):
        """Get tensor from shared memory (your existing interface)"""
        info = self.tensor_info[name]
        
        # Connect to existing shared memory
        shm = shared_memory.SharedMemory(name=info['shm_name'])
        
        # Create numpy array view of shared memory
        np_array = np.ndarray(
            info['shape'], 
            dtype=np.dtype(info['dtype']), 
            buffer=shm.buf
        )
        
        # Convert to torch tensor (shares memory with numpy array)
        return torch.from_numpy(np_array)
    
    def get_tensor_dict(self):
        """Get all tensors as a dictionary"""
        tensor_dict = {}
        for name in self.tensor_info.keys():
            tensor_dict[name] = self.get_tensor(name)
        return tensor_dict
    
    def get_tensor_info(self):
        """Get metadata for worker initialization"""
        return self.tensor_info.copy()
    
    def cleanup(self):
        """Clean up shared memory segments"""
        print("Cleaning up shared memory...")
        for name, shm in self.shared_memories.items():
            try:
                shm.close()
                shm.unlink()
                print(f"Cleaned up shared memory for '{name}'")
            except Exception as e:
                print(f"Error cleaning up '{name}': {e}")

# Global variables for worker processes
TENSOR_INFO = None
WORKER_GPU_ID = None
CACHED_TENSOR_DICT = {}
WORKER_SHARED_MEMORIES = {}  # Track shared memory objects in worker

def init_worker(tensor_info, gpu_id):
    """Initialize worker with shared memory tensors (no file access!)"""
    global TENSOR_INFO, WORKER_GPU_ID, CACHED_TENSOR_DICT, WORKER_SHARED_MEMORIES
    
    TENSOR_INFO = tensor_info
    WORKER_GPU_ID = gpu_id
    CACHED_TENSOR_DICT = {}
    WORKER_SHARED_MEMORIES = {}
    
    torch.cuda.set_device(gpu_id)
    
    # Connect to shared memory segments and create tensor dictionary
    for tensor_name, info in tensor_info.items():
        # Connect to existing shared memory (created by main process)
        shm = shared_memory.SharedMemory(name=info['shm_name'])
        
        # Create numpy array view
        np_array = np.ndarray(
            info['shape'], 
            dtype=np.dtype(info['dtype']), 
            buffer=shm.buf
        )
        
        # Convert to torch tensor
        CACHED_TENSOR_DICT[tensor_name] = torch.from_numpy(np_array)
        WORKER_SHARED_MEMORIES[tensor_name] = shm  # Keep reference
    
    print(f"Worker initialized on GPU {gpu_id} with shared memory tensors: {list(CACHED_TENSOR_DICT.keys())}")

def client_task(client_id, tensor_operations, processing_time=0.05):
    """Client task using shared memory tensors"""
    global CACHED_TENSOR_DICT, WORKER_GPU_ID
    
    results = []
    
    for tensor_name, slice_info in tensor_operations:
        # Access tensor from shared memory (zero-copy!)
        tensor = CACHED_TENSOR_DICT[tensor_name]
        
        # Apply your operations
        if isinstance(slice_info, tuple):
            start_idx, end_idx = slice_info
            cpu_chunk = tensor[start_idx:end_idx]
        elif isinstance(slice_info, slice):
            cpu_chunk = tensor[slice_info]
        else:
            cpu_chunk = tensor[slice_info]
        
        # Move to GPU
        gpu_chunk = cpu_chunk.to(f"cuda:{WORKER_GPU_ID}", non_blocking=True)
        
        # Your processing
        processed = gpu_chunk * 2.0  # Replace with your logic
        results.append((tensor_name, processed.shape))
    
    time.sleep(processing_time)
    return (client_id, WORKER_GPU_ID, results, "completed")

# ... (rest of DynamicClientManager stays the same) ...

class DynamicClientManager:
    def __init__(self, tensor_manager, 
                 initial_workers_per_gpu=2, 
                 max_workers_per_gpu=20,
                 scale_up_threshold=0.8,
                 scale_down_threshold=0.3,
                 monitoring_interval=2.0):
        
        self.tensor_manager = tensor_manager
        self.initial_workers_per_gpu = initial_workers_per_gpu
        self.max_workers_per_gpu = max_workers_per_gpu
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.monitoring_interval = monitoring_interval
        
        self.num_gpus = torch.cuda.device_count()
        self.tensor_info = tensor_manager.get_tensor_info()
        
        self.executors = {}
        self.executor_metrics = {}
        self.active_futures = defaultdict(set)
        self.completed_tasks = defaultdict(int)
        
        self.lock = threading.RLock()
        self.shutdown_event = threading.Event()
        self.monitor_thread = None
        self.next_gpu = 0
        
        self._initialize_executors()
        self._start_monitoring()
    
    def _initialize_executors(self):
        with self.lock:
            for gpu_id in range(self.num_gpus):
                executor = cf.ProcessPoolExecutor(
                    max_workers=self.initial_workers_per_gpu,
                    initializer=init_worker,
                    initargs=(self.tensor_info, gpu_id),
                    mp_context=mp.get_context('spawn')
                )
                
                self.executors[gpu_id] = executor
                self.executor_metrics[gpu_id] = {
                    'current_workers': self.initial_workers_per_gpu,
                    'active_tasks': 0,
                    'total_completed': 0,
                    'last_scale_time': time.time(),
                    'utilization': 0.0
                }
        
        print(f"Initialized {self.num_gpus} GPU executors with {self.initial_workers_per_gpu} workers each")
    
    def _start_monitoring(self):
        self.monitor_thread = threading.Thread(target=self._monitor_and_scale, daemon=True)
        self.monitor_thread.start()
    
    def _monitor_and_scale(self):
        while not self.shutdown_event.wait(self.monitoring_interval):
            try:
                self._update_metrics()
                self._auto_scale()
            except Exception as e:
                print(f"Error in monitoring thread: {e}")
    
    def _update_metrics(self):
        with self.lock:
            for gpu_id in range(self.num_gpus):
                metrics = self.executor_metrics[gpu_id]
                active_tasks = len(self.active_futures[gpu_id])
                current_workers = metrics['current_workers']
                
                utilization = active_tasks / current_workers if current_workers > 0 else 0
                metrics['active_tasks'] = active_tasks
                metrics['utilization'] = utilization
    
    def _auto_scale(self):
        current_time = time.time()
        
        with self.lock:
            for gpu_id in range(self.num_gpus):
                metrics = self.executor_metrics[gpu_id]
                utilization = metrics['utilization']
                current_workers = metrics['current_workers']
                last_scale_time = metrics['last_scale_time']
                
                if current_time - last_scale_time < 10:
                    continue
                
                if (utilization > self.scale_up_threshold and 
                    current_workers < self.max_workers_per_gpu):
                    
                    new_workers = min(current_workers + 2, self.max_workers_per_gpu)
                    self._scale_gpu_workers(gpu_id, new_workers)
                    print(f"🔼 Scaled UP GPU {gpu_id}: {current_workers} → {new_workers} workers")
                
                elif (utilization < self.scale_down_threshold and 
                      current_workers > self.initial_workers_per_gpu):
                    
                    new_workers = max(current_workers - 1, self.initial_workers_per_gpu)
                    self._scale_gpu_workers(gpu_id, new_workers)
                    print(f"🔽 Scaled DOWN GPU {gpu_id}: {current_workers} → {new_workers} workers")
    
    def _scale_gpu_workers(self, gpu_id, new_worker_count):
        with self.lock:
            old_executor = self.executors[gpu_id]
            
            new_executor = cf.ProcessPoolExecutor(
                max_workers=new_worker_count,
                initializer=init_worker,
                initargs=(self.tensor_info, gpu_id),
                mp_context=mp.get_context('spawn')
            )
            
            self.executors[gpu_id] = new_executor
            self.executor_metrics[gpu_id]['current_workers'] = new_worker_count
            self.executor_metrics[gpu_id]['last_scale_time'] = time.time()
            
            threading.Thread(target=lambda: old_executor.shutdown(wait=True), daemon=True).start()
    
    def submit_client_task(self, client_id, tensor_operations, processing_time=0.05):
        with self.lock:
            best_gpu = min(range(self.num_gpus), 
                          key=lambda gpu: self.executor_metrics[gpu]['utilization'])
            
            executor = self.executors[best_gpu]
            future = executor.submit(client_task, client_id, tensor_operations, processing_time)
            
            self.active_futures[best_gpu].add(future)
            
            def on_complete(fut):
                with self.lock:
                    self.active_futures[best_gpu].discard(fut)
                    self.executor_metrics[best_gpu]['total_completed'] += 1
            
            future.add_done_callback(on_complete)
            return future, best_gpu
    
    def get_status(self):
        with self.lock:
            status = {}
            for gpu_id in range(self.num_gpus):
                metrics = self.executor_metrics[gpu_id]
                status[gpu_id] = {
                    'workers': metrics['current_workers'],
                    'active_tasks': metrics['active_tasks'],
                    'utilization': f"{metrics['utilization']:.2f}",
                    'total_completed': metrics['total_completed']
                }
            return status
    
    def shutdown(self):
        print("Shutting down dynamic client manager...")
        
        self.shutdown_event.set()
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        with self.lock:
            for executor in self.executors.values():
                executor.shutdown(wait=True)

def main():
    mp.set_start_method("spawn", force=True)
    
    # Your existing tensor dictionary
    your_tensor_dict = {
        "model_weights": torch.randn(1000000, 128),
        "embeddings": torch.randn(500000, 256),
        "features": torch.randn(800000, 64),
    }
    
    # Create shared memory tensors (NO FILE I/O!)
    tensor_manager = ZeroCopyTensorManager(your_tensor_dict)
    
    try:
        manager = DynamicClientManager(
            tensor_manager,
            initial_workers_per_gpu=8,
            max_workers_per_gpu=8
        )
        
        # Test with your tensor operations
        futures = []
        for i in range(8):
            tensor_operations = [
                ("model_weights", (i*100, (i+1)*100)),
                ("embeddings", (i*50, (i+1)*50)),
            ]
            
            future, gpu_id = manager.submit_client_task(i, tensor_operations)
            futures.append((i, future))
        
        for client_id, future in futures:
            result = future.result()
            print(f"✅ Client {result[0]} completed: {result[2]}")
        
    finally:
        manager.shutdown()
        tensor_manager.cleanup()

if __name__ == "__main__":
    main()