import torch
import torch.multiprocessing as mp
import numpy as np
import os
import tempfile
import concurrent.futures as cf
import threading
import time

class TrulySharedTensorManager:
    def __init__(self, temp_dir=None):
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.tensor_files = {}
        self.tensor_info = {}
    
    def create_shared_tensor(self, name, shape, dtype=torch.float32):
        file_path = os.path.join(self.temp_dir, f"shared_tensor_{name}.dat")
        
        np_dtype = torch.zeros(1, dtype=dtype).numpy().dtype
        total_elements = int(np.prod(shape))
        
        data = np.random.randn(total_elements).astype(np_dtype)
        data.tofile(file_path)
        
        self.tensor_files[name] = file_path
        self.tensor_info[name] = {
            'shape': shape,
            'dtype': str(np_dtype),
            'file_path': file_path
        }
        
        print(f"Created shared tensor '{name}' at {file_path}")
        return self.get_tensor(name)
    
    def get_tensor(self, name):
        info = self.tensor_info[name]
        np_array = np.memmap(
            info['file_path'], 
            dtype=info['dtype'], 
            mode='r+',
            shape=info['shape']
        )
        return torch.from_numpy(np_array)
    
    def get_tensor_info(self):
        return self.tensor_info.copy()
    
    def cleanup(self):
        for file_path in self.tensor_files.values():
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Cleaned up {file_path}")

# Global variables for worker processes
TENSOR_INFO = None
WORKER_GPU_ID = None
CACHED_TENSORS = {}  # Cache tensors in each worker process

def init_worker(tensor_info, gpu_id):
    """Initialize worker and pre-load all tensors once"""
    global TENSOR_INFO, WORKER_GPU_ID, CACHED_TENSORS
    TENSOR_INFO = tensor_info
    WORKER_GPU_ID = gpu_id
    CACHED_TENSORS = {}  # Reset cache for this worker
    
    torch.cuda.set_device(gpu_id)
    
    # Pre-load all tensors once in this worker
    for tensor_name, info in tensor_info.items():
        np_array = np.memmap(
            info['file_path'], 
            dtype=info['dtype'], 
            mode='r+',
            shape=info['shape']
        )
        CACHED_TENSORS[tensor_name] = torch.from_numpy(np_array)
    
    print(f"Worker initialized on GPU {gpu_id} with cached tensors: {list(CACHED_TENSORS.keys())}")

def client_task(client_id, tensor_key, start_idx, end_idx):
    """Client task - uses pre-cached tensor (no new memory allocation)"""
    global CACHED_TENSORS, WORKER_GPU_ID
    
    # Use pre-cached tensor (no new memmap creation!)
    shared_tensor = CACHED_TENSORS[tensor_key]
    
    # Get chunk (this is a view, not a copy)
    cpu_chunk = shared_tensor[start_idx:end_idx]
    
    # Move to GPU
    gpu_chunk = cpu_chunk.to(f"cuda:{WORKER_GPU_ID}", non_blocking=True)
    
    # Simulate work
    time.sleep(0.05)
    result = torch.randn(64, 64, 3, device=f"cuda:{WORKER_GPU_ID}")
    
    return (client_id, WORKER_GPU_ID, result.shape, "completed")

class TrueZeroCopyClientManager:
    def __init__(self, tensor_manager, max_workers_per_gpu=5):
        self.tensor_manager = tensor_manager
        self.max_workers_per_gpu = max_workers_per_gpu
        self.num_gpus = torch.cuda.device_count()
        self.executors = {}
        self.next_gpu = 0
        self.lock = threading.Lock()
        
        tensor_info = tensor_manager.get_tensor_info()
        
        for gpu_id in range(self.num_gpus):
            executor = cf.ProcessPoolExecutor(
                max_workers=max_workers_per_gpu,
                initializer=init_worker,
                initargs=(tensor_info, gpu_id),
                mp_context=mp.get_context('spawn')
            )
            self.executors[gpu_id] = executor
        
        print(f"Created {self.num_gpus} executors with {max_workers_per_gpu} workers each")
    
    def submit_client_task(self, client_id, tensor_key, start_idx, end_idx):
        with self.lock:
            gpu_id = self.next_gpu % self.num_gpus
            self.next_gpu += 1
        
        executor = self.executors[gpu_id]
        future = executor.submit(client_task, client_id, tensor_key, start_idx, end_idx)
        return future
    
    def shutdown(self):
        for executor in self.executors.values():
            executor.shutdown(wait=True)

def test_task_scaling():
    """Test that memory stays flat regardless of number of tasks"""
    mp.set_start_method("spawn", force=True)
    
    tensor_manager = TrulySharedTensorManager()
    tensor_manager.create_shared_tensor("A", (100000, 128))
    tensor_manager.create_shared_tensor("B", (500000, 256))
    
    try:
        # Create manager once with fixed number of workers
        manager = TrueZeroCopyClientManager(tensor_manager, max_workers_per_gpu=5)
        
        # Test with 2 tasks
        print("=== Testing with 2 tasks ===")
        futures = []
        for i in range(1):
            future = manager.submit_client_task(i, "A", i*100, (i+1)*100)
            futures.append(future)
        
        for future in cf.as_completed(futures):
            result = future.result()
            print(f"Client {result[0]} completed on GPU {result[1]}")
        
        # print("2 tasks completed. Check memory usage now.")
        # time.sleep(2)  # Give time to check memory
        
        # # Test with 20 tasks (should NOT increase memory)
        # print("\n=== Testing with 20 tasks ===")
        # futures = []
        # for i in range(20):
        #     future = manager.submit_client_task(i, "A", i*100, (i+1)*100)
        #     futures.append(future)
        
        # for future in cf.as_completed(futures):
        #     result = future.result()
        #     print(f"Client {result[0]} completed on GPU {result[1]}")
        
        # print("20 tasks completed. Memory should be the same!")
        
        manager.shutdown()
        
    finally:
        tensor_manager.cleanup()

def test_memory_behavior():
    """Verify memory behavior is now correct"""
    mp.set_start_method("spawn", force=True)
    
    tensor_manager = TrulySharedTensorManager()
    tensor_manager.create_shared_tensor("A", (1000000, 128))
    
    try:
        manager = TrueZeroCopyClientManager(tensor_manager, max_workers_per_gpu=5)
        
        # Test different task counts
        for num_tasks in [1, 5, 10, 20, 50]:
            print(f"\n=== Testing {num_tasks} tasks ===")
            
            futures = []
            for i in range(num_tasks):
                future = manager.submit_client_task(i, "A", i*10, (i+1)*10)
                futures.append(future)
            
            completed = 0
            for future in cf.as_completed(futures):
                result = future.result()
                completed += 1
            
            print(f"Completed {completed} tasks. Check memory now.")
            time.sleep(1)  # Time to observe memory
        
        manager.shutdown()
        
    finally:
        tensor_manager.cleanup()
def main():
    # test_task_scaling()

    test_memory_behavior()
if __name__ == "__main__":
    main()