import threading
import time
import gi
import torch
gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')
gi.require_version('GstVideo', '1.0')
# gi.require_version('GstCuda', '1.0')
from gi.repository import Gst, GstApp, GstVideo, GLib, GObject
import cv2
import numpy as np
# import cupy as cp
# import pycuda.driver as cuda
# import pycuda.autoinit
# import gst_cuda_wrapper
from PIL import Image
import ctypes
import gst_h264_endoder_pipeline

libc = ctypes.CDLL("libc.so.6") 


Gst.init(None)

IMAGE_WIDTH = 4096
IMAGE_HEIGHT = 3000
FRAMERATE = 9


def bayer_to_rgb(bayer_image):
  return cv2.cvtColor(bayer_image, cv2.COLOR_BayerRG2BGR)

def push_cuda_tensor_frame(file, appsrc, context):
  global frame_number
  raw_data = file.read(IMAGE_WIDTH * IMAGE_HEIGHT)
  
  if len(raw_data) != IMAGE_WIDTH * IMAGE_HEIGHT:
      return False  # End of file or not enough data
  
  bayer_image = np.frombuffer(raw_data, dtype=np.uint8).reshape(IMAGE_HEIGHT, IMAGE_WIDTH)
  rgb_img = bayer_to_rgb(bayer_image)
  # output_path = f"{1+1}.png"
  # Image.fromarray(rgb_img).save(output_path)
  # size_in_bytes = rgb_img.nbytes
  # rgba_img= cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGBA)
    # Use cupy to allocate CUDA memory and copy data
  # cuda_mem = cp.asarray(rgb_img.tobytes())
  # size_in_bytes = cuda_mem.nbytes

  # Allocate CUDA memory
  # size_in_bytes = rgba_img.nbytes
  # cuda_mem = cuda.mem_alloc(size_in_bytes)
  tensor = torch.from_numpy(rgb_img).to('cuda')  # Move to CUDA device
  size = tensor.numel() * tensor.element_size()  # Total number of bytes

  # Get the pointer to the underlying data
  data_ptr = tensor.data_ptr() 





 
global frame_number
frame_number = 0
def process_buffers():
  cnt = 0
  try:
      while True:
          # Get the next buffer data as a numpy array
          size, data_ptr = gst_h264_endoder_pipeline.get_next_buffer_data()
          # data = gst_h264_endoder_pipeline.get_next_buffer_data()
          if size ==0 :
            break
          with open(f"{cnt}.h264", "wb") as f:
            # g = (ctypes.c_char*size).from_address(data_ptr)
            g = ctypes.string_at(data_ptr, size) 
            # ptr = ctypes.cast(data_ptr, ctypes.POINTER(ctypes.c_uint8))
            # libc.free(ptr)
            f.write(g)
           
          # print("Received buffer of size:", size, data_ptr)
          # Process the data as needed
          cnt += 1
  except KeyboardInterrupt:
      print("Stopped processing buffers.")

  
  

if __name__ == "__main__":
  pipeline = gst_h264_endoder_pipeline.main_fun()
  proc = threading.Thread(target=process_buffers)
  proc.start()
  with open("/home/hamit/Downloads/bayer8.bin", "rb") as file:
    while True:
      
      start = time.time()
      raw_data = file.read(IMAGE_WIDTH * IMAGE_HEIGHT)
      if not raw_data:
        break
      # mutable_data = bytearray(raw_data)
      # mutable_data = np.frombuffer(raw_data, dtype=np.uint8)
      
      bayer_image = np.frombuffer(raw_data, dtype=np.uint8).reshape(IMAGE_HEIGHT, IMAGE_WIDTH)
      rgb_img = bayer_to_rgb(bayer_image)
      tensor = torch.from_numpy(rgb_img).cuda()  # Move to CUDA device
      size = tensor.numel() * tensor.element_size()  # Total number of bytes

      # Get the pointer to the underlying data
      data_ptr = tensor.data_ptr() 
      # size = mutable_data.nbytes
      end = time.time()
      # print(end-start)
      # data_ctypes_ptr = ctypes.cast(data_ptr, ctypes.POINTER(ctypes.c_uint8))
      # wrapper = lib.PipelineWrapper_new(data_ctypes_ptr, size, frame_number, FRAMERATE)
      # print("push start")
      gst_h264_endoder_pipeline.push_cuda_tensor_frame(data_ptr, size, frame_number, FRAMERATE)
      # print("push done")
      # buf= (ctypes.c_char*size).from_address(data)
      frame_number += 1

    # lib.PipelineWrapper_callEOS(wrapper)
    # lib.PipelineWrapper_free(wrapper)
    print("end of while")
    gst_h264_endoder_pipeline.close_pipeline()
  proc.join()
   
  print("Done")

