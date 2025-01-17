from setuptools import setup, Extension
import pybind11
import os


inc_dirs = [
	'/usr/lib/x86_64-linux-gnu/glib-2.0/include',
	'/usr/local/gstreamer-1.25/include/gstreamer-1.0/',
	'/usr/include/glib-2.0',
	'/usr/include/gstreamer-1.0',
	'/usr/include/libsoup-2.4',
	'/include/json-glib-1.0',
	'/home/hamit/miniconda3/envs/nerfstudio/lib/python3.8/site-packages/pybind11/include',
	'/home/hamit/miniconda3/envs/nerfstudio/include/python3.8',
]


cuda_include = ['/usr/local/cuda/include']
cuda_lib_dir = ['/usr/local/cuda/lib64']
gst_libs_dir = ['/usr/local/gstreamer-1.25/lib/x86_64-linux-gnu', 
                '/home/hamit/miniconda3/envs/nerfstudio/lib/']
cuda_libs = ['cudart']
gst_libs =  [ 'gstapp-1.0', 'gstcuda-1.0']
ext_modules = [
  Extension(
      'gst_h264_endoder_pipeline',
      ['gst_h264_endoder_pipeline.cpp'],
      include_dirs= inc_dirs + [ pybind11.get_include() ] + cuda_include ,
      libraries = cuda_libs + gst_libs ,
      library_dirs = cuda_lib_dir + gst_libs_dir,
      # extra_compile_args=opencv_cflags.split(),
      # extra_link_args=opencv_libs.split(),
      language='c++'
  ),
]

setup(
  name='Gst h264 endoder',
  version='1.0',
  description='A sample C++ extension to encode images using Gstreamer and pybind11',
  ext_modules=ext_modules,
)
