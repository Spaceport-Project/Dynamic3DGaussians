from setuptools import setup, Extension
import pybind11
import os

# opencv_include_dir = os.popen('pkg-config --cflags-only-I opencv4').read().strip().replace('-I', '')
opencv_cflags = os.popen('pkg-config --cflags opencv4').read().strip()

opencv_libs = os.popen('pkg-config --libs opencv4').read().strip()
cuda_include = ['/usr/local/cuda/include']
cuda_lib_dir = ['/usr/local/cuda/lib64']
cuda_libs = ['cudart']
jpeg_libs = ['turbojpeg']
ext_modules = [
  Extension(
      'jpeg_encoder',
      ['jpeg_encoder.cpp'],
      # include_dirs=[pybind11.get_include(), opencv_cflags.split(), cuda_include],
      include_dirs= [ pybind11.get_include(), opencv_cflags.split() ] + cuda_include ,

      libraries=cuda_libs + jpeg_libs,
      library_dirs = cuda_lib_dir,
      extra_compile_args=opencv_cflags.split(),
      extra_link_args=opencv_libs.split(),
      language='c++'
  ),
]

setup(
  name='Jpeg Encoder',
  version='1.0',
  description='A sample C++ extension to encode images using OpenCV and pybind11',
  ext_modules=ext_modules,
)
