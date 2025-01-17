cmake_minimum_required(VERSION 2.8)

project (test)
set(CMAKE_BUILD_TYPE Release)
find_package(OpenCV REQUIRED)
# set(Boost_USE_STATIC_LIBS OFF)
# set(Boost_USE_MULTITHREADED ON)
# set(Boost_USE_STATIC_RUNTIME OFF)
# find_package(Boost  REQUIRED )
# find_package( Boost REQUIRED  )
find_package( Boost REQUIRED COMPONENTS system program_options regex )
find_package(CUDA REQUIRED)


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(
        /usr/lib/x86_64-linux-gnu/glib-2.0/include
        /usr/local/gstreamer-1.25/include/gstreamer-1.0/
        /usr/include/glib-2.0
        /usr/include/gstreamer-1.0
        /usr/include/libsoup-2.4
        /usr/include/json-glib-1.0
        /home/hamit/miniconda3/envs/nerfstudio/lib/python3.8/site-packages/pybind11/include
        /home/hamit/miniconda3/envs/nerfstudio/include/python3.8
        )
link_directories( /usr/local/gstreamer-1.25/lib/x86_64-linux-gnu /home/hamit/miniconda3/envs/nerfstudio/lib/python3.8/config-3.8-x86_64-linux-gnu /home/hamit/miniconda3/envs/nerfstudio/lib/)

# add_library(camera_app SHARED appsrc.cpp)
# target_link_libraries( camera_app PRIVATE  gstcuda-1.0 gstapp-1.0 cudart  )

# add_library(appsrcwrapper SHARED appsrcwrapper.cpp appsrcwrapper.h )
# target_link_libraries( appsrcwrapper PRIVATE  gstcuda-1.0 gstapp-1.0 cudart  )


# add_library(pipeline_wrapper SHARED pipeline_wrapper.cpp pipeline_wrapper.h )
# target_link_libraries( pipeline_wrapper PRIVATE  gstcuda-1.0 gstapp-1.0 cudart )

set (MyBoost_INCLUDE_DIRS /home/hamit/miniconda3/envs/nerfstudio/include/python3.8)
set (MyBoost_LIBRARIES  boost_python38  boost_numpy38 crypt pthread dl  util m  )

add_library(example SHARED example.cpp )
target_include_directories(example PRIVATE /home/hamit/miniconda3/envs/nerfstudio/include/python3.8 ${MyBoost_INCLUDE_DIRS} )
target_link_libraries( example PRIVATE ${MyBoost_LIBRARIES})

add_library(example2 SHARED example2.cpp )
target_include_directories(example2 PRIVATE /home/hamit/miniconda3/envs/nerfstudio/include/python3.8 ${Boost_INCLUDE_DIRS} )
target_link_libraries( example2 PRIVATE ${Boost_LIBRARIES})
message(${Boost_INCLUDE_DIRS} ${Boost_LIBRARIES})


# add_library(boost_pipeline_wrapper SHARED boost_pipeline_wrapper.cpp boost_pipeline_wrapper.h)
# target_compile_options(boost_pipeline_wrapper PRIVATE -fPIC -Wl,-soname,example -Wno-undef  -export-dynamic)
# target_include_directories(boost_pipeline_wrapper PRIVATE ${CUDA_INCLUDE_DIRS} /home/hamit/miniconda3/envs/nerfstudio/include/python3.8 ${Boost_INCLUDE_DIRS} ${MyBoost_INCLUDE_DIRS})
# target_link_libraries( boost_pipeline_wrapper PRIVATE ${Boost_LIBRARIES} ${MyBoost_LIBRARIES} gstcuda-1.0 gstapp-1.0 cudart)

# add_library(pybind11_pipeline_wrapper SHARED pybind11_pipeline_wrapper.cpp pybind11_pipeline_wrapper.h)
# target_compile_options(pybind11_pipeline_wrapper PRIVATE -fPIC -Wl,-soname,example -Wno-undef  -export-dynamic)
# target_include_directories(pybind11_pipeline_wrapper PRIVATE ${CUDA_INCLUDE_DIRS} /home/hamit/miniconda3/envs/nerfstudio/include/python3.8 ${Boost_INCLUDE_DIRS} ${MyBoost_INCLUDE_DIRS})
# target_link_libraries( pybind11_pipeline_wrapper PRIVATE ${Boost_LIBRARIES} ${MyBoost_LIBRARIES} gstcuda-1.0 gstapp-1.0 cudart)

add_library(pybind11_pipeline_wrapper_fun SHARED pybind11_pipeline_wrapper_fun.cpp )
target_compile_options(pybind11_pipeline_wrapper_fun PRIVATE -fPIC -Wl,-soname,example -Wno-undef  -export-dynamic)
target_include_directories(pybind11_pipeline_wrapper_fun PRIVATE ${CUDA_INCLUDE_DIRS} /home/hamit/miniconda3/envs/nerfstudio/include/python3.8  ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${MyBoost_INCLUDE_DIRS})
target_link_libraries( pybind11_pipeline_wrapper_fun PRIVATE ${Boost_LIBRARIES} ${MyBoost_LIBRARIES} ${OpenCV_LIBS}  gstcuda-1.0 gstapp-1.0 cudart) 