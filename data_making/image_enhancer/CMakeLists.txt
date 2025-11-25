cmake_minimum_required(VERSION 3.10)
project(ImageEnhancer)

set(CMAKE_CXX_STANDARD 11)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Add executable
add_executable(image_enhancer main.cpp)

# Link OpenCV libraries
target_link_libraries(image_enhancer ${OpenCV_LIBS})

# Include OpenCV headers
target_include_directories(image_enhancer PRIVATE ${OpenCV_INCLUDE_DIRS})