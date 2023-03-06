cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(CUDA_KDtree)
set(CMAKE_CUDA_COMPILER  "/usr/local/cuda/bin/nvcc")
set(CUDA_HOME /usr/local/cuda-11.6)
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
# put predefined cmake projects in their own solution folder
# set_property(GLOBAL PROPERTY USE_FOLDERS ON)
find_package(CUDA)
string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_86,code=sm_86")

set(SOURCE_FILES main.cpp KDtree.cpp CUDA_KDtree.cpp CUDA_KDtree.cu )

include_directories(${CMAKE_SOURCE_DIR})

cuda_add_executable(CUDA_KDtree ${SOURCE_FILES})
