cmake_minimum_required (VERSION 3.1 FATAL_ERROR)

project (ACMMP)


set(CMAKE_PREFIX_PATH /home/zhujun/anaconda3/envs/py39/lib/python3.9/site-packages/torch/share/cmake)
set(CMAKE_CUDA_COMPILER  "/usr/local/cuda/bin/nvcc")
set(CUDA_HOME /usr/local/cuda-11.6)
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
set(CMAKE_CXX_STANDARD 14)
set(TORCH_CUDA_ARCH_LIST "8.6")

find_package(Torch REQUIRED)

find_package(OpenCV REQUIRED)

message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
add_definitions(-w) #* 关闭警告
#* /usr/bin/ld: cannot find -lmkl_intel_thread 这个库在下面的路径下
link_directories(/opt/intel/oneapi/mkl/2022.2.1/lib/intel64)

# find_package(Boost REQUIRED)

# if(Boost_FOUND)
#     include_directories(${Boost_INCLUDE_DIRS})
#     MESSAGE( STATUS "Boost_INCLUDE_DIRS = ${Boost_INCLUDE_DIRS}.")
#     MESSAGE( STATUS "Boost_LIBRARIES = ${Boost_LIBRARIES}.")
#     MESSAGE( STATUS "Boost_LIB_VERSION = ${Boost_LIB_VERSION}.")
#     # add_executable(foo foo.cpp)
#     # target_link_libraries (foo ${Boost_LIBRARIES})
# endif()

#* ikdtree
find_package(Eigen3)
include_directories(
    .
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    kd_tree
    tools
    cuda_kdtree
    )
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 --use_fast_math  --maxrregcount=128 --ptxas-options=-v -std=c++14 --compiler-options -Wall -gencode arch=compute_86,code=sm_86 --expt-relaxed-constexpr)
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(SRCS  cuda_py_api.cpp ACMMP.cpp ACMMP.cu  pointcloud_rgbd.cpp pointcloud_rgbd.cu kd_tree/ikd_Tree.cpp)
set(simpletools_src simple_tools.cpp simple_tools.cu kd_tree/ikd_Tree.cpp)
set(cuda_kdtree_src cuda_kdtree/CUDA_KDtree.cpp cuda_kdtree/CUDA_KDtree.cu  cuda_kdtree/KDtree.cpp)

message(STATUS "The TORCH_LIBRARIES: ${TORCH_LIBRARIES}\n")
message(STATUS "The TORCH_PYTHON_LIBRARY:  ${TORCH_PYTHON_LIBRARY}\n")

add_library(acmmppy SHARED ${SRCS})
target_link_libraries(acmmppy ${TORCH_LIBRARIES}   ${OpenCV_LIBS})

add_library(simpletools SHARED  ${cuda_kdtree_src} ${simpletools_src}   )
target_link_libraries(simpletools ${TORCH_LIBRARIES}  ${OpenCV_LIBS} )


# nm 命令，用于读取库被导出的符号 可以用来看看那些函数没有定义
# nm -gDC build/libsimpletools.so | grep -i CUDA_KDTree
# ldd 命令，用于查找某个动态库所依赖的库是否存在
# ldd -r ACMMP_net/build/libacmmppy.so
# ldd -r  ACMMP_net/build/libsimpletools.so
# nm -gDC ACMMP_net/build/libsimpletools.so | grep -i  CUDA_KDTree
# nm -gDC ACMMP_net/build/libacmmppy.so | grep -i  imread
# nm -u ACMMP_net/build/libacmmppy.so 
