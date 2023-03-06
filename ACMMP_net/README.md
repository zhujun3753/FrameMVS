# ACMMP
[News] The code for [ACMH](https://github.com/GhiXu/ACMH) is released!!!  
[News] The code for [ACMM](https://github.com/GhiXu/ACMM) is released!!!  
[News] The code for [ACMP](https://github.com/GhiXu/ACMP) is released!!!
## About
This repository contains the code for Multi-Scale Geometric Consistency Guided and Planar Prior Assisted Multi-View Stereo, which is the extension of ACMM and ACMP.
## Dependencies
The code has been tested on Ubuntu 16.04 with GTX 1080 Ti.  
* [Cuda](https://developer.nvidia.com/zh-cn/cuda-downloads) >= 6.0
* [OpenCV](https://opencv.org/) >= 2.4
* [cmake](https://cmake.org/)
## Usage
* Complie ACMMP
```  
cmake .  
make
```
* Test 
``` 
Use script colmap2mvsnet_acm.py to convert COLMAP SfM result to ACMMP input   
Run ./ACMMP $data_folder to get reconstruction results
```
## Acknowledgemets
This code largely benefits from the following repositories: [Gipuma](https://github.com/kysucix/gipuma) and [COLMAP](https://colmap.github.io/). Thanks to their authors for opening source of their excellent works.

## 问题记录
```bash
# nvcc fatal : unsupported gpu architecture 'compute_89'
export TORCH_CUDA_ARCH_LIST="8.6"

/usr/include/eigen3/Eigen/Core:42:14: fatal error: math_functions.hpp: 没有那个文件或目录
sudo ln -s /usr/local/cuda/include/crt/math_functions.hpp /usr/local/cuda/include/math_functions.hpp

error: token ""__CUDACC_VER__ is no longer supported.
I just comments /usr/local/cuda/include/crt/common_functions.h line 64:
#define __CUDACC_VER__ "__CUDACC_VER__ is no longer supported. Use __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__, and __CUDACC_VER_BUILD__ instead."

/usr/include/eigen3/Eigen/src/Core/arch/CUDA/Half.h(596): error: no suitable constructor exists to convert from "float" to "Eigen::half"
sudo gedit /usr/include/eigen3/Eigen/src/Core/arch/CUDA/Half.h
将596行直接注释

from torch_scatter import scatter_mean
段错误 (核心已转储)
torch_scatter 目前就只支持到11.6 。。。。重装cuda11.6。。。
sudo sh cuda_11.6.2_510.47.03_linux.run
o uninstall the CUDA Toolkit, run cuda-uninstaller in /usr/local/cuda-11.6/bin
***WARNING: Incomplete installation! This installation did not install the CUDA Driver. A driver of version at least 510.00 is required for CUDA 11.6 functionality to work.
To install the driver using this installer, run the following command, replacing <CudaInstaller> with the name of this run file:
    sudo <CudaInstaller>.run --silent --driver

换到11.6之后还是不行，只能下载源码编译。。。
在文件夹： /media/zhujun/0DFD06D20DFD06D2/Documents/pytorch_scatter-master  下
python setup.py install

/usr/include/eigen3/Eigen/Core:42:14: fatal error: math_functions.hpp: 没有那个文件或目录
但实际上后续的eigen已经改为了cuda_runtime.h，编译遇到错误将math_functions.hpp改成cuda_runtime.h即可

/usr/include/eigen3/Eigen/src/Core/util/Macros.h:402:85: error: token ""__CUDACC_VER__ is no longer supported.  Use __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__, and __CUDACC_VER_BUILD__ instead."" is not valid in preprocessor expressions
/usr/include/eigen3/Eigen/Core:232:33: error: token ""__CUDACC_VER__ is no longer supported.  Use __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__, and __CUDACC_VER_BUILD__ instead."" is not valid in preprocessor expressions
/usr/include/eigen3/Eigen/src/Core/arch/CUDA/Half.h:392:63: error: token ""__CUDACC_VER__ is no longer supported.  Use __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__, and __CUDACC_VER_BUILD__ instead."" is not valid in preprocessor expressions
按照要求替换

my_image_cpp_cuda_tool/my_tool.cpp:3:10: fatal error: THC/THC.h: 没有那个文件或目录

```


