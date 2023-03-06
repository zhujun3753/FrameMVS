from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

if __name__ == '__main__':
    #* 应对之前找不到 Python.h 
    include_dirs = ['/home/zhujun/anaconda3/envs/bnv_fusion/include/python3.9', '/usr/local/cuda/include/']
    include_dirs = []
    print(include_dirs)
    extra_compile_args = {'cxx': ['-w']} #* no warning
    setup(
        name='self-image-cpp-cuda-tool',
        version='0.0.0',
        description='Python tools for image processing implemented by cpp cuda.',
        install_requires=[
            'numpy',
            'torch>=1.1',
        ],
        author='Zhu Jun',
        author_email='zhujun3753@163.com',
        license='Apache License 2.0',
        packages=find_packages(),
        cmdclass={'build_ext': BuildExtension,},
        ext_modules=[
            CUDAExtension(
                name="my_image_cpp_cuda_tool.my_image_tool",
                sources=[
                    "my_image_cpp_cuda_tool/my_tool_api.cpp",
                    "my_image_cpp_cuda_tool/my_tool.cpp",
                    "my_image_cpp_cuda_tool/my_tool_gpu.cu",
                ],
                extra_compile_args=extra_compile_args,
                include_dirs=include_dirs,),
        ],
    )
