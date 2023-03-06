## 修改顺序
1. 先在`my_cpp_cuda_tool/my_tool_api.cpp`中添加所需要的函数
    ```c
    m.def("kpts_depth_pred", &kpts_depth_pred_cpp, "kpts_depth_pred");
    ```
2. 在`my_cpp_cuda_tool/my_tool_gpu.h`中给出函数的定义
    ```c
    int kpts_depth_pred_cpp(int b, int h, int w, int p, at::Tensor image_tensor, at::Tensor mask_tensor);
    void kpts_depth_pred_cuda(int b, int h, int w, int p, const float * image, float * mask);
    ```
    在`kpts_depth_pred_cpp`中检查输入合法性，并调用`kpts_depth_pred_cuda`
3. 在`my_cpp_cuda_tool/my_tool.cpp`中给出`kpts_depth_pred_cpp`的定义(包含对`kpts_depth_pred_cuda`的调用)
   ```c
   int kpts_depth_pred_cpp(int b, int h, int w, int p, at::Tensor image_tensor, at::Tensor mask_tensor) {
    //* batch_size, height, height, patchsize, image_tensor.cuda(), mask_tensor.cuda()
    // 检查输入是否为contiguous的torch.cuda变量
    CHECK_INPUT(image_tensor);
    CHECK_INPUT(mask_tensor);
    // 建立指针
    const float *image = image_tensor.data<float>();
    float *mask = mask_tensor.data<float>();
    // 放入到CUDA中进行具体的算法实现
    kpts_depth_pred_cuda(b, h, w, p, image, mask);
    return 1;
    }
   ```
4. 在`my_cpp_cuda_tool/my_tool_gpu.cu`中给出`kpts_depth_pred_cuda`的定义，和关联cuda上执行函数的定义
   ```c
   __global__ void kpts_depth_pred_kernel_cuda(int b,int h, int w, int p, const float *__restrict__ image, float *__restrict__ mask) {
        //* int h, int w, int p, const float * image, float * mask
        // threadIdx是一个三维的向量，可以用.x .y .z分别调用其三个维度。此处我们只初始化了第一个维度为THREADS_PER_BLOCK
        // blockIdx也是三维向量。我们初始化用的DIVUP(m, THREADS_PER_BLOCK), b分别对应blockIdx.x和blockIdx.y
        // blockDim代表block的长度
        //* 只对每行进行处理
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        int col = blockIdx.y * blockDim.y + threadIdx.y;
        int b_idx = blockIdx.z;

        if (row>=h-p || row<p || col>=w-p || col<p || b_idx>=b) 
            return;
        if(row%20>0) 
            return;
        if(mask[b_idx*h*w + row*w+col]<0) 
            return;
        if(col<20 || col>=w-20) 
            return;
        bool left_exist=false;
        bool right_exist=false;
        for (int new_col = col-20; new_col<col; ++new_col)
        {
            if(mask[b_idx*h*w + row*w+new_col]>0) 
                left_exist=true;
        }
        for (int new_col = col+1; new_col<col+20; ++new_col)
        {
            if(mask[b_idx*h*w + row*w+new_col]>0) 
                right_exist=true;
        }
        if(left_exist && right_exist) mask[b_idx*h*w + row*w+col]=-1;
    }


    void kpts_depth_pred_cuda(int b, int h, int w, int p, const float * image, float * mask) {
        // cudaError_t变量用来记录CUDA的err信息，在最后需要check
        cudaError_t err;
        // divup定义在cuda_utils.h,DIVUP(m, t)相当于把m个点平均划分给t个block中的线程，每个block可以处理THREADS_PER_BLOCK个线程。
        // THREADS_PER_BLOCK=256，假设我有m=1024个点，那就是我需要4个block，一共256*4个线程去处理这1024个点。
        dim3 blocks(DIVUP(h, 16),DIVUP(w, 16),b);  // blockIdx.x(col), blockIdx.y(row)
        dim3 threads(16,16); //* x*y*z<=1024 !!

        // 可函数需要用<<<blocks, threads>>> 去指定调用的块数和线程数，总共调用的线程数为blocks*threads
        // std::cout<<"kpts here"<<std::endl;
        kpts_selector_kernel_cuda<<<blocks, threads>>>(b, h, w, p, image, mask);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());

        kpts_suppresion_kernel_cuda<<<blocks, threads>>>(b, h, w, p, image, mask);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        
        // 如果cuda操作错误，则打印错误信息
        err = cudaGetLastError();
        if (cudaSuccess != err) {
            fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
            exit(-1);
        }
    }
   ```
5. 然后编译即可
   ```bash
   python setup.py develop
   https://blog.csdn.net/qq_39031960/article/details/106211695
   CUDAExtension(
            'roi_align.crop_and_resize_gpu',
            ['roi_align/src/crop_and_resize_gpu.cpp',
             'roi_align/src/cuda/crop_and_resize_kernel.cu'],
            include_dirs=include_dirs,
            extra_compile_args={'cxx': ['-g', '-fopenmp'],
                                'nvcc': ['-O2']}
            
        )
   ```
6. 树结构
   ```bash
   .
    ├── build
    │   ├── lib.linux-x86_64-3.8
    │   │   └── my_cpp_cuda_tool
    │   │       └── my_tool.cpython-38-x86_64-linux-gnu.so
    │   └── temp.linux-x86_64-3.8
    │       └── my_cpp_cuda_tool
    │           ├── my_tool_api.o
    │           ├── my_tool_gpu.o
    │           └── my_tool.o
    ├── my_cpp_cuda_tool
    │   ├── cuda_utils.h
    │   ├── __init__.py
    │   ├── my_tool_api.cpp
    │   ├── my_tool.cpp
    │   ├── my_tool.cpython-38-x86_64-linux-gnu.so
    │   ├── my_tool_gpu.cu
    │   ├── my_tool_gpu.h
    │   └── __pycache__
    │       └── __init__.cpython-38.pyc
    ├── python_code
    │   ├── edge_maps.npy
    │   └── my_tool_example.py
    ├── readMe.md
    ├── run.sh
    ├── self_cpp_cuda_tool.egg-info
    │   ├── dependency_links.txt
    │   ├── PKG-INFO
    │   ├── requires.txt
    │   ├── SOURCES.txt
    │   └── top_level.txt
    └── setup.py
   ```

## 添加参数
1. 在`my_cpp_cuda_tool/my_tool_gpu.h`的`struct Params`中添加需要的参数
   ```c
   struct Params {
    int b=0;
    int h=0;
    int w=0;
    int p=0;
    int npts=0;
    int ndepth=0;
    int nsrc=0;
    int height=0;
    int width=0;
    Params(){}
    // Params(int batch)
    // {
    //     this->b=batch;
    // }
    };
   ```
2. 在`my_cpp_cuda_tool/my_tool_api.cpp`的`PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)`中进行必要配置即可
   ```c
   py::class_<Params>(m, "Params")  
    .def(py::init())
    .def_readwrite("b",         &Params::b)
    .def_readwrite("h",         &Params::h)
    .def_readwrite("w",         &Params::w)
    .def_readwrite("p",         &Params::p)
    .def_readwrite("npts",      &Params::npts)
    .def_readwrite("ndepth",    &Params::ndepth)
    .def_readwrite("nsrc",      &Params::nsrc)
    .def_readwrite("height",    &Params::height)
    .def_readwrite("width",     &Params::width);  
   ```