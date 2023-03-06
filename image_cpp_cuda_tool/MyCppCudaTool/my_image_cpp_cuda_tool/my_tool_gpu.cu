#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "my_tool_gpu.h"
#include "cuda_utils.h"


//* （1）__host__ int foo(int a){}与C或者C++中的foo(int a){}相同，是由CPU调用，由CPU执行的函数，__host__可缺省。
//* （2）__global__ int foo(int a){}表示一个内核函数，是一组由GPU执行的并行计算任务，以foo<<>>(a)的形式或者driver API的形式调用。
//*     目前__global__函数必须由CPU调用，并将并行计算任务发射到GPU的任务调用单元。随着GPU可编程能力的进一步提高，未来可能可以由GPU调用。
//* （3）__device__ int foo(int a){}则表示一个由GPU中一个线程调用的函数。由于Tesla架构的GPU允许线程调用函数，
//*     因此实际上是将__device__ 函数以__inline形式展开后直接编译到二进制代码中实现的，并不是真正的函数。

// CUDA使用__global__来定义kernel
__global__ void init_seg_kernel_cuda(int b,int h, int w, int p, const float *__restrict__ image, float *__restrict__ mask) {
        //* int h, int w, int p, const float * image, float * mask

    // threadIdx是一个三维的向量，可以用.x .y .z分别调用其三个维度。此处我们只初始化了第一个维度为THREADS_PER_BLOCK
    // blockIdx也是三维向量。我们初始化用的DIVUP(m, THREADS_PER_BLOCK), b分别对应blockIdx.x和blockIdx.y
    // blockDim代表block的长度
    //* 只对每行进行处理
    int b_idx = blockIdx.y;
    int row =blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= h-p || row<p || b_idx>=b) return;
    // if(row>=h) return;
    // p=0;
    // if(row==1)
    //     printf("col: %d",w);

    for (int col = 0+p; col < w-p; ++col)
    {
        // mask[row*w+col]=image[row*w+col];
        if(image[b_idx*h*w + row*w+col]>0)
        {
            if(mask[b_idx*h*w + row*w+col-1]>-1)
            {
                mask[b_idx*h*w + row*w+col]=mask[b_idx*h*w + row*w+col-1];
            }
            else
            {
                mask[b_idx*h*w + row*w+col]=float(b_idx*h*w + row*w+col);
            }
        }
    }
}

__global__ void label_combine_down(int b, int row, int h, int w, int p, const float *__restrict__ image, float *__restrict__ mask) {
    //* int h, int w, int p, const float * image, float * mask
    int b_idx = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= w-p || col<p || b_idx>=b) return;
    if(image[b_idx*h*w + row*w+col]<1) return;

    float new_label=mask[b_idx*h*w + row*w+col];
    for (int new_col = col; new_col<w-p; ++new_col)
    {
        if(image[b_idx*h*w + row*w+new_col]<1) break;
        if(mask[b_idx*h*w + row*w+new_col-w]>0 && mask[b_idx*h*w + row*w+new_col-w]<new_label)
        {
            new_label=mask[b_idx*h*w + row*w+new_col-w];
        }
    }
    for (int new_col = col; new_col>=p; --new_col)
    {
        if(image[b_idx*h*w + row*w+new_col]<1) break;
        if(mask[b_idx*h*w + row*w+new_col-w]>0 && mask[b_idx*h*w + row*w+new_col-w]<new_label)
        {
            new_label=mask[b_idx*h*w + row*w+new_col-w];
        }
    }
    if(new_label<mask[b_idx*h*w + row*w+col]) mask[b_idx*h*w + row*w+col]=new_label;
}

__global__ void label_combine_up(int b, int row, int h, int w, int p, const float *__restrict__ image, float *__restrict__ mask) {
    //* int h, int w, int p, const float * image, float * mask
    int b_idx = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= w-p || col<p || b_idx>=b) return;
    if(image[b_idx*h*w + row*w+col]<1) return;

    float new_label=mask[b_idx*h*w + row*w+col];
    if(mask[b_idx*h*w + row*w+col+w]==mask[b_idx*h*w + row*w+col]) return;

    for (int new_col = col; new_col<w-p; ++new_col)
    {
        if(image[b_idx*h*w + row*w+new_col]<1) break;
        if(mask[b_idx*h*w + row*w+new_col+w]>0 && mask[b_idx*h*w + row*w+new_col+w]<new_label)
        {
            new_label=mask[b_idx*h*w + row*w+new_col+w];
        }
    }
    for (int new_col = col; new_col>=p; --new_col)
    {
        if(image[b_idx*h*w + row*w+new_col]<1) break;
        if(mask[b_idx*h*w + row*w+new_col+w]>0 && mask[b_idx*h*w + row*w+new_col+w]<new_label)
        {
            new_label=mask[b_idx*h*w + row*w+new_col+w];
        }
    }
    if(new_label<mask[b_idx*h*w + row*w+col]) mask[b_idx*h*w + row*w+col]=new_label;
}

void image_seg_cuda(int b, int h, int w, int p, const float * image, float * mask) {
        
    // cudaError_t变量用来记录CUDA的err信息，在最后需要check
    cudaError_t err;
    // divup定义在cuda_utils.h,DIVUP(m, t)相当于把m个点平均划分给t个block中的线程，每个block可以处理THREADS_PER_BLOCK个线程。
    // THREADS_PER_BLOCK=256，假设我有m=1024个点，那就是我需要4个block，一共256*4个线程去处理这1024个点。
    dim3 blocks(DIVUP(h, THREADS_PER_BLOCK),b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    // std::cout<<"images seg"<<std::endl;
    // 可函数需要用<<<blocks, threads>>> 去指定调用的块数和线程数，总共调用的线程数为blocks*threads
    //* 初步分类
    init_seg_kernel_cuda<<<blocks, threads>>>(b, h, w, p, image, mask);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    //* 类别合并
    dim3 blocks2(DIVUP(w, THREADS_PER_BLOCK),b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads2(THREADS_PER_BLOCK);
    for(int row=0+p;row<h-p;row++)
    {
        label_combine_down<<<blocks2, threads2>>>(b, row, h, w, p, image, mask);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }
    for(int row=h-p-1;row>=p;row--)
    {
        label_combine_up<<<blocks2, threads2>>>(b, row, h, w, p, image, mask);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }
    
    // 如果cuda操作错误，则打印错误信息
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

__global__ void kpts_selector_kernel_cuda(int b,int h, int w, int p, const float *__restrict__ image, float *__restrict__ mask) {
        //* int h, int w, int p, const float * image, float * mask
    // threadIdx是一个三维的向量，可以用.x .y .z分别调用其三个维度。此处我们只初始化了第一个维度为THREADS_PER_BLOCK
    // blockIdx也是三维向量。我们初始化用的DIVUP(m, THREADS_PER_BLOCK), b分别对应blockIdx.x和blockIdx.y
    // blockDim代表block的长度
    //* 只对每行进行处理
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int b_idx = blockIdx.z;

    if (row>=h-p || row<p || col>=w-p || col<p || b_idx>=b) return;
    if(row%20>0) return;
    // if(row%10>0) return;

    if(image[b_idx*h*w + row*w+col]<0.9) return;
    float cur_v=image[b_idx*h*w + row*w+col];
    bool change=false;
    for (int new_col = col-5; new_col<col+5; ++new_col)
    {
        if(image[b_idx*h*w + row*w+new_col]>cur_v) change=true;
    }
    if(!change) mask[b_idx*h*w + row*w+col]=1.0;
}

__global__ void kpts_suppresion_kernel_cuda(int b,int h, int w, int p, const float *__restrict__ image, float *__restrict__ mask) {
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

void kpts_selector_cuda(int b, int h, int w, int p, const float * image, float * mask) {
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

__global__ void calculate_score_of_each_depth_kernel_cuda(
    const int npts, const int ndepth, const int nsrc, const int height, const int width,
    const float * __restrict__ ref_img, const float * __restrict__ src_imgs, 
    const float * __restrict__ ref_edge_map, const float * __restrict__ src_edge_maps,
    const float * __restrict__ ref_kpts, const float * __restrict__ pts_srcs_img, 
    float * __restrict__ depth_pred, float * __restrict__ depth_conf)
{
    //* 第 pts_i 个关键点，第 depth_i 个深度
    int pts_i = blockIdx.x * blockDim.x + threadIdx.x;
    int depth_i = blockIdx.y * blockDim.y + threadIdx.y;
    //* 取参考图中第 pts_i 个关键点对应的图像块与该点在第 depth_i 个深度下在每张源图中的投影位置的图像块计算NCC

    if (pts_i>=npts || depth_i>=ndepth) 
        return;
    //* 关键点在 ref 中的坐标
    int ref_i_id=pts_i*2;
    int2 ref_pt=make_int2((int)round(ref_kpts[ref_i_id+1]),(int)round(ref_kpts[ref_i_id+0]));
    float scores[32]={0.0f}; //* 假设最多不超过32个源图
    float all_score=0.0f;
    float edge_map_thred=0.8; //! 总觉得0.9有点高，后面再调整
    for(int nsrc_i=0; nsrc_i<nsrc; nsrc_i++)
    {
        int src_i_id=pts_i*ndepth*nsrc*2 + depth_i*nsrc*2 + nsrc_i*2;
        int2 src_i_pt=make_int2((int)round(pts_srcs_img[src_i_id+1]),(int)round(pts_srcs_img[src_i_id+0]));
        if (src_i_pt.x>=width || src_i_pt.x<0 || src_i_pt.y>=height || src_i_pt.y<0)
        {
            scores[nsrc_i]=0.0f;
            continue;
        }
        // printf("nsrc_i: %d, src_i_pt.y: %d, src_i_pt.x: %d\n",nsrc_i,src_i_pt.y,src_i_pt.x);
        if(src_edge_maps[nsrc_i*height*width+src_i_pt.y*width+src_i_pt.x]<edge_map_thred)
        {
            scores[nsrc_i]=0.0f;
            continue;
        }
        else
        {
            scores[nsrc_i]=src_edge_maps[nsrc_i*height*width+src_i_pt.y*width+src_i_pt.x];
            //* 计算NCC
        }
        all_score+=scores[nsrc_i];
    }
    all_score/=nsrc;
    if(all_score>0.4)
        depth_conf[pts_i*ndepth + depth_i]=all_score;

}

__global__ void depth_pred_kernel_cuda(
    const int npts, const int ndepth, const int nsrc, const int height, const int width,
    const float * __restrict__ ref_img, const float * __restrict__ src_imgs, 
    const float * __restrict__ ref_edge_map, const float * __restrict__ src_edge_maps,
    const float * __restrict__ ref_kpts, const float * __restrict__ pts_srcs_img, 
    float * __restrict__ depth_pred, float * __restrict__ depth_conf)
{
    //* 单独处理每个点的所有深度
    int pts_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (pts_i>=npts) 
        return;
    for(int ndepth_i=0; ndepth_i<ndepth; ndepth_i++)
    {
        if(depth_conf[pts_i*ndepth + ndepth_i]==0.0f)
            continue;
        
    }
}


void kpts_depth_pred_cuda(
    const int npts, const int ndepth, const int nsrc, const int height, const int width,
    const float * ref_img, const float * src_imgs, 
    const float * ref_edge_map, const float * src_edge_maps,
    const float * ref_kpts, const float * pts_srcs_img, 
    float * depth_pred, float * depth_conf)
{
    // cudaError_t变量用来记录CUDA的err信息，在最后需要check
    cudaError_t err;
    // std::cout<<"kpts here"<<std::endl;
    //* calculate score of each depth
    //* if score is zero, the depth could be ignored.
    dim3 blocks(DIVUP(npts, 16),DIVUP(ndepth, 32));
    dim3 threads(16,32); //* x*y*z<=1024 !!
    calculate_score_of_each_depth_kernel_cuda<<<blocks, threads>>>(npts, ndepth, nsrc, height, width, 
                                            ref_img, src_imgs, ref_edge_map, src_edge_maps, ref_kpts, pts_srcs_img, depth_pred, depth_conf);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    //* depth_pred
    dim3 blocks2(DIVUP(npts, 256));
    dim3 threads2(256);
    depth_pred_kernel_cuda<<<blocks2, threads2>>>(npts, ndepth, nsrc, height, width, 
                                            ref_img, src_imgs, ref_edge_map, src_edge_maps, ref_kpts, pts_srcs_img, depth_pred, depth_conf);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    
    // 如果cuda操作错误，则打印错误信息
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}



// !

__global__ void calculate_score_of_each_depth_struct_kernel_cuda(
    struct Params params,
    const float * __restrict__ ref_img, const float * __restrict__ src_imgs, 
    const float * __restrict__ ref_edge_map, const float * __restrict__ src_edge_maps,
    const float * __restrict__ ref_kpts, const float * __restrict__ pts_srcs_img, 
    float * __restrict__ depth_pred, float * __restrict__ depth_conf)
{

    //* 第 pts_i 个关键点，第 depth_i 个深度
    int pts_i = blockIdx.x * blockDim.x + threadIdx.x;
    int depth_i = blockIdx.y * blockDim.y + threadIdx.y;
    //* 取参考图中第 pts_i 个关键点对应的图像块与该点在第 depth_i 个深度下在每张源图中的投影位置的图像块计算NCC
    int npts=params.npts;
    int ndepth=params.ndepth;
    int nsrc=params.nsrc;
    int height=params.height;
    int width=params.width;
    int stage_nsrc=params.stage_nsrc;

    if (pts_i>=npts || depth_i>=ndepth) 
        return;
    //* 关键点在 ref 中的坐标
    int ref_i_id=pts_i*2;
    int2 ref_pt=make_int2((int)round(ref_kpts[ref_i_id+1]),(int)round(ref_kpts[ref_i_id+0]));
    float all_score=0.0f;
    float edge_map_thred=0.8; //! 总觉得0.9有点高，后面再调整
    int stages=nsrc/stage_nsrc;
    for(int stage_i=0; stage_i<stages; stage_i++)
    {
        for(int nsrc_i=stage_i*stage_nsrc; nsrc_i<(stage_i+1)*stage_nsrc; nsrc_i++)
        {
            int src_i_id=pts_i*ndepth*nsrc*2 + depth_i*nsrc*2 + nsrc_i*2;
            int2 src_i_pt=make_int2((int)round(pts_srcs_img[src_i_id+1]),(int)round(pts_srcs_img[src_i_id+0]));
            if (src_i_pt.x>=width || src_i_pt.x<0 || src_i_pt.y>=height || src_i_pt.y<0)
            {
                continue;
            }
            // printf("nsrc_i: %d, src_i_pt.y: %d, src_i_pt.x: %d\n",nsrc_i,src_i_pt.y,src_i_pt.x);
            if(src_edge_maps[nsrc_i*height*width+src_i_pt.y*width+src_i_pt.x]<edge_map_thred)
            {
                continue;
            }
            else
            {
                all_score+=src_edge_maps[nsrc_i*height*width+src_i_pt.y*width+src_i_pt.x];
            }
        }
        all_score/=stage_nsrc;
        if(all_score>0.4)
        {
            depth_conf[pts_i*ndepth*stages + depth_i*stages + stage_i]=all_score;
        }
        all_score=0.0f;
    }

}

__global__ void depth_pred_struct_kernel_cuda(
    struct Params params,
    const float * __restrict__ ref_img, const float * __restrict__ src_imgs, 
    const float * __restrict__ ref_edge_map, const float * __restrict__ src_edge_maps,
    const float * __restrict__ ref_kpts, const float * __restrict__ pts_srcs_img, 
    float * __restrict__ depth_pred, float * __restrict__ depth_conf)
{
    //* 单独处理每个点的所有深度
    int pts_i = blockIdx.x * blockDim.x + threadIdx.x;
    int npts=params.npts;
    int ndepth=params.ndepth;
    int nsrc=params.nsrc;
    int height=params.height;
    int width=params.width;
    int stage_nsrc=params.stage_nsrc;
    if (pts_i>=npts) 
        return;
    // for(int ndepth_i=0; ndepth_i<ndepth; ndepth_i++)
    // {
    //     if(depth_conf[pts_i*ndepth + ndepth_i]==0.0f)
    //         continue;
        
    // }
}


__global__ void calculate_ncc_of_each_depth_each_stage_kernel_cuda(
    struct Params params,
    const float * __restrict__ ref_img, const float * __restrict__ src_imgs, 
    const float * __restrict__ ref_edge_map, const float * __restrict__ src_edge_maps,
    const float * __restrict__ ref_kpts, const float * __restrict__ pts_srcs_img, 
    float * __restrict__ depth_pred, float * __restrict__ depth_conf)
{

    //* 第 pts_i 个关键点，第 depth_i 个深度
    int pts_i = blockIdx.x * blockDim.x + threadIdx.x;
    int depth_i = blockIdx.y * blockDim.y + threadIdx.y;
    int stage_i = blockIdx.z;
    //* 取参考图中第 pts_i 个关键点对应的图像块与该点在第 depth_i 个深度下在每张源图中的投影位置的图像块计算NCC
    int npts=params.npts;
    int ndepth=params.ndepth;
    int nsrc=params.nsrc;
    int height=params.height;
    int width=params.width;
    int stage_nsrc=params.stage_nsrc;
    int stages=nsrc/stage_nsrc;

    if (pts_i>=npts || depth_i>=ndepth || stage_i>= stages) 
        return;
    if(depth_conf[pts_i*ndepth*stages + depth_i*stages + stage_i]<=0.4)
        return;
    //* 关键点在 ref 中的坐标
    int ref_i_id=pts_i*2;
    int2 ref_pt=make_int2((int)round(ref_kpts[ref_i_id+1]),(int)round(ref_kpts[ref_i_id+0]));
    int patch_size=10;
    if(ref_pt.x-patch_size<0 || ref_pt.y-patch_size<0 || ref_pt.x+patch_size>=width || ref_pt.y+patch_size>=height)
    {
        depth_conf[pts_i*ndepth*stages + depth_i*stages + stage_i]=0.0f;
        return;
    }
    //* 计算当前patch的均值和方差
    //! 这里用个稀疏的也行，不一定要这么稠密
    int num_pts=(2*patch_size+1)*(2*patch_size+1);
    float ref_mean=0.0f;
    float ref_std=0.0f;
    float ref_p[512]={0.0f}; //* 假设不超过512个点
    if(num_pts>512)
    {
        printf("Too large size of patch");
        return;
    }
    int p_count=0;
    //* E(X)
    for(int row=ref_pt.y-patch_size; row<=ref_pt.y+patch_size; row++)
        for(int col=ref_pt.x-patch_size; col<=ref_pt.x+patch_size; col++)
        {
            ref_p[p_count]=ref_img[row*width+col];
            ref_mean+=ref_p[p_count];
            p_count++;
        }
    ref_mean/=num_pts;
    //* std(X)
    for(int p_count_i=0; p_count_i<p_count; p_count_i++)
        ref_std+=pow(ref_p[p_count_i]-ref_mean,2);
    ref_std=sqrt(ref_std/num_pts);

    float ncc[32]={0.0f}; //* 假设最多不超过32个源图
    for(int nsrc_i=stage_i*stage_nsrc; nsrc_i<(stage_i+1)*stage_nsrc; nsrc_i++)
    {
        int src_i_id=pts_i*ndepth*nsrc*2 + depth_i*nsrc*2 + nsrc_i*2;
        int2 src_i_pt=make_int2((int)round(pts_srcs_img[src_i_id+1]),(int)round(pts_srcs_img[src_i_id+0]));
        if(src_i_pt.x-patch_size<0 || src_i_pt.y-patch_size<0 || src_i_pt.x+patch_size>=width || src_i_pt.y+patch_size>=height)
            continue;
        float src_mean=0.0f;
        float src_std=0.0f;
        float ref_src_mean=0.0f;
        float src_p[512]={0.0f};
        p_count=0;
        //* src 均值 E(Y)
        for(int row=src_i_pt.y-patch_size; row<=src_i_pt.y+patch_size; row++)
            for(int col=src_i_pt.x-patch_size; col<=src_i_pt.x+patch_size; col++)
            {
                src_p[p_count]=src_imgs[nsrc_i*height*width+row*width+col];
                src_mean+=src_p[p_count];
                p_count++;
            }
        src_mean/=num_pts;
        //* src 标准差 std(Y)
        for(int p_count_i=0; p_count_i<p_count; p_count_i++)
            src_std+=pow(src_p[p_count_i]-src_mean,2);
        src_std=sqrt(src_std/num_pts);
        //* E(XY)
        for(int p_count_i=0; p_count_i<p_count; p_count_i++)
            ref_src_mean+=ref_p[p_count_i]*src_p[p_count_i];
        ref_src_mean/=num_pts;
        //* ncc = cov(X,Y)/std(X)/std(Y) = E(X-EX)(Y-EY) / std(X)/std(Y) = (EXY - EXEY)  / std(X)/std(Y) 
        if(ref_std<1e-5f || src_std<1e-5f || ref_src_mean-ref_mean*src_mean<0)
            ncc[nsrc_i-stage_i*stage_nsrc]=0.0f;
        else
            ncc[nsrc_i-stage_i*stage_nsrc]=(ref_src_mean-ref_mean*src_mean)/ref_std/src_std;
    }
    float score=1.0f;
    for(int stage_nsrc_i=0; stage_nsrc_i<stage_nsrc; stage_nsrc_i++)
        score*=ncc[stage_nsrc_i];
    depth_conf[pts_i*ndepth*stages + depth_i*stages + stage_i]=score;
}

void kpts_depth_pred_struct_cuda(
    struct Params params,
    const float * ref_img, const float * src_imgs, 
    const float * ref_edge_map, const float * src_edge_maps,
    const float * ref_kpts, const float * pts_srcs_img, 
    float * depth_pred, float * depth_conf)
{
    // const int npts, const int ndepth, const int nsrc, const int height, const int width,

    // cudaError_t变量用来记录CUDA的err信息，在最后需要check
    cudaError_t err;
    // std::cout<<"kpts here"<<std::endl;
    //* calculate score of each depth
    //* if score is zero, the depth could be ignored.
    dim3 blocks(DIVUP(params.npts, 16),DIVUP(params.ndepth, 32));
    dim3 threads(16,32); //* x*y*z<=1024 !!
    calculate_score_of_each_depth_struct_kernel_cuda<<<blocks, threads>>>(params, 
                                            ref_img, src_imgs, ref_edge_map, src_edge_maps, ref_kpts, pts_srcs_img, depth_pred, depth_conf);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    //* calculate ncc of each depth each stage
    dim3 blocks_ncc(DIVUP(params.npts, 16),DIVUP(params.ndepth, 32),DIVUP(params.nsrc, params.stage_nsrc));
    dim3 threads_ncc(16,32); //* x*y*z<=1024 !!
    calculate_ncc_of_each_depth_each_stage_kernel_cuda<<<blocks_ncc, threads_ncc>>>(params, 
                                            ref_img, src_imgs, ref_edge_map, src_edge_maps, ref_kpts, pts_srcs_img, depth_pred, depth_conf);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    //* depth_pred
    dim3 blocks2(DIVUP(params.npts, 256));
    dim3 threads2(256);
    depth_pred_struct_kernel_cuda<<<blocks2, threads2>>>(params,
                                            ref_img, src_imgs, ref_edge_map, src_edge_maps, ref_kpts, pts_srcs_img, depth_pred, depth_conf);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    
    // 如果cuda操作错误，则打印错误信息
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}


//* 计算ncc
__global__ void kpts_depth_ncc_kernel_cuda(
    struct Params params,
    const float * __restrict__ ref_img, const float * __restrict__ src_imgs, 
    const float * __restrict__ ref_edge_map, const float * __restrict__ src_edge_maps,
    const float * __restrict__ ref_kpts, const float * __restrict__ pts_srcs_img, 
    float * __restrict__ kpts_depth_edge, float * __restrict__ kpts_depth_ncc)
{
    //* 第 pts_i 个关键点，第 depth_i 个深度 第nsrc_i个源图
    int pts_i = blockIdx.x * blockDim.x + threadIdx.x;
    int depth_i = blockIdx.y * blockDim.y + threadIdx.y;
    int nsrc_i = blockIdx.z;
    //* 取参考图中第 pts_i 个关键点对应的图像块与该点在第 depth_i 个深度下在每张源图中的投影位置的图像块计算NCC
    int npts=params.npts;
    int ndepth=params.ndepth;
    int nsrc=params.nsrc;
    int height=params.height;
    int width=params.width;
    int stage_nsrc=params.stage_nsrc;
    int patch_size=params.patch_size;
    float edge_thred=params.edge_thred;
    int ncc_mode=params.ncc_mode;


    if (pts_i>=npts || depth_i>=ndepth || nsrc_i>= nsrc) 
        return;
    //* 关键点在 ref 中的坐标
    int ref_i_id=pts_i*2;
    int2 ref_pt=make_int2((int)round(ref_kpts[ref_i_id+1]),(int)round(ref_kpts[ref_i_id+0]));
    //* 关键点在 nsrc_i 中的坐标
    int src_i_id=pts_i*ndepth*nsrc*2 + depth_i*nsrc*2 + nsrc_i*2;
    int2 src_i_pt=make_int2((int)round(pts_srcs_img[src_i_id+1]),(int)round(pts_srcs_img[src_i_id+0]));
    //* 顺便保存edge
    if(src_i_pt.x<0 || src_i_pt.y<0 || src_i_pt.x>=width || src_i_pt.y>=height)
        kpts_depth_edge[pts_i*ndepth*nsrc + depth_i*nsrc + nsrc_i]=0.0f;
    else
        kpts_depth_edge[pts_i*ndepth*nsrc + depth_i*nsrc + nsrc_i]=src_edge_maps[nsrc_i*height*width + src_i_pt.y*width + src_i_pt.x];

    //* 保证patch在图像上
    if(ref_pt.x-patch_size<0 || ref_pt.y-patch_size<0 || ref_pt.x+patch_size>=width || ref_pt.y+patch_size>=height)
    {
        kpts_depth_ncc[pts_i*ndepth*nsrc + depth_i*nsrc + nsrc_i]=0.0f;
        return;
    }
    if(src_i_pt.x-patch_size<0 || src_i_pt.y-patch_size<0 || src_i_pt.x+patch_size>=width || src_i_pt.y+patch_size>=height)
    {
        kpts_depth_ncc[pts_i*ndepth*nsrc + depth_i*nsrc + nsrc_i]=0.0f;
        return;
    }
    //* 如果edge太小则直接返回
    if(src_edge_maps[nsrc_i*height*width + src_i_pt.y*width + src_i_pt.x]<edge_thred)
    {
        kpts_depth_ncc[pts_i*ndepth*nsrc + depth_i*nsrc + nsrc_i]=0.0f;
        return;
    }

    //* 计算当前patch的均值和方差
    //! 这里用个稀疏的也行，不一定要这么稠密
    int num_pts=0;
    float ref_mean=0.0f;
    float ref_std=0.0f;
    float ref_p[512]={0.0f}; //* 假设不超过512个点
    if(num_pts>512)
    {
        printf("Too large size of patch");
        return;
    }
    //* E(X)
    for(int row=ref_pt.y-patch_size; row<=ref_pt.y+patch_size; row++)
        for(int col=ref_pt.x-patch_size; col<=ref_pt.x+patch_size; col++)
        {
            ref_p[num_pts]=ref_img[row*width+col];
            ref_mean+=ref_p[num_pts];
            num_pts++;
        }
    ref_mean/=num_pts;
    //* std(X)
    for(int num_pts_i=0; num_pts_i<num_pts; num_pts_i++)
        ref_std+=pow(ref_p[num_pts_i]-ref_mean,2);
    ref_std=sqrt(ref_std/num_pts);
    
    float src_mean=0.0f;
    float src_std=0.0f;
    float ref_src_mean=0.0f;
    float src_p[512]={0.0f};
    
    int p_count=0;
    //* src 均值 E(Y)
    for(int row=src_i_pt.y-patch_size; row<=src_i_pt.y+patch_size; row++)
        for(int col=src_i_pt.x-patch_size; col<=src_i_pt.x+patch_size; col++)
        {
            src_p[p_count]=src_imgs[nsrc_i*height*width+row*width+col];
            src_mean+=src_p[p_count];
            p_count++;
        }
    src_mean/=num_pts;
    //* src 标准差 std(Y)
    for(int num_pts_i=0; num_pts_i<num_pts; num_pts_i++)
        src_std+=pow(src_p[num_pts_i]-src_mean,2);
    src_std=sqrt(src_std/num_pts);
    //* E(XY)
    for(int num_pts_i=0; num_pts_i<num_pts; num_pts_i++)
        ref_src_mean+=ref_p[num_pts_i]*src_p[num_pts_i];
    ref_src_mean/=num_pts;
    //* ncc = cov(X,Y)/std(X)/std(Y) = E(X-EX)(Y-EY) / std(X)/std(Y) = (EXY - EXEY)  / std(X)/std(Y) 
    if(ref_std<1e-5f || src_std<1e-5f || ref_src_mean-ref_mean*src_mean<0)
        kpts_depth_ncc[pts_i*ndepth*nsrc + depth_i*nsrc + nsrc_i]=0.0f;
    else
    {
        float weight=1.0f;
        if (ncc_mode==1)
            weight=exp(-0.1*abs(ref_mean-src_mean));
            weight*=exp(-0.1*abs(ref_std-src_std));
        kpts_depth_ncc[pts_i*ndepth*nsrc + depth_i*nsrc + nsrc_i]=weight*(ref_src_mean-ref_mean*src_mean)/ref_std/src_std;
    }

}

void kpts_depth_ncc_cuda(
    struct Params params,
    const float * ref_img, const float * src_imgs, 
    const float * ref_edge_map, const float * src_edge_maps,
    const float * ref_kpts, const float * pts_srcs_img, 
    float * kpts_depth_edge, float * kpts_depth_ncc)
{
    cudaError_t err;
    //* calculate ncc of each depth each stage
    dim3 blocks_ncc(DIVUP(params.npts, 16),DIVUP(params.ndepth, 32),params.nsrc);
    dim3 threads_ncc(16,32); //* x*y*z<=1024 !!
    kpts_depth_ncc_kernel_cuda<<<blocks_ncc, threads_ncc>>>(params, 
                                            ref_img, src_imgs, ref_edge_map, src_edge_maps, ref_kpts, pts_srcs_img, kpts_depth_edge, kpts_depth_ncc);
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}


//* 关键点检测
__global__ void kpts_detector_kernel_cuda(struct Params params, const float *__restrict__ ref_edge_map, float *__restrict__ kpts_mask) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int b_idx = blockIdx.z;
    int h=params.h;
    int w=params.w;
    int p=params.p;
    int b=params.b;
    int skip_lines=params.skip_lines;
    float thred=params.kpts_detect_edge_thred;
    int kpt_radius=params.kpt_radius;

    if (row>=h-p || row<p || col>=w-p || col<p || b_idx>=b) return;
    if(row%skip_lines>0) return;

    if(ref_edge_map[b_idx*h*w + row*w+col]<thred) return;
    float cur_v=ref_edge_map[b_idx*h*w + row*w+col];
    bool change=false;
    //* 左右各kpt_radius个像素返回内的最大值
    for (int new_col = col-kpt_radius; new_col<col+kpt_radius; ++new_col)
    {
        if(ref_edge_map[b_idx*h*w + row*w+new_col]>cur_v) change=true;
    }
    if(!change) kpts_mask[b_idx*h*w + row*w+col]=1.0;
}

__global__ void kpts_detector_suppresion_kernel_cuda(struct Params params, const float *__restrict__ ref_edge_map, float *__restrict__ kpts_mask) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int b_idx = blockIdx.z;
    int h=params.h;
    int w=params.w;
    int p=params.p;
    int b=params.b;
    int skip_lines=params.skip_lines;
    float thred=params.kpts_detect_edge_thred;
    int kpt_radius=params.kpt_radius;
    int kpts_suppresion_radius=params.kpts_suppresion_radius;

    if (row>=h-p || row<p || col>=w-p || col<p || b_idx>=b) 
        return;
    if(row%skip_lines>0) 
        return;
    if(kpts_mask[b_idx*h*w + row*w+col]<0) 
        return;
    if(col<kpts_suppresion_radius || col>=w-kpts_suppresion_radius) 
        return;
    bool left_exist=false;
    bool right_exist=false;
    for (int new_col = col-kpts_suppresion_radius; new_col<col; ++new_col)
    {
        if(kpts_mask[b_idx*h*w + row*w+new_col]>0) 
            left_exist=true;
    }
    for (int new_col = col+1; new_col<col+kpts_suppresion_radius; ++new_col)
    {
        if(kpts_mask[b_idx*h*w + row*w+new_col]>0) 
            right_exist=true;
    }
    if(left_exist && right_exist) kpts_mask[b_idx*h*w + row*w+col]=-1;
}

void kpts_detector_cuda(
    struct Params params,
    const float * ref_edge_map, float * kpts_mask)
{
    // cudaError_t变量用来记录CUDA的err信息，在最后需要check
    cudaError_t err;
    // divup定义在cuda_utils.h,DIVUP(m, t)相当于把m个点平均划分给t个block中的线程，每个block可以处理THREADS_PER_BLOCK个线程。
    // THREADS_PER_BLOCK=256，假设我有m=1024个点，那就是我需要4个block，一共256*4个线程去处理这1024个点。
    dim3 blocks(DIVUP(params.h, 16),DIVUP(params.w, 16),params.b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(16,16); //* x*y*z<=1024 !!

    // 可函数需要用<<<blocks, threads>>> 去指定调用的块数和线程数，总共调用的线程数为blocks*threads
    // std::cout<<"kpts here"<<std::endl;
    kpts_detector_kernel_cuda<<<blocks, threads>>>(params, ref_edge_map, kpts_mask);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    kpts_detector_suppresion_kernel_cuda<<<blocks, threads>>>(params, ref_edge_map, kpts_mask);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    
    // 如果cuda操作错误，则打印错误信息
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}