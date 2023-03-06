#ifndef _MY_TOOL_GPU_H
#define _MY_TOOL_GPU_H

#include <torch/serialize/tensor.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <vector_types.h>


#include <vector>
#include <iostream>
#include <cstdarg>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>


#define CH_CUDA_SAFE_CALL( call) {                                    \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    } \
}

#define CUDA_SAFE_CALL(call) CH_CUDA_SAFE_CALL(call)

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
    float edge_thred=0.9; //* ncc 计算是edge阈值
    int patch_size=10;
    int stage_nsrc=0;
    int ncc_mode=0; //* 0:ncc; 1:weight-ncc; 2:bilateral-ncc
    float kpts_detect_edge_thred=0.9;
    int skip_lines=1; //* 关键点检测时跳过的行
    int kpt_radius=5; //* 按行半径5个像素内edge最大的为关键点
    int kpts_suppresion_radius=20;  //* 按行半径20个像素内只有一个关键点
    Params(){}
    // Params(int batch)
    // {
    //     this->b=batch;
    // }
};

void struct_test(struct Params params);

//* 图像分割
int image_seg_cpp(int b, int h, int w, int p, at::Tensor image_tensor, at::Tensor mask_tensor);
void image_seg_cuda(int b, int h, int w, int p, const float * image, float * mask);

//* 关键点提取
int kpts_selector_cpp(int b, int h, int w, int p, at::Tensor image_tensor, at::Tensor mask_tensor);
void kpts_selector_cuda(int b, int h, int w, int p, const float * image, float * mask);

//* 深度估计
int kpts_depth_pred_cpp(
    const int npts, const int ndepth, const int nsrc, const int height, const int width,
    at::Tensor ref_img_tensor, at::Tensor src_imgs_tensor, 
    at::Tensor ref_edge_map_tensor, at::Tensor src_edge_maps_tensor,
    at::Tensor ref_kpts_tensor, at::Tensor pts_srcs_img_tensor, 
    at::Tensor depth_pred_tensor, at::Tensor depth_conf_tensor);
void kpts_depth_pred_cuda(
    const int npts, const int ndepth, const int nsrc, const int height, const int width,
    const float * ref_img, const float * src_imgs, 
    const float * ref_edge_map, const float * src_edge_maps,
    const float * ref_kpts, const float * pts_srcs_img, 
    float * depth_pred, float * depth_conf);

//* (使用结构体)深度估计
int kpts_depth_pred_struct_cpp(
    struct Params params,
    at::Tensor ref_img_tensor, at::Tensor src_imgs_tensor, 
    at::Tensor ref_edge_map_tensor, at::Tensor src_edge_maps_tensor,
    at::Tensor ref_kpts_tensor, at::Tensor pts_srcs_img_tensor, 
    at::Tensor depth_pred_tensor, at::Tensor depth_conf_tensor);
void kpts_depth_pred_struct_cuda(
    struct Params params,
    const float * ref_img, const float * src_imgs, 
    const float * ref_edge_map, const float * src_edge_maps,
    const float * ref_kpts, const float * pts_srcs_img, 
    float * depth_pred, float * depth_conf);

//* 计算ncc
int kpts_depth_ncc_cpp(
    struct Params params,
    at::Tensor ref_img_tensor, at::Tensor src_imgs_tensor, 
    at::Tensor ref_edge_map_tensor, at::Tensor src_edge_maps_tensor,
    at::Tensor ref_kpts_tensor, at::Tensor pts_srcs_img_tensor, 
    at::Tensor kpts_depth_edge_tensor, at::Tensor kpts_depth_ncc_tensor);
void kpts_depth_ncc_cuda(
    struct Params params,
    const float * ref_img, const float * src_imgs, 
    const float * ref_edge_map, const float * src_edge_maps,
    const float * ref_kpts, const float * pts_srcs_img, 
    float * kpts_depth_edge, float * kpts_depth_ncc);

//* 关键点检测 代替前面的关键点提取
int kpts_selector_cpp(int b, int h, int w, int p, at::Tensor image_tensor, at::Tensor mask_tensor);
void kpts_selector_cuda(int b, int h, int w, int p, const float * image, float * mask);
int kpts_detector_cpp(
    struct Params params,
    at::Tensor ref_edge_map_tensor, at::Tensor kpts_mask_tensor);
void kpts_detector_cuda(
    struct Params params,
    const float * ref_edge_map, float * kpts_mask);

#endif
