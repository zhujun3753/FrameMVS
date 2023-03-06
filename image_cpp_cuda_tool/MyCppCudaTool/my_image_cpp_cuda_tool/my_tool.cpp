#include <torch/serialize/tensor.h>
#include <vector>
// #include <THC/THC.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include "my_tool_gpu.h"


// extern THCState *state;

#define CHECK_CUDA(x) do { \
	  if (!x.type().is_cuda()) { \
		      fprintf(stderr, "%s must be CUDA tensor at %s:%d\n", #x, __FILE__, __LINE__); \
		      exit(-1); \
		    } \
} while (0)
#define CHECK_CONTIGUOUS(x) do { \
	  if (!x.is_contiguous()) { \
		      fprintf(stderr, "%s must be contiguous tensor at %s:%d\n", #x, __FILE__, __LINE__); \
		      exit(-1); \
		    } \
} while (0)
#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)

int image_seg_cpp(int b, int h, int w, int p, at::Tensor image_tensor, at::Tensor mask_tensor) {
    //* batch_size, height, height, patchsize, image_tensor.cuda(), mask_tensor.cuda()

    // 检查输入是否为contiguous的torch.cuda变量
    CHECK_INPUT(image_tensor);
    CHECK_INPUT(mask_tensor);

    // 建立指针
    const float *image = image_tensor.data<float>();
    float *mask = mask_tensor.data<float>();

    // 放入到CUDA中进行具体的算法实现
    image_seg_cuda(b, h, w, p, image, mask);
    
    return 1;
}

void struct_test(struct Params params)
{
    int b =      params.b;
    int h =      params.h;
    int w =      params.w;
    int p =      params.p;
    int npts =   params.npts;
    int ndepth = params.ndepth;
    int nsrc =   params.nsrc;
    int height = params.height;
    int width =  params.width;
    printf("Struct test!!!\n");
    printf("b: %d width: %d\n",b,width);
}

int kpts_selector_cpp(int b, int h, int w, int p, at::Tensor image_tensor, at::Tensor mask_tensor) {
    //* batch_size, height, height, patchsize, image_tensor.cuda(), mask_tensor.cuda()

    // 检查输入是否为contiguous的torch.cuda变量
    CHECK_INPUT(image_tensor);
    CHECK_INPUT(mask_tensor);

    // 建立指针
    const float *image = image_tensor.data<float>();
    float *mask = mask_tensor.data<float>();

    // 放入到CUDA中进行具体的算法实现
    kpts_selector_cuda(b, h, w, p, image, mask);
    
    return 1;
}

int kpts_depth_pred_cpp(
    const int npts, const int ndepth, const int nsrc, const int height, const int width,
    at::Tensor ref_img_tensor, at::Tensor src_imgs_tensor, 
    at::Tensor ref_edge_map_tensor, at::Tensor src_edge_maps_tensor,
    at::Tensor ref_kpts_tensor, at::Tensor pts_srcs_img_tensor, 
    at::Tensor depth_pred_tensor, at::Tensor depth_conf_tensor)
{
    //* batch_size, height, height, patchsize, image_tensor.cuda(), mask_tensor.cuda()

    // 检查输入是否为contiguous的torch.cuda变量
    CHECK_INPUT(ref_img_tensor);
    CHECK_INPUT(src_imgs_tensor);
    CHECK_INPUT(ref_edge_map_tensor);
    CHECK_INPUT(src_edge_maps_tensor);
    CHECK_INPUT(ref_kpts_tensor);
    CHECK_INPUT(pts_srcs_img_tensor);
    CHECK_INPUT(depth_pred_tensor);
    CHECK_INPUT(depth_conf_tensor);


    // 建立指针
    const float *ref_img = ref_img_tensor.data<float>();
    const float *src_imgs = src_imgs_tensor.data<float>();
    const float *ref_edge_map = ref_edge_map_tensor.data<float>();
    const float *src_edge_maps = src_edge_maps_tensor.data<float>();
    const float *ref_kpts = ref_kpts_tensor.data<float>();
    const float *pts_srcs = pts_srcs_img_tensor.data<float>();
    


    float *depth_pred = depth_pred_tensor.data<float>();
    float *depth_conf = depth_conf_tensor.data<float>();


    // 放入到CUDA中进行具体的算法实现
    kpts_depth_pred_cuda(npts, ndepth, nsrc, height, width, 
                        ref_img, src_imgs, ref_edge_map, src_edge_maps, ref_kpts, pts_srcs, depth_pred, depth_conf);
    
    return 1;
}

int kpts_depth_pred_struct_cpp(
    struct Params params,
    at::Tensor ref_img_tensor, at::Tensor src_imgs_tensor, 
    at::Tensor ref_edge_map_tensor, at::Tensor src_edge_maps_tensor,
    at::Tensor ref_kpts_tensor, at::Tensor pts_srcs_img_tensor, 
    at::Tensor depth_pred_tensor, at::Tensor depth_conf_tensor)
{
    //* batch_size, height, height, patchsize, image_tensor.cuda(), mask_tensor.cuda()
    // 检查输入是否为contiguous的torch.cuda变量
    CHECK_INPUT(ref_img_tensor);
    CHECK_INPUT(src_imgs_tensor);
    CHECK_INPUT(ref_edge_map_tensor);
    CHECK_INPUT(src_edge_maps_tensor);
    CHECK_INPUT(ref_kpts_tensor);
    CHECK_INPUT(pts_srcs_img_tensor);
    CHECK_INPUT(depth_pred_tensor);
    CHECK_INPUT(depth_conf_tensor);


    // 建立指针
    const float *ref_img = ref_img_tensor.data<float>();
    const float *src_imgs = src_imgs_tensor.data<float>();
    const float *ref_edge_map = ref_edge_map_tensor.data<float>();
    const float *src_edge_maps = src_edge_maps_tensor.data<float>();
    const float *ref_kpts = ref_kpts_tensor.data<float>();
    const float *pts_srcs = pts_srcs_img_tensor.data<float>();
    


    float *depth_pred = depth_pred_tensor.data<float>();
    float *depth_conf = depth_conf_tensor.data<float>();


    // 放入到CUDA中进行具体的算法实现
    kpts_depth_pred_struct_cuda(params, 
                        ref_img, src_imgs, ref_edge_map, src_edge_maps, ref_kpts, pts_srcs, depth_pred, depth_conf);
    
    return 1;
}

//* 计算ncc
int kpts_depth_ncc_cpp(
    struct Params params,
    at::Tensor ref_img_tensor, at::Tensor src_imgs_tensor, 
    at::Tensor ref_edge_map_tensor, at::Tensor src_edge_maps_tensor,
    at::Tensor ref_kpts_tensor, at::Tensor pts_srcs_img_tensor, 
    at::Tensor kpts_depth_edge_tensor, at::Tensor kpts_depth_ncc_tensor)
{
    //* batch_size, height, height, patchsize, image_tensor.cuda(), mask_tensor.cuda()
    // 检查输入是否为contiguous的torch.cuda变量
    CHECK_INPUT(ref_img_tensor);
    CHECK_INPUT(src_imgs_tensor);
    CHECK_INPUT(ref_edge_map_tensor);
    CHECK_INPUT(src_edge_maps_tensor);
    CHECK_INPUT(ref_kpts_tensor);
    CHECK_INPUT(pts_srcs_img_tensor);
    CHECK_INPUT(kpts_depth_edge_tensor);
    CHECK_INPUT(kpts_depth_ncc_tensor);


    // 建立指针
    const float *ref_img = ref_img_tensor.data<float>();
    const float *src_imgs = src_imgs_tensor.data<float>();
    const float *ref_edge_map = ref_edge_map_tensor.data<float>();
    const float *src_edge_maps = src_edge_maps_tensor.data<float>();
    const float *ref_kpts = ref_kpts_tensor.data<float>();
    const float *pts_srcs = pts_srcs_img_tensor.data<float>();

    float *kpts_depth_edge = kpts_depth_edge_tensor.data<float>();
    float *kpts_depth_ncc = kpts_depth_ncc_tensor.data<float>();


    // 放入到CUDA中进行具体的算法实现
    kpts_depth_ncc_cuda(params, 
                        ref_img, src_imgs, ref_edge_map, src_edge_maps, ref_kpts, pts_srcs, kpts_depth_edge, kpts_depth_ncc);
    
    return 1;
}


//* 关键点检测
int kpts_detector_cpp(
    struct Params params,
    at::Tensor ref_edge_map_tensor, at::Tensor kpts_mask_tensor)
{
    //* batch_size, height, height, patchsize, image_tensor.cuda(), mask_tensor.cuda()
    // 检查输入是否为contiguous的torch.cuda变量
    CHECK_INPUT(ref_edge_map_tensor);
    CHECK_INPUT(kpts_mask_tensor);

    // 建立指针
    const float *ref_edge_map = ref_edge_map_tensor.data<float>();

    float *kpts_mask = kpts_mask_tensor.data<float>();

    // 放入到CUDA中进行具体的算法实现
    kpts_detector_cuda(params, 
                        ref_edge_map, kpts_mask);
    
    return 1;
}

