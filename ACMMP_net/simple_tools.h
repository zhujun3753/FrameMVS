
#include "main.h"
#include <cstdarg>

//* PCL
// #include <pcl/ModelCoefficients.h>
// #include <pcl/common/io.h>
// #include <pcl/io/ply_io.h>
// #include <pcl/features/normal_3d.h>
// #include <pcl/features/normal_3d_omp.h>
// #include <pcl/features/principal_curvatures.h>
// #include <pcl/filters/extract_indices.h>
// #include <pcl/filters/voxel_grid.h>
// #include <pcl/kdtree/kdtree_flann.h>
// #include <pcl/sample_consensus/ransac.h>
// #include <pcl/sample_consensus/sac_model_plane.h>
// #include <pcl/search/kdtree.h>
// #include <pcl/segmentation/sac_segmentation.h>
// #include <pcl_conversions/pcl_conversions.h>


#define CUDA_SAFE_CALL(error) CudaSafeCall(error, __FILE__, __LINE__)
#define CUDA_CHECK_ERROR() CudaCheckError(__FILE__, __LINE__)

void CudaSafeCall(const cudaError_t error, const std::string& file, const int line);
void CudaCheckError(const char* file, const int line);

//* uvz 转换为深度图（python循环太慢。。。。）
torch::Tensor proj2depth(const torch::Tensor & uv_valid, const torch::Tensor & z_valid, const torch::Tensor & depth);

//* 深度补全
torch::Tensor depth_interp_comp(const torch::Tensor & image_info_torch, const torch::Tensor & cam_param_torch);

void depth_interp_comp_cuda(int width, int height, const float * depth_cuda, float * color_smooth_cuda, float3 * normal_cuda, float * depth_new_cuda, curandState *rand_states_cuda, float3 * cam_pts_cuda, float3 * esti_normal_cuda, float * curve_cuda, float * cam_param_cuda);

//* 深度滤波
torch::Tensor depth_filter(const torch::Tensor & image_info_torch);

void depth_filter_cuda(int width, int height, const float * depth_cuda, float * depth_new_cuda, curandState *rand_states_cuda);


//* uvz normal 转换为深度图和法线
torch::Tensor proj_depth_normal(const torch::Tensor & uvd_normal_torch, int64_t width, int64_t height);

//* 随机采样一致
torch::Tensor ransac_plane(const torch::Tensor & basic_pts, double distance_threshold, int64_t ransac_n, int64_t num_iterations);

void ransac_plane_cal_dist_cuda(int n_pts, const float3 * ref_pts_cuda, float * dist_cuda, float4 plane_param);

void cuda_kdtree_test();

