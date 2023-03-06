#ifndef _ACMMP_H_
#define _ACMMP_H_

#include "main.h"

int readDepthDmb(const std::string file_path, cv::Mat_<float> &depth);
int readNormalDmb(const std::string file_path, cv::Mat_<cv::Vec3f> &normal);
int writeDepthDmb(const std::string file_path, const cv::Mat_<float> depth);
int writeNormalDmb(const std::string file_path, const cv::Mat_<cv::Vec3f> normal);

Camera ReadCamera(const std::string &cam_path);
void  RescaleImageAndCamera(cv::Mat_<cv::Vec3b> &src, cv::Mat_<cv::Vec3b> &dst, cv::Mat_<float> &depth, Camera &camera);
float3 Get3DPointonWorld(const int x, const int y, const float depth, const Camera camera);
void ProjectonCamera(const float3 PointX, const Camera camera, float2 &point, float &depth);
float GetAngle(const cv::Vec3f &v1, const cv::Vec3f &v2);
void StoreColorPlyFileBinaryPointCloud (const std::string &plyFilePath, const std::vector<PointList> &pc);

void RunJBU(const cv::Mat_<float>  &scaled_image_float, const cv::Mat_<float> &src_depthmap, const std::string &dense_folder , const Problem &problem);

#define CUDA_SAFE_CALL(error) CudaSafeCall(error, __FILE__, __LINE__)
#define CUDA_CHECK_ERROR() CudaCheckError(__FILE__, __LINE__)

void CudaSafeCall(const cudaError_t error, const std::string& file, const int line);
void CudaCheckError(const char* file, const int line);

void RunPatchMatch_cuda(std::vector<Camera> cameras, cudaTextureObjects *texture_objects_cuda, Camera *cameras_cuda, float4 *plane_hypotheses_cuda,  float4 *scaled_plane_hypotheses_cuda, float *costs_cuda,  float *pre_costs_cuda,  curandState *rand_states_cuda, unsigned int *selected_views_cuda, float4 *prior_planes_cuda, unsigned int *plane_masks_cuda, const PatchMatchParams params, cudaTextureObjects *texture_depths_cuda, float4 * plane_hypotheses_host, float * costs_host,float * depths_cuda);

//* 借助初始稀疏深度图寻找合适的点云轮廓
void find_edge_cuda(const Camera camera, float * depths_cuda, float * depth_edge_mask_cuda, float * depth_edge_mask_host);

void CudaRun_cuda(JBUParameters jp_h, JBUParameters * jp_d, JBUTexObj * jt_d, float * depth_d, float * depth_h);

void struct_test(struct Params params);

inline bool file_or_path_exist (const std::string& name);

//* 函数的定义放在类、结构体的前面，cu文件里最好不要有类成员实现。。。



struct ACMMP : torch::CustomClassHolder {
private:
    int num_images;
    std::vector<cv::Mat> images;
    std::vector<cv::Mat> images_orig;
    std::vector<cv::Mat> depths;
    std::vector<Camera> cameras;
    cudaTextureObjects texture_objects_host;
    cudaTextureObjects texture_depths_host;
    float4 *plane_hypotheses_host;
    float4 *scaled_plane_hypotheses_host;
    float *costs_host;
    float *pre_costs_host;
    float4 *prior_planes_host;
    unsigned int *plane_masks_host;
    PatchMatchParams params;

    Camera *cameras_cuda;
    cudaArray *cuArray[MAX_IMAGES];
    cudaArray *cuDepthArray[MAX_IMAGES];
    cudaTextureObjects *texture_objects_cuda;
    cudaTextureObjects *texture_depths_cuda;
    float4 *plane_hypotheses_cuda;
    float4 *scaled_plane_hypotheses_cuda;
    float *costs_cuda;
    float *pre_costs_cuda;
    curandState *rand_states_cuda;
    unsigned int *selected_views_cuda;
    float *depths_cuda;
    float4 *prior_planes_cuda;
    unsigned int *plane_masks_cuda;
    int max_num_downscale;
    std::string dense_folder;
    std::vector<Problem> problems;
    cv::Mat init_depth;
    KD_TREE ikdtree;
    std::shared_ptr<PointCloudXYZINormal>  ref_pts;
    std::shared_ptr<PointCloudXYZINormal>  src_pts;
    std::shared_ptr<PointCloudXYZINormal>  src_pts_updated;
    std::shared_ptr<PointCloudXYZINormal> laserCloudOri;
    std::shared_ptr<PointCloudXYZINormal> coeffSel;
    StatesGroup state;
    std::vector< float > point_selected_surf;
    float m_maximum_pt_kdtree_dis = 0.1;
    float m_maximum_res_dis = 1.0;
    float m_planar_check_dis = 0.05;
    float m_lidar_imu_time_delay = 0;
    float m_long_rang_pt_dis = 500.0;
    //* 点云轮廓提取
    float * depth_edge_mask_cuda;
    float * depth_edge_mask_host;

public:
    ACMMP();
    ~ACMMP();

    void InuputInitialization(const std::string &dense_folder, const std::vector<Problem> &problem, const int idx);
    void Colmap2MVS(const std::string &dense_folder, std::vector<Problem> &problems);
    void CudaSpaceInitialization(const std::string &dense_folder, const Problem &problem);
    void RunPatchMatch();
    void SetGeomConsistencyParams(bool multi_geometry);
    void SetPlanarPriorParams();
    void SetHierarchyParams();

    int GetReferenceImageWidth();
    int GetReferenceImageHeight();
    cv::Mat GetReferenceImage();
    float4 GetPlaneHypothesis(const int index);
    float GetCost(const int index);
    void GetSupportPoints(std::vector<cv::Point>& support2DPoints);
    std::vector<Triangle> DelaunayTriangulation(const cv::Rect boundRC, const std::vector<cv::Point>& points);
    float4 GetPriorPlaneParams(const Triangle triangle, const cv::Mat_<float> depths);
    float GetDepthFromPlaneParam(const float4 plane_hypothesis, const int x, const int y);
    float GetMinDepth();
    float GetMaxDepth();
    void CudaPlanarPriorInitialization(const std::vector<float4> &PlaneParams, const cv::Mat_<float> &masks);
    void acmmp_init_test(const std::string &dense_folder, const int64_t ref_idx, const torch::Tensor &image_torch, const torch::Tensor & K_torch, const torch::Tensor & ext_torch, const torch::Tensor & depth_torch);
    void Run();
    void ProcessProblem(bool geom_consistency, bool planar_prior, bool hierarchy, bool multi_geometrty=false);
    void InuputInitialization_simple();
    void JointBilateralUpsampling();
    void RunFusion(bool geom_consistency=false);
    torch::Tensor GetDepth();
    torch::Tensor GetCosts();
    void kdtree_test();
    void build_basic_tree(const torch::Tensor & basic_pts); //* 构建kdtree
    torch::Tensor align_pts(const torch::Tensor & src_pts, const torch::Tensor & T);
    void findCorrespondingSurfFeatures(int iterCount, bool rematch_en, StatesGroup &lio_state);
    void pointBodyToWorld(PointType const *const pi, PointType *const po, StatesGroup &lio_state);
    torch::Tensor GetUsedMask();
    torch::Tensor GetDepthEdge();
    

};

// void CudaRun_cuda(JBUParameters jp_h, JBUParameters * jp_d, JBUTexObj * jt_d, float * depth_d, float * depth_h);
class JBU {
public:
    JBU();
    ~JBU();

    // Host Parameters
    float *depth_h;
    JBUTexObj jt_h;
    JBUParameters jp_h;

    // Device Parameters
    float *depth_d;
    cudaArray *cuArray[JBU_NUM]; // The first for reference image, and the second for stereo depth image
    JBUTexObj *jt_d;
    JBUParameters *jp_d;

    void InitializeParameters(int n);
    void CudaRun();
    
};

#endif // _ACMMP_H_
