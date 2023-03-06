#pragma once

#include "main.h"

#include "tools_kd_hash.hpp"
#include "random"

enum pt_attr_set {UNKNOWN=0,GROUND, WALL};

struct time_last
{
    clock_t cur_time;
    void tic()
    {
        cur_time = clock();
    }
    float toc(std::string str="")
    {
        float time_last = (float)(clock()-cur_time)/CLOCKS_PER_SEC;
        std::cout<<str<<" cost time: "<<time_last<<"s"<<std::endl;
        return time_last;
    }
};

float3 cross_product(float3 a, float3 b);
float dot_product(float3 a, float3 b);
float norm2(float3 v);
void R_from_angle_axis(float angle, float3 u, float R[3][3]);
void calculate_matrix(float3 vectorBefore, float3 vectorAfter, float R[3][3]);
void depth_remove_outlier_cuda(float * depth_old, float * depth_new, int width, int height);
void depth_comp_with_attr(float * depth_old, float * depth_new, float3 * plane_params, float * K, int width, int height);
void proj_pts2depth_with_attr(float3 * pts, int * valid_local_ids, int num_pts, float * K, float * T, float * depth, int width, int height, int opertator_flag);
void point_to_plane_dist_cuda(int n_pts, const float3 *ref_pts_cuda, float *dist_cuda, float4 plane_param);
void point_to_plane_sign_dist_cuda(int n_pts, const float3 *ref_pts_cuda, float *dist_cuda, float4 plane_param);
void bilinear_interplote_depth_comp(float * depth, float * K, int width, int height);
void proj_pts2depth(float3 * pts, int num_pts, float * K, float * T, float * depth, int width, int height);
void CheckCUDAError(const char *msg);

// #include "assert.h"
#define R3LIVE_MAP_MAJOR_VERSION 1
#define R3LIVE_MAP_MINOR_VERSION 0

const float process_noise_sigma = 0.1;
const float image_obs_cov = 15;
// extern std::atomic< long > g_pts_index;
class RGB_pts
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    float m_pos[ 3 ] = { 0 }; //* 位置
    float m_rgb[ 3 ] = { 0 }; //* 颜色
    float m_cov_rgb[ 3 ] = { 0 }; //* 颜色协防差
    float plane_param[ 3 ] = { 0, 0, 0}; //* 平面参数 n^T P + 1 = 0
    pt_attr_set pt_attr = UNKNOWN;;
    float m_gray = 0; //* 灰度
    int m_N_gray = 0; //* 灰度更新次数，均值更新
    int m_N_rgb = 0; //* 彩色更新次数
    int m_pt_index = 0; //* 在m_rgb_pts_vec中的索引
    int m_is_out_lier_count = 0; //* 被评为错误点的次数
    cv::Scalar m_dbg_color; //* 随机颜色，为了显示当前被在追踪的点
    float m_obs_dis = 0; //* 最小观测距离
    float m_last_obs_time = 0; //* 上一次被观测时间
    
    void clear()
    {
        cv::RNG g_rng = cv::RNG(0);
        m_rgb[ 0 ] = 0;
        m_rgb[ 1 ] = 0;
        m_rgb[ 2 ] = 0;
        m_gray = 0;
        m_N_gray = 0;
        m_N_rgb = 0;
        m_obs_dis = 0;
        m_last_obs_time = 0;
        int r = g_rng.uniform( 0, 256 );
        int g = g_rng.uniform( 0, 256 );
        int b = g_rng.uniform( 0, 256 );
        m_dbg_color = cv::Scalar( r, g, b );
        // m_rgb = vec_3(255, 255, 255);
    };

    RGB_pts() { clear(); };
    ~RGB_pts(){};

    void set_pos( const vec_3 &pos ) 
    {
        m_pos[0] = pos(0); 
        m_pos[1] = pos(1); 
        m_pos[2] = pos(2); 
    };
    vec_3 get_pos(){ return vec_3(m_pos[0], m_pos[1], m_pos[2]); };
    vec_3 get_rgb(){ return vec_3(m_rgb[0], m_rgb[1], m_rgb[2]); };
    mat_3_3 get_rgb_cov()
    {
        mat_3_3 cov_mat = mat_3_3::Zero();
        for (int i = 0; i < 3; i++)
            cov_mat(i, i) = m_cov_rgb[i];
        return cov_mat;
    };

    void update_gray( const float gray, float obs_dis = 1.0 )
    {
        if (m_obs_dis != 0 && (obs_dis > m_obs_dis * 1.2))
            return;
        m_gray = (m_gray * m_N_gray + gray) / (m_N_gray + 1);
        if (m_obs_dis == 0 || (obs_dis < m_obs_dis))
            m_obs_dis = obs_dis;
        m_N_gray++;
    };

    int update_rgb( const vec_3 &rgb, float obs_dis=0.0, vec_3 obs_sigma = vec_3(image_obs_cov, image_obs_cov, image_obs_cov), float obs_time=0.0 )
    {
        //* 两次观测距离差不能太大
        if (m_obs_dis != 0 && (obs_dis > m_obs_dis * 1.2))
            return 0;

        if( m_N_rgb == 0)
        {
            // For first time of observation.
            m_last_obs_time = obs_time;
            m_obs_dis = obs_dis;
            for (int i = 0; i < 3; i++)
            {
                m_rgb[i] = rgb[i];
                m_cov_rgb[i] = obs_sigma(i) ;
            }
            m_N_rgb = 1;
            return 0;
        }
        // State estimation for robotics, section 2.2.6, page 37-38
        for(int i = 0 ; i < 3; i++)
        {
            m_cov_rgb[i] = (m_cov_rgb[i] + process_noise_sigma * (obs_time - m_last_obs_time)); // Add process noise
            float old_sigma = m_cov_rgb[i];
            m_cov_rgb[i] = sqrt( 1.0 / (1.0 / m_cov_rgb[i] / m_cov_rgb[i] + 1.0 / obs_sigma(i) / obs_sigma(i)) );
            m_rgb[i] = m_cov_rgb[i] * m_cov_rgb[i] * ( m_rgb[i] / old_sigma / old_sigma + rgb(i) / obs_sigma(i) / obs_sigma(i) );
        }

        if (obs_dis < m_obs_dis)
            m_obs_dis = obs_dis;
        m_last_obs_time = obs_time;
        m_N_rgb++;
        return 1;
    };
};
using RGB_pt_ptr = std::shared_ptr< RGB_pts >;

class RGB_Voxel
{
public:
    std::vector< RGB_pt_ptr > m_pts_in_grid;
    float m_last_visited_time = 0;
    vec_3 pose = {0,0,0}; //* 大格子位置，如果这个点的投影不在图像上，就不考虑这个格子中的点，减少遍历
    int num_outlier = 0;
    //* 加入m_voxels_recent_visited_between_keyframes中，但是超出图像边界的次数
    //* 超出次太多就可以认为是不在最近帧视野范围内的了，将其从m_voxels_recent_visited_between_keyframes中去掉
    RGB_Voxel() = default;
    ~RGB_Voxel() = default;
    void add_pt( RGB_pt_ptr &rgb_pts ) 
    { 
        m_pts_in_grid.push_back( rgb_pts ); 
    }
    // 删除点的时候直接用唯一标识删除
    void erase_pt(int m_pt_index)
    {
        for(int i=0; i<m_pts_in_grid.size(); i++)
        {
            if(m_pts_in_grid[i]->m_pt_index == m_pt_index)
            {
                m_pts_in_grid.erase(m_pts_in_grid.begin()+i);
                // m_pts_in_grid[i] = nullptr;
                return;
            }
        }
    }
};

struct Image
{
    float T[4][4];
    std::string filename;
    cv::Mat read_color_img()
    {
        cv::Mat img = cv::imread(filename, cv::IMREAD_COLOR);
        if(!img.data)  //判断是否有数据
        {  
            cout<<"Can not read from: "<<filename<<endl;  
            return cv::Mat();  
        }
        
        return img;
        // IMREAD_UNCHANGED,			//-1   使图像保持原样输出  
        // IMREAD_GRAYSCALE,			//0   把图像转成单通道的灰度图输出
        // IMREAD_COLOR ,				//1	//把图像转成三通道的rgb图输出
        // IMREAD_ANYDEPTH, 			//2   //If set, return 16-bit/32-bit image when the input has the corresponding depth, otherwise convert it to 8-bit.
        // IMREAD_ANYCOLOR	,			//4   //以任何可能的颜色格式读取图像
    }

    cv::Mat read_gray_img()
    {
        cv::Mat img = cv::imread(filename, cv::IMREAD_GRAYSCALE);
        if(!img.data)  //判断是否有数据
        {  
            cout<<"Can not read from: "<<filename<<endl;  
            return cv::Mat();  
        }
        return img;
    }
};

using RGB_voxel_ptr = std::shared_ptr< RGB_Voxel >;
using Voxel_set_iterator = std::unordered_set< std::shared_ptr< RGB_Voxel > >::iterator;
struct DataForReturn
{
    std::vector<float> edge_index_v;
    std::vector<float> inlier_index_v;
    std::vector<float> inlier_index_v_filter;
    // std::vector<float> img_test;
};

struct Global_map : torch::CustomClassHolder
{
    int                                                          m_map_major_version = R3LIVE_MAP_MAJOR_VERSION;
    int                                                          m_map_minor_version = R3LIVE_MAP_MINOR_VERSION;
    int                                                          m_if_get_all_pts_in_boxes_using_mp = 1;
    std::vector< RGB_pt_ptr >                                    m_rgb_pts_vec;
    // std::vector< RGB_pt_ptr >                    m_rgb_pts_in_recent_visited_voxels;
    std::shared_ptr< std::vector< RGB_pt_ptr> >                  m_pts_rgb_vec_for_projection = nullptr;
    std::shared_ptr< std::mutex >                                m_mutex_pts_vec;
    std::shared_ptr< std::mutex >                                m_mutex_recent_added_list;
    std::shared_ptr< std::mutex >                                m_mutex_img_pose_for_projection;
    std::shared_ptr< std::mutex >                                m_mutex_rgb_pts_in_recent_hitted_boxes;
    std::shared_ptr< std::mutex >                                m_mutex_m_box_recent_hitted;
    std::shared_ptr< std::mutex >                                m_mutex_m_voxels_recent_visited_between_keyframes;
    std::shared_ptr< std::mutex >                                m_mutex_pts_last_visited;
    float                                                        m_recent_visited_voxel_activated_time = 0.0;
    bool                                                         m_in_appending_pts = 0;
    int                                                          m_updated_frame_index = 0;
    std::shared_ptr< std::thread >                               m_thread_service;
    int                                                          m_if_reload_init_voxel_and_hashed_pts = true;

    Hash_map_3d< long, RGB_pt_ptr >   m_hashmap_3d_pts;
    Hash_map_3d< long, std::shared_ptr< RGB_Voxel > > m_hashmap_voxels;
    std::unordered_set< std::shared_ptr< RGB_Voxel > > m_voxels_recent_visited;
    //* 记录关键帧之间的最近的访问的大格子，包含跨越多帧的数据，当大格子中的点在当前帧中的投影越界的时候，就可以去掉了
    //* 
    std::unordered_set< std::shared_ptr< RGB_Voxel > > m_voxels_recent_visited_between_keyframes;
    std::vector< std::shared_ptr< RGB_pts > >          m_pts_last_hitted;
    float                                   m_minimum_pts_size = 0.05; // 5cm minimum distance.
    float                                   m_voxel_resolution = 0.1;
    float                                   m_maximum_depth_for_projection = 200;
    float                                   m_minimum_depth_for_projection = 3;
    int                                      m_last_updated_frame_idx = -1;
    //* 相机内参
    float K[3][3];
    //* 图像大小
    int wh[2]={0,0};
    //* 图像信息需要的时候再读取
    std::vector<Image> images;
    bool debug = true;

    std::vector<float3> pts_for_depth;
    std::vector<int> pts_for_depth_ids_in_all;

    KD_TREE ikdtree; //* 这个kdtree还只能定义在这里，不能放在函数实现中，否则直接段错误。。。。。
    std::shared_ptr<PointCloudXYZINormal>  ref_pts;
    float4 ground_plane_param = {0,0,0,0};
    DataForReturn data_for_return;

    torch::Tensor enrich_ground(const torch::Tensor & ground_pts_torch, const torch::Tensor & edge_pts_torch, const torch::Tensor & plane_param_torch);

    torch::Tensor return_data(const std::string & str)
    {
        
        if(str.compare("edge_index_v") == 0 && data_for_return.edge_index_v.size()>0)
        {
            torch::Tensor output = torch::from_blob(data_for_return.edge_index_v.data(), /*sizes=*/{(int)data_for_return.edge_index_v.size(), 1}).clone();
            return output;
        }
        if(str.compare("inlier_index_v") == 0 && data_for_return.inlier_index_v.size()>0)
        {
            torch::Tensor output = torch::from_blob(data_for_return.inlier_index_v.data(), /*sizes=*/{(int)data_for_return.inlier_index_v.size(), 1}).clone();
            return output;
        }
        if(str.compare("inlier_index_v_filter") == 0 && data_for_return.inlier_index_v_filter.size()>0)
        {
            torch::Tensor output = torch::from_blob(data_for_return.inlier_index_v_filter.data(), /*sizes=*/{(int)data_for_return.inlier_index_v_filter.size(), 1}).clone();
            return output;
        }
        torch::Tensor output = torch::empty(0);
        return output;
    }

    torch::Tensor get_ground_plane_param()
    {

        cv::Mat ground_plane_param_cv(4, 1, CV_32FC1, cv::Scalar::all(0));
        ground_plane_param_cv.ptr<float>(0, 0)[0] = ground_plane_param.x;
        ground_plane_param_cv.ptr<float>(1, 0)[0] = ground_plane_param.y;
        ground_plane_param_cv.ptr<float>(2, 0)[0] = ground_plane_param.z;
        ground_plane_param_cv.ptr<float>(3, 0)[0] = ground_plane_param.w;
        torch::Tensor output = torch::from_blob(ground_plane_param_cv.ptr<float>(), /*sizes=*/{4,1}).clone();
        return output;
    }
    torch::Tensor point_cloud_segmentation(const torch::Tensor & image_info_torch, int64_t img_id, bool refresh);
    torch::Tensor get_depth_with_attr(int64_t img_id, bool refresh);
    void ikdtree_test();
    float4 kdtree_fit_plane(float3 *ref_pts_host, int n_pts, double distance_threshold, int64_t ransac_n, int64_t num_iterations);
    torch::Tensor ransac_fit_ground_plane(bool refresh, double distance_threshold, int64_t ransac_n, int64_t num_iterations);
    torch::Tensor bilinear_interplote_depth(const torch::Tensor & depth_torch_in);
    torch::Tensor bilinear_interplote_depth_id(int64_t img_id, bool refresh);
    torch::Tensor get_K()
    {
        if(wh[0]==0)
        {
            std::cout<<" No param"<<std::endl;
            return torch::empty(0);
        }
        cv::Mat K_cv(3, 3, CV_32FC1, cv::Scalar::all(0));
        for(int i=0; i<3; i++)
        {
            for(int j=0; j<3; j++)
            {
                K_cv.ptr<float>(i, j)[0] = K[i][j];
            }
        }
        torch::Tensor output = torch::from_blob(K_cv.ptr<float>(), /*sizes=*/{3,3}).clone();
        return output;
    }
    torch::Tensor get_ext(int64_t img_id);
    torch::Tensor get_image(int64_t img_id);
    void preprocess_for_depth();
    torch::Tensor get_depth(int64_t img_id, bool refresh);
    void delete_pt(float x, float y, float z);
    void add_img(const torch::Tensor & T_torch, const std::string & filename)
    {
        auto shape = T_torch.sizes();
        if(shape[0]!=4 || shape[1]!=4)
        {
            std::cout<<"Wrong shape of pts!!"<<std::endl;
            // return torch::empty(0);
            return;
        }
        Image img;
        torch::Tensor T_torch_tmp = T_torch.clone().to(torch::kCPU).contiguous();
        cv::Mat T_cv = cv::Mat{4, 4, CV_32FC(1), T_torch_tmp.data_ptr<float>()}; //* 3通道float
        for(int i=0; i<4; i++)
        {
            img.T[i][0] = T_cv.ptr<float>(i, 0)[0];
            img.T[i][1] = T_cv.ptr<float>(i, 1)[0];
            img.T[i][2] = T_cv.ptr<float>(i, 2)[0];
            img.T[i][3] = T_cv.ptr<float>(i, 3)[0];
        }
        img.filename = filename;
        images.emplace_back(img);
        if(debug && 0)
        {
            std::cout<<"T:\n";
            for(int i=0; i<4; i++)
            {
                std::cout<<img.T[i][0]<<", "<<img.T[i][1]<<", "<<img.T[i][2]<<", "<<img.T[i][3]<<std::endl;
            }
            std::cout<<"filename: "<<img.filename<<std::endl;
            cv::Mat image = img.read_gray_img(); //* read_color_img
            std::cout << "dims:" << image.dims << std::endl;
            std::cout << "rows:" << image.rows << std::endl;
            std::cout << "cols:" << image.cols << std::endl;
            std::cout << "channels:" << image.channels() << endl;
            cv::imshow("image",image);
            cv::waitKey(0);
        }
        // std::cout<<"current images size: "<<images.size()<<std::endl;
    }
    void set_K_wh(const torch::Tensor & K_torch, double w, double h)
    {
        auto shape = K_torch.sizes();
        if(shape[0]!=3 || shape[1]!=3)
        {
            std::cout<<"Wrong shape of pts!!"<<std::endl;
            // return torch::empty(0);
            return;
        }
        torch::Tensor K_torch_tmp = K_torch.clone().to(torch::kCPU).contiguous();
        cv::Mat K_cv = cv::Mat{3, 3, CV_32FC(1), K_torch_tmp.data_ptr<float>()}; //* 3通道float
        for(int i=0; i<3; i++)
        {
            K[i][0] = K_cv.ptr<float>(i, 0)[0];
            K[i][1] = K_cv.ptr<float>(i, 1)[0];
            K[i][2] = K_cv.ptr<float>(i, 2)[0];
        }
        wh[0] = (int)w;
        wh[1] = (int)h;
        if(debug && 0)
        {
            std::cout<<"K:\n";
            for(int i=0; i<3; i++)
            {
                std::cout<<K[i][0]<<", "<<K[i][1]<<", "<<K[i][2]<<std::endl;
            }
            std::cout<<"wh: "<<wh[0]<<", "<<wh[1]<<std::endl;
        }
    }
    void str_test(const std::string & str)
    {
        std::cout<<str<<std::endl;
    }
    void clear();
    void set_minmum_dis( float minimum_dis );
    Global_map();
    ~Global_map();
    // void service_refresh_pts_for_projection();
    // void render_points_for_projection( std::shared_ptr< Image_frame > &img_ptr );
    // void update_pose_for_projection( std::shared_ptr< Image_frame > &img, float fov_margin = 0.0001 );
    bool is_busy();
    // template < typename T >
    void append_points_to_global_map( const torch::Tensor & src_pts_torch );
    void save_to_pcd( std::string dir_name, std::string file_name = std::string( "/rgb_pt" ) , int save_pts_with_views = 3);
    void save_and_display_pointcloud( std::string dir_name = std::string( "/home/ziv/temp/" ), std::string file_name = std::string( "/rgb_pt" ) ,  int save_pts_with_views = 3);
    torch::Tensor get_pc();
    void set_resolution(double grid_res=0.05, double box_res=0.1)
    {
        std::cout<<"Set new resolution"<<std::endl;
        m_minimum_pts_size = grid_res;
        m_voxel_resolution = box_res;
        cur_resolution();
    }
    void cur_resolution()
    {
        std::cout<<"grid_res: "<<m_minimum_pts_size<<"m, box_res: "<<m_voxel_resolution<<"m.\n";
    }
};

