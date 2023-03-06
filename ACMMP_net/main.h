#ifndef _MAIN_H_
#define _MAIN_H_

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"

// Includes CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_texture_types.h>
#include <curand_kernel.h>
#include <vector_types.h>

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <map>
#include <memory>
#include "iomanip"

#include <sys/stat.h> // mkdir
#include <sys/types.h> // mkdir

#include <torch/custom_class.h>
#include <torch/script.h>
// #include <torch/extension.h>

//* ikdtree
#include "ikd_Tree.h"

//* 随机数
#include <cstdlib>
//* StatesGroup
#include <tools_eigen.hpp>
#include <so3_math.h>
#include <time.h>


#define MAX_IMAGES 256
#define JBU_NUM 2
#define DIM_OF_STATES (18)
#define INIT_COV (0.0001)


struct vec3f
{
    float x,y,z;

    vec3f(float x_, float y_, float z_) : x(x_) , y(y_), z(z_) {}
    vec3f(float3 v) : x(v.x) , y(v.y), z(v.z) {}
    vec3f(float4 v) : x(v.x) , y(v.y), z(v.z) {}

    friend ostream & operator<<(ostream & os, const vec3f & v)
    {
        std::cout<<v.x<<", "<<v.y<<", "<<v.z<<std::endl;
    }

    float3 cross_product(float3 b)
    {
        float3 c;
        c.x = y * b.z - z * b.y;
        c.y = z * b.x - x * b.z;
        c.z = x * b.y - y * b.x;
        return c;
    }

    float dot_product(float3 b)
    {
        float result = x * b.x + y * b.y + z * b.z;
        return result;
    }

    float norm2()
    {
        float result = sqrt(x * x + y * y + z * z);
        return result;
    }

};

struct StatesGroup
{

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Eigen::Matrix3d rot_end;                                 // [0-2] the estimated attitude (rotation matrix) at the end lidar point
    Eigen::Vector3d pos_end;                                 // [3-5] the estimated position at the end lidar point (world frame)
    Eigen::Vector3d vel_end;                                 // [6-8] the estimated velocity at the end lidar point (world frame)
    Eigen::Vector3d bias_g;                                  // [9-11] gyroscope bias
    Eigen::Vector3d bias_a;                                  // [12-14] accelerator bias
    Eigen::Vector3d gravity;                                 // [15-17] the estimated gravity acceleration

    Eigen::Matrix3d rot_ext_i2c;                             // [18-20] Extrinsic between IMU frame to Camera frame on rotation.
    Eigen::Vector3d pos_ext_i2c;                             // [21-23] Extrinsic between IMU frame to Camera frame on position.
    double          td_ext_i2c_delta;                        // [24]    Extrinsic between IMU frame to Camera frame on position.
    vec_4           cam_intrinsic;                           // [25-28] Intrinsice of camera [fx, fy, cx, cy]
    Eigen::Matrix<double, DIM_OF_STATES, DIM_OF_STATES> cov; // states covariance
    double last_update_time = 0;
    double          td_ext_i2c;
    Eigen::Vector3d Lidar_offset_to_IMU;
    
    StatesGroup()
    {
        rot_end = Eigen::Matrix3d::Identity();
        pos_end = vec_3::Zero();
        vel_end = vec_3::Zero();
        bias_g = vec_3::Zero();
        bias_a = vec_3::Zero();
        gravity = Eigen::Vector3d(0.0, 0.0, 9.805);
        // gravity = Eigen::Vector3d(0.0, 9.805, 0.0);

        //Ext camera w.r.t. IMU
        rot_ext_i2c = Eigen::Matrix3d::Identity();
        pos_ext_i2c = vec_3::Zero();

        cov = Eigen::Matrix<double, DIM_OF_STATES, DIM_OF_STATES>::Identity() * INIT_COV;
        // cov.block(18, 18, 6,6) *= 0.1;
        last_update_time = 0;
        td_ext_i2c_delta = 0;
        td_ext_i2c = 0;
    }

    ~StatesGroup(){}

    StatesGroup operator+(const Eigen::Matrix<double, DIM_OF_STATES, 1> &state_add)
    {
        StatesGroup a = *this;
        // a.rot_end = this->rot_end * Sophus::SO3d::exp(vec_3(state_add(0, 0), state_add(1, 0), state_add(2, 0) ) );
        a.rot_end = this->rot_end * Exp(state_add(0), state_add(1), state_add(2));
        a.pos_end = this->pos_end + state_add.block<3, 1>(3, 0);
        a.vel_end = this->vel_end + state_add.block<3, 1>(6, 0);
        a.bias_g = this->bias_g + state_add.block<3, 1>(9, 0);
        a.bias_a = this->bias_a + state_add.block<3, 1>(12, 0);
        #if ESTIMATE_GRAVITY
            a.gravity = this->gravity + state_add.block<3, 1>(15, 0);
        #endif

        a.cov = this->cov;
        a.last_update_time = this->last_update_time;
        #if ENABLE_CAMERA_OBS                
            //Ext camera w.r.t. IMU
            a.rot_ext_i2c = this->rot_ext_i2c * Exp(  state_add(18), state_add(19), state_add(20) );
            a.pos_ext_i2c = this->pos_ext_i2c + state_add.block<3,1>( 21, 0 );
            a.td_ext_i2c_delta = this->td_ext_i2c_delta + state_add(24);
            a.cam_intrinsic = this->cam_intrinsic + state_add.block(25, 0, 4, 1);
        #endif
        return a;
    }

    StatesGroup &operator+=(const Eigen::Matrix<double, DIM_OF_STATES, 1> &state_add)
    {
        this->rot_end = this->rot_end * Exp(state_add(0, 0), state_add(1, 0), state_add(2, 0));
        this->pos_end += state_add.block<3, 1>(3, 0);
        this->vel_end += state_add.block<3, 1>(6, 0);
        this->bias_g += state_add.block<3, 1>(9, 0);
        this->bias_a += state_add.block<3, 1>(12, 0);
        #if ESTIMATE_GRAVITY
            this->gravity += state_add.block<3, 1>(15, 0);
        #endif
        #if ENABLE_CAMERA_OBS        
            //Ext camera w.r.t. IMU
            this->rot_ext_i2c = this->rot_ext_i2c * Exp(  state_add(18), state_add(19), state_add(20));
            this->pos_ext_i2c = this->pos_ext_i2c + state_add.block<3,1>( 21, 0 );
            this->td_ext_i2c_delta = this->td_ext_i2c_delta + state_add(24);
            this->cam_intrinsic = this->cam_intrinsic + state_add.block(25, 0, 4, 1);   
        #endif
        return *this;
    }

    Eigen::Matrix<double, DIM_OF_STATES, 1> operator-(const StatesGroup &b)
    {
        Eigen::Matrix<double, DIM_OF_STATES, 1> a;
        Eigen::Matrix3d rotd(b.rot_end.transpose() * this->rot_end);
        a.block<3, 1>(0, 0) = SO3_LOG(rotd);
        a.block<3, 1>(3, 0) = this->pos_end - b.pos_end;
        a.block<3, 1>(6, 0) = this->vel_end - b.vel_end;
        a.block<3, 1>(9, 0) = this->bias_g - b.bias_g;
        a.block<3, 1>(12, 0) = this->bias_a - b.bias_a;
        a.block<3, 1>(15, 0) = this->gravity - b.gravity;

        #if ENABLE_CAMERA_OBS    
            //Ext camera w.r.t. IMU
            Eigen::Matrix3d rotd_ext_i2c(b.rot_ext_i2c.transpose() * this->rot_ext_i2c);
            a.block<3, 1>(18, 0) = SO3_LOG(rotd_ext_i2c);
            a.block<3, 1>(21, 0) = this->pos_ext_i2c - b.pos_ext_i2c;
            a(24) = this->td_ext_i2c_delta - b.td_ext_i2c_delta;
            a.block<4, 1>(25, 0) = this->cam_intrinsic - b.cam_intrinsic;
        #endif
        return a;
    }

    static void display(const StatesGroup &state, std::string str = std::string("State: "))
    {
        vec_3 angle_axis = SO3_LOG(state.rot_end) * 57.3;
        std::cout << ANSI_COLOR_GREEN;
        printf("%s \n", str.c_str());
        printf("last_update_time: [%.5f]\n", state.last_update_time);
        printf("angle_axis: (%.5f, %.5f, %.5f)\n", angle_axis(0), angle_axis(1), angle_axis(2));
        printf("pos_end:    (%.5f, %.5f, %.5f)\n", state.pos_end(0), state.pos_end(1), state.pos_end(2));
        printf("vel_end:    (%.5f, %.5f, %.5f)\n", state.vel_end(0), state.vel_end(1), state.vel_end(2));
        printf("gravity:    (%.5f, %.5f, %.5f)\n", state.gravity(0), state.gravity(1), state.gravity(2));
        printf("bias_g:     (%.5f, %.5f, %.5f)\n", state.bias_g(0),  state.bias_g(1),  state.bias_g(2));
        printf("bias_a:     (%.5f, %.5f, %.5f)\n", state.bias_a(0),  state.bias_a(1),  state.bias_a(2));
        std::cout<<"rot_ext_i2c:\n"<<state.rot_ext_i2c<<std::endl;
        std::cout<<"pos_ext_i2c:     "<<state.pos_ext_i2c.transpose()<<std::endl;
        std::cout<<"td_ext_i2c_delta:"<<state.td_ext_i2c_delta<<std::endl;
        std::cout<<"cam_intrinsic:   "<<state.cam_intrinsic.transpose()<<std::endl;
        std::cout<< ANSI_COLOR_RESET << std::endl;

    }
};


struct Camera {
    float K[9];
    float R[9];
    float t[3];
    int height;
    int width;
    float depth_min;
    float depth_max;
};

struct Problem {
    int ref_image_id;
    std::vector<int> src_image_ids;
    int max_image_size = 3200;
    int num_downscale = 0;
    int cur_image_size = 3200;
};

struct Triangle {
    cv::Point pt1, pt2, pt3;
    Triangle (const cv::Point _pt1, const cv::Point _pt2, const cv::Point _pt3) : pt1(_pt1) , pt2(_pt2), pt3(_pt3) {}
};

struct PointList {
    float3 coord;
    float3 normal;
    float3 color;
};

struct cudaTextureObjects {
    cudaTextureObject_t images[MAX_IMAGES];
};

struct PatchMatchParams {
    int max_iterations = 3;
    int patch_size = 11;
    int num_images = 5;
    int max_image_size=3200;
    int radius_increment = 2;
    float sigma_spatial = 5.0f;
    float sigma_color = 3.0f;
    int top_k = 4;
    float baseline = 0.54f;
    float depth_min = 0.0f;
    float depth_max = 1.0f;
    float disparity_min = 0.0f;
    float disparity_max = 1.0f;

    float scaled_cols;
    float scaled_rows;

    bool geom_consistency = false;
    bool planar_prior = false;
    bool multi_geometry = false;
    bool hierarchy = false;
    bool upsample = false;
    bool init_depth_flag = false;
};

struct Params : torch::CustomClassHolder {
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
    void test()
    {
        printf("class test!\n");
        test2();
    }

    void test2()
    {
        printf("class test2!\n");
    }
    // Params(int batch)
    // {
    //     this->b=batch;
    // }
};



struct TexObj {
    cudaTextureObject_t imgs[MAX_IMAGES];
};

struct JBUParameters {
    int height;
    int width;
    int s_height;
    int s_width;
    int Imagescale;
};

struct JBUTexObj {
    cudaTextureObject_t imgs[JBU_NUM];
};

#endif // _MAIN_H_
