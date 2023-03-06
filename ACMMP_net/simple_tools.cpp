
#include "simple_tools.h"
#include<set>
#include "random"

//* cuda kdtree
#include <cstdio>
#include <vector>
#include <cstdlib>
#include <float.h>
#include <sys/time.h>
#include <iostream>
#include "KDtree.h"
#include "CUDA_KDtree.h"

using Time = std::chrono::time_point<std::chrono::high_resolution_clock>;

double TimeDiff(timeval t1, timeval t2)
{
    double t;
    t = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
    t += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms

    return t;
}

void cuda_kdtree_test()
{
    KDtree tree;
    CUDA_KDTree GPU_tree;
    timeval t1, t2;
    int max_tree_levels = 13; // play around with this value to get the best result
    vector <Point> data(100000);
    vector <Point> queries(100000);
    vector <int> gpu_indexes, cpu_indexes;
    vector <float> gpu_dists, cpu_dists;
    std::cout<<"创建数据"<<std::endl;
    for(unsigned int i=0; i < data.size(); i++) {
        data[i].coords[0] = 0 + 100.0*(rand() / (1.0 + RAND_MAX));
        data[i].coords[1] = 0 + 100.0*(rand() / (1.0 + RAND_MAX));
        data[i].coords[2] = 0 + 100.0*(rand() / (1.0 + RAND_MAX));
    }
    for(unsigned int i=0; i < queries.size(); i++) {
        queries[i].coords[0] = 0 + 100.0*(rand() / (1.0 + RAND_MAX));
        queries[i].coords[1] = 0 + 100.0*(rand() / (1.0 + RAND_MAX));
        queries[i].coords[2] = 0 + 100.0*(rand() / (1.0 + RAND_MAX));
    }
    std::cout<<"CreateKDTree"<<std::endl;
    // Time to create the tree
    gettimeofday(&t1, NULL);
    tree.Create(data, max_tree_levels);
    GPU_tree.CreateKDTree(tree.GetRoot(), tree.GetNumNodes(), data);
    gettimeofday(&t2, NULL);
    double gpu_create_time = TimeDiff(t1,t2);
    std::cout<<"Search"<<std::endl;
    // Time to search the tree
    gettimeofday(&t1, NULL);
    GPU_tree.Search(queries, gpu_indexes, gpu_dists);
    gettimeofday(&t2, NULL);
    double gpu_search_time = TimeDiff(t1,t2);
    std::cout<<"GPU_tree.Search end"<<std::endl;
    printf("Points in the tree: %ld\n", data.size());
    printf("Query points: %ld\n", queries.size());
    printf("GPU max tree depth: %d\n", max_tree_levels);
    printf("GPU create + search: %g + %g = %g ms\n", gpu_create_time, gpu_search_time, gpu_create_time + gpu_search_time);
}

Time start_timer() 
{
    return std::chrono::high_resolution_clock::now();
}

double get_elapsed_time(Time start) 
{
    Time end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> d = end - start;
    std::chrono::microseconds us = std::chrono::duration_cast<std::chrono::microseconds>(d);
    return us.count() / 1000.0f;
}

void StringAppendV(std::string *dst, const char *format, va_list ap)
{
    // First try with a small fixed size buffer.
    static const int kFixedBufferSize = 1024;
    char fixed_buffer[kFixedBufferSize];

    // It is possible for methods that use a va_list to invalidate
    // the data in it upon use.  The fix is to make a copy
    // of the structure before using it and use that copy instead.
    va_list backup_ap;
    va_copy(backup_ap, ap);
    int result = vsnprintf(fixed_buffer, kFixedBufferSize, format, backup_ap);
    va_end(backup_ap);

    if (result < kFixedBufferSize)
    {
        if (result >= 0)
        {
            // Normal case - everything fits.
            dst->append(fixed_buffer, result);
            return;
        }

#ifdef _MSC_VER
        // Error or MSVC running out of space.  MSVC 8.0 and higher
        // can be asked about space needed with the special idiom below:
        va_copy(backup_ap, ap);
        result = vsnprintf(nullptr, 0, format, backup_ap);
        va_end(backup_ap);
#endif

        if (result < 0)
        {
            // Just an error.
            return;
        }
    }

    // Increase the buffer size to the size requested by vsnprintf,
    // plus one for the closing \0.
    const int variable_buffer_size = result + 1;
    std::unique_ptr<char> variable_buffer(new char[variable_buffer_size]);

    // Restore the va_list before we use it again.
    va_copy(backup_ap, ap);
    result =
        vsnprintf(variable_buffer.get(), variable_buffer_size, format, backup_ap);
    va_end(backup_ap);

    if (result >= 0 && result < variable_buffer_size)
    {
        dst->append(variable_buffer.get(), result);
    }
}

std::string StringPrintf(const char *format, ...)
{
    va_list ap;
    va_start(ap, format);
    std::string result;
    StringAppendV(&result, format, ap);
    va_end(ap);
    return result;
}

void CudaSafeCall(const cudaError_t error, const std::string &file, const int line)
{
    if (error != cudaSuccess)
    {
        std::cerr << StringPrintf("%s in %s at line %i", cudaGetErrorString(error), file.c_str(), line) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void CudaCheckError(const char *file, const int line)
{
    cudaError error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cerr << StringPrintf("cudaCheckError() failed at %s:%i : %s", file,
                                  line, cudaGetErrorString(error))
                  << std::endl;
        exit(EXIT_FAILURE);
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    error = cudaDeviceSynchronize();
    if (cudaSuccess != error)
    {
        std::cerr << StringPrintf("cudaCheckError() with sync failed at %s:%i : %s",
                                  file, line, cudaGetErrorString(error))
                  << std::endl;
        std::cerr
            << "This error is likely caused by the graphics card timeout "
               "detection mechanism of your operating system. Please refer to "
               "the FAQ in the documentation on how to solve this problem."
            << std::endl;
        exit(EXIT_FAILURE);
    }
}

void solve_linear(int N, float a[3][4])
{
    int k = 0, i = 0, r = 0, j = 0;
    float t;
    //* 行变换求对角矩阵，NxN的矩阵只需要对N-1列处理即可，最后一列不处理
    for (k = 0; k < N - 1; k++)
    {
        //* 寻找当前列对角元及其以下元素中的绝对值最大值，避免用0
        r = k;
        for (i = k+1; i < N; i++)
        {
            if (fabs(a[i][k]) > fabs(a[r][k]))
            {
                r = i;
            }
        }
        if (a[r][k] == 0) //* 如果当前列对角元及其以下元素全为0，则无法计算
        {
            break; //* 无解
        }
        //* 交换行，将当前列元素绝对值最大值移到对角元位置，不用考虑上面和左边的，因为已经处理过了，当前列对角元及其以下元素的左边全为零
        for (j = k; j < N + 1; j++)
        {
            t = a[r][j];
            a[r][j] = a[k][j];
            a[k][j] = t;
        }
        //* 行变换，用当前列的对角元a[k][k]消掉对角元以下的值
        //* 这里就不用计算当前列了，因为当前列以下的元素用不到了，直接处理后面列即可
        for (i = k + 1; i < N; i++)
        {
            for (j = k + 1; j < N + 1; j++)
            {
                a[i][j] = a[i][j] - a[i][k] / a[k][k] * a[k][j];
            }
        }
    }
    //* 从最后一列的对角元开始，计算最终的结果
    float he = 0;
    for (k = N - 1; k >= 0; k--)
    {
        he = 0;
        for (j = k + 1; j < N; j++)
        {
            he = he + a[k][j] * a[j][N];
        }
        a[k][N] = (a[k][N] - he) / a[k][k];
    }
}

void cal_plane_param(const float3 * selected_pts, int N, float M[3])
{
    float c[3][4];
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            c[i][j] = 0;
        }
    }
    c[0][0] = N;
    for (int i = 0; i < N; i++)
    {
        c[0][1] = c[0][1] + selected_pts[i].x;
        c[0][2] = c[0][2] + selected_pts[i].y;
        c[0][3] = c[0][3] + selected_pts[i].z;
        c[1][1] = c[1][1] + selected_pts[i].x * selected_pts[i].x;
        c[1][2] = c[1][2] + selected_pts[i].x * selected_pts[i].y;
        c[1][3] = c[1][3] + selected_pts[i].x * selected_pts[i].z;
        c[2][2] = c[2][2] + selected_pts[i].y * selected_pts[i].y;
        c[2][3] = c[2][3] + selected_pts[i].y * selected_pts[i].z;
    }
    c[1][0] = c[0][1];
    c[2][0] = c[0][2];
    c[2][1] = c[1][2];

    solve_linear(3, c);

    for (int i = 0; i < 3; i++)
    {
        M[i] = c[i][3];
    }
}

torch::Tensor ransac_plane_bak(const torch::Tensor & basic_pts, double distance_threshold, int64_t ransac_n, int64_t num_iterations)
{
    // To recreate the same random numbers across runs of the program, set seed to a specific
    // number instead of a number from random_device
    std::random_device rd;
    uint32_t seed = rd();
    std::mt19937 rnd(seed);  // mersenne_twister_engine
    clock_t start,end, start_fit,end_fit; //定义clock_t变量
    start = clock(); //开始时间
    bool debug = true;
    srand((unsigned)time(NULL));
    // std::cout << (rand()/double(RAND_MAX)) << " "; //生成[0,1]范围内的随机数
    //* basic_pts [N ,3+n]
    auto shape = basic_pts.sizes();
    if(shape.size()!=2 || shape[1]<3)
    {
        std::cout<<"Wrong shape of pts!!"<<std::endl;
        return torch::empty(0);
    }
    int n_all_pts = shape[0];
    int ch = shape[1];
    torch::Tensor basic_pts_torch = basic_pts.clone().to(torch::kCPU).contiguous();
    //* 转为opencv数据 下面并不是复制，而是引用！！！！用这种方式访问比较快
    cv::Mat basic_pts_cv = cv::Mat{n_all_pts, 1, CV_32FC(ch), basic_pts_torch.data_ptr<float>()}; //* 3通道float
    if(n_all_pts<NUM_MATCH_POINTS)
    {
        std::cout<<"Too few pts!!"<<std::endl;
        return torch::empty(0);
    }
    //* 直通滤波，为了提取地面,提取地面之后，再将地面用kdtree表示，直接处理点云
    int n_pts = 0;
    std::vector<float3> ref_pts_host, all_pts_host;
    std::vector<int> ref_pts_ids;
    for(int i=0; i<n_all_pts; i++)
    {
        float3 pt;
        pt.x = basic_pts_cv.ptr<float>(i, 0)[0];
        pt.y = basic_pts_cv.ptr<float>(i, 0)[1];
        pt.z = basic_pts_cv.ptr<float>(i, 0)[2];
        all_pts_host.emplace_back(pt);
        if(basic_pts_cv.ptr<float>(i, 0)[2]<1.5)
        {
            ref_pts_host.emplace_back(pt);
            ref_pts_ids.emplace_back(i);
            n_pts++;
        }
    }

    // start = clock(); //开始时间
    // float3 * ref_pts_host, * selected_pts;
    float3 * selected_pts;
    float3 * ref_pts_cuda;
    float * dist_host;
    float * dist_cuda;
    cudaMalloc((void **)&ref_pts_cuda,    sizeof(float3)  * (n_all_pts));
    cudaMalloc((void **)&dist_cuda,       sizeof(float)   * (n_all_pts));
    dist_host = new float[n_all_pts];
    selected_pts = new float3[ransac_n];
    cudaMemcpy(ref_pts_cuda, &ref_pts_host[0], sizeof(float3) * (n_pts), cudaMemcpyHostToDevice);
    // end = clock();   //结束时间
    // cout<<"time = "<<double(end-start)/CLOCKS_PER_SEC<<"s"<<endl; //* 0.017327s
    float ths = distance_threshold;
    int max_inlies_num=0;
    float best_plane_param[4];
    std::set<int> selected_ids;
    selected_ids.clear();
    int num_no_change = 0;
    std::uniform_int_distribution<uint32_t> int_rand(0, n_pts - 1);
    for(int k=0; k<num_iterations; k++)
    {
        //* 生成随机索引
        if(selected_ids.empty())
        {
            while(selected_ids.size()<ransac_n)
            {
                int id = int_rand(rnd);
                selected_ids.insert(id);
            }
        }
        //* 点选择
        int selected_pts_id=0;
        for(set<int>::iterator it=selected_ids.begin(); it!=selected_ids.end(); it++)  //使用迭代器进行遍历 
        {
            float3 pt;
            pt.x = ref_pts_host[*it].x;
            pt.y = ref_pts_host[*it].y;
            pt.z = ref_pts_host[*it].z;
            if(selected_pts_id<ransac_n)
            {
                selected_pts[selected_pts_id] = pt;
                selected_pts_id++;
            }
            else
            {
                break;
            }
        }
        selected_ids.clear();
        //* 平面拟合
        float M[3];
        cal_plane_param(selected_pts, ransac_n, M);
        int step = 1;
        int inliers_num=0;
        float A = M[1];
        float B = M[2];
        float C = -1;
        float D = M[0];
        float norm_n = sqrt(A*A + B*B + C*C);
        float4 plane_param;
        plane_param.x = A;
        plane_param.y = B;
        plane_param.z = C;
        plane_param.w = D;
        ransac_plane_cal_dist_cuda(n_pts, ref_pts_cuda, dist_cuda, plane_param);
        cudaMemcpy(dist_host, dist_cuda, sizeof(float) * n_pts, cudaMemcpyDeviceToHost);
        for (int i = 0; i < n_pts; i+=step)
        {
            if(dist_host[i]<ths)
            {
                inliers_num++;
            }
        }
        // std::cout<<A<<", "<<B<<", "<<C<<", "<<D<<std::endl;
        if(inliers_num>max_inlies_num)
        {
            max_inlies_num = inliers_num;
            best_plane_param[0] = M[1];
            best_plane_param[1] = M[2];
            best_plane_param[2] = -1;
            best_plane_param[3] = M[0];
            num_no_change = 0;
            std::cout<<"k: "<<k<<", inliers_num: "<<inliers_num<<std::endl;
        }
        else
        {
            num_no_change++;
        }
    }
    printf("max_inlies_num: %d, total_num: %d\n\n", max_inlies_num, n_pts);
    //* 获取inlier index
    int inliers_num=0;
    float A = best_plane_param[0];
    float B = best_plane_param[1];
    float C = best_plane_param[2];
    float D = best_plane_param[3];
    float norm_n = sqrt(A*A + B*B + C*C);
    float4 plane_param;
    plane_param.x = A/D;
    plane_param.y = B/D;
    plane_param.z = C/D;
    plane_param.w = 1;
    ransac_plane_cal_dist_cuda(n_pts, ref_pts_cuda, dist_cuda, plane_param);
    cudaMemcpy(dist_host, dist_cuda, sizeof(float) * n_pts, cudaMemcpyDeviceToHost);
    float Hessien[3][4];
    for(int iter_num=0; iter_num<10; iter_num++)
    {
        for(int i=0;i<3;i++)
        {
            Hessien[i][0] = Hessien[i][1] = Hessien[i][2] = Hessien[i][3] = 0.0;
        }
        for (int i = 0; i < n_pts; i+=1)
        {
            if(dist_host[i]<2*ths)
            {
                const float3 &  pt = ref_pts_host[i];
                float r = plane_param.x*pt.x + plane_param.y*pt.y + plane_param.z*pt.z + 1;
                Hessien[0][0] += pt.x*pt.x; Hessien[0][1] += pt.x*pt.y; Hessien[0][2] += pt.x*pt.z; Hessien[0][3] -= pt.x * r;
                Hessien[1][0] += pt.y*pt.x; Hessien[1][1] += pt.y*pt.y; Hessien[1][2] += pt.y*pt.z; Hessien[1][3] -= pt.y * r;
                Hessien[2][0] += pt.z*pt.x; Hessien[2][1] += pt.z*pt.y; Hessien[2][2] += pt.z*pt.z; Hessien[2][3] -= pt.z * r;
            }
        }
        solve_linear(3, Hessien);
        plane_param.x += Hessien[0][3];
        plane_param.y += Hessien[1][3];
        plane_param.z += Hessien[2][3];
        plane_param.w = 1;
        ransac_plane_cal_dist_cuda(n_pts, ref_pts_cuda, dist_cuda, plane_param);
        cudaMemcpy(dist_host, dist_cuda, sizeof(float) * n_pts, cudaMemcpyDeviceToHost);
        inliers_num = 0;
        for (int i = 0; i < n_pts; i+=1)
        {
            if(dist_host[i]<ths)
            {
                inliers_num++;
            }
        }
        std::cout<<"inliers_num: "<<inliers_num<<std::endl;
    }
    
    cudaMemcpy(ref_pts_cuda, &all_pts_host[0], sizeof(float3) * (n_all_pts), cudaMemcpyHostToDevice);
    ransac_plane_cal_dist_cuda(n_all_pts, ref_pts_cuda, dist_cuda, plane_param);
    cudaMemcpy(dist_host, dist_cuda, sizeof(float) * n_all_pts, cudaMemcpyDeviceToHost);
    // max_inlies_num = inliers_num;
    // cv::Mat inlier_index(max_inlies_num, 1, CV_32FC1, cv::Scalar::all(0));
    std::vector<float> inlier_index_v;
    inliers_num = 0;
    for (int i = 0; i < n_all_pts; i+=1)
    {
        if(dist_host[i]<ths*2)
        {
            // inlier_index.ptr<float>(inliers_num,0)[0] = i;
            inlier_index_v.push_back(i);
            inliers_num++;
        }
    }
    std::cout<<"end ransac_plane"<<std::endl;
    // torch::Tensor output = torch::from_blob(inlier_index.ptr<float>(), /*sizes=*/{max_inlies_num, 1}).clone();
    max_inlies_num = inlier_index_v.size();
    // cuda_kdtree_test();

    //* hash map 测试
    std::vector <Point> selected_plane(inlier_index_v.size());
    // std::vector <Point> selected_plane(100000);
    //* 确定点云范围
    float x_min=0.0, x_max=0.0, y_min=0.0, y_max=0.0, z_min=0.0, z_max=0.0;
    for(int i=0; i<inlier_index_v.size(); i++)
    {
        const float3 & inlier_pt = all_pts_host[(int)(inlier_index_v[i])];
        x_min = x_min > inlier_pt.x ? inlier_pt.x : x_min;
        x_max = x_max < inlier_pt.x ? inlier_pt.x : x_max;
        y_min = y_min > inlier_pt.y ? inlier_pt.y : y_min;
        y_max = y_max < inlier_pt.y ? inlier_pt.y : y_max;
        z_min = z_min > inlier_pt.z ? inlier_pt.z : z_min;
        z_max = z_max < inlier_pt.z ? inlier_pt.z : z_max;
        // if(i>=100000)
        //     continue;
        selected_plane[i].coords[0] = inlier_pt.x;
        selected_plane[i].coords[1] = inlier_pt.y;
        selected_plane[i].coords[2] = inlier_pt.z;
    }
    // for(unsigned int i=0; i < 100000; i++) {
    //     selected_plane[i].coords[0] = 0 - 100.0*(rand() / (1.0 + RAND_MAX));
    //     selected_plane[i].coords[1] = 0 - 100.0*(rand() / (1.0 + RAND_MAX));
    //     selected_plane[i].coords[2] = 0 - 100.0*(rand() / (1.0 + RAND_MAX));
    // }
    std::cout<<"inlier_index_v.size(): "<<inlier_index_v.size()<<std::endl;
    std::cout<<"x range: ("<<x_min<<", "<<x_max<<"), y range: ("<<y_min<<", "<<y_max<<"), z range: ("<<z_min<<", "<<z_max<<")"<<std::endl;
    int max_tree_levels = 13; // play around with this value to get the best result
    std::vector <Point> queries(100000);
    std::vector <int> gpu_indexes, cpu_indexes;
    std::vector <float> gpu_dists, cpu_dists;
    for(unsigned int i=0; i < queries.size(); i++) {
        queries[i].coords[0] = 100 + 100.0*(rand() / (1.0 + RAND_MAX));
        queries[i].coords[1] = 100 + 100.0*(rand() / (1.0 + RAND_MAX));
        queries[i].coords[2] = 100 + 100.0*(rand() / (1.0 + RAND_MAX));
    }
    std::cout<<"CreateKDTree"<<std::endl;
    // Time to create the tree
    timeval t1, t2;
    gettimeofday(&t1, NULL);
    KDtree tree;
    CUDA_KDTree GPU_tree;
    tree.Create(selected_plane, max_tree_levels);
    GPU_tree.CreateKDTree(tree.GetRoot(), tree.GetNumNodes(), selected_plane);
    gettimeofday(&t2, NULL);
    double gpu_create_time = TimeDiff(t1,t2);
    std::cout<<"Search"<<std::endl;
    // Time to search the tree
    gettimeofday(&t1, NULL);
    GPU_tree.Search(queries, gpu_indexes, gpu_dists);
    gettimeofday(&t2, NULL);
    double gpu_search_time = TimeDiff(t1,t2);
    std::cout<<"GPU_tree.Search end"<<std::endl;
    printf("Points in the tree: %ld\n", selected_plane.size());
    printf("Query points: %ld\n", queries.size());
    printf("GPU max tree depth: %d\n", max_tree_levels);
    printf("GPU create + search: %g + %g = %g ms\n", gpu_create_time, gpu_search_time, gpu_create_time + gpu_search_time);

    torch::Tensor output = torch::from_blob(inlier_index_v.data(), /*sizes=*/{max_inlies_num, 1}).clone();
    cudaFree(ref_pts_cuda);
    cudaFree(dist_cuda);
    // delete [] ref_pts_host;
    delete [] dist_host;
    end = clock();   //结束时间
    cout<<"cal time = "<<float(end-start)/CLOCKS_PER_SEC<<"s"<<endl; //* 0.013133s
    return output;
    // return torch::empty(0);
}

torch::Tensor ransac_plane(const torch::Tensor & basic_pts, double distance_threshold, int64_t ransac_n, int64_t num_iterations)
{
    // To recreate the same random numbers across runs of the program, set seed to a specific
    // number instead of a number from random_device
    std::random_device rd;
    uint32_t seed = rd();
    std::mt19937 rnd(seed);  // mersenne_twister_engine
    clock_t start,end, start_fit,end_fit; //定义clock_t变量
    start = clock(); //开始时间
    bool debug = true;
    srand((unsigned)time(NULL));
    // std::cout << (rand()/double(RAND_MAX)) << " "; //生成[0,1]范围内的随机数
    //* basic_pts [N ,3+n]
    auto shape = basic_pts.sizes();
    if(shape.size()!=2 || shape[1]<3)
    {
        std::cout<<"Wrong shape of pts!!"<<std::endl;
        return torch::empty(0);
    }
    int n_all_pts = shape[0];
    int ch = shape[1];
    torch::Tensor basic_pts_torch = basic_pts.clone().to(torch::kCPU).contiguous();
    //* 转为opencv数据 下面并不是复制，而是引用！！！！用这种方式访问比较快
    cv::Mat basic_pts_cv = cv::Mat{n_all_pts, 1, CV_32FC(ch), basic_pts_torch.data_ptr<float>()}; //* 3通道float
    if(n_all_pts<NUM_MATCH_POINTS)
    {
        std::cout<<"Too few pts!!"<<std::endl;
        return torch::empty(0);
    }
    //* 直通滤波，为了提取地面,提取地面之后，再将地面用kdtree表示，直接处理点云
    int n_pts = 0;
    std::vector<float3> ref_pts_host, all_pts_host;
    std::vector<int> ref_pts_ids;
    for(int i=0; i<n_all_pts; i++)
    {
        float3 pt;
        pt.x = basic_pts_cv.ptr<float>(i, 0)[0];
        pt.y = basic_pts_cv.ptr<float>(i, 0)[1];
        pt.z = basic_pts_cv.ptr<float>(i, 0)[2];
        all_pts_host.emplace_back(pt);
        if(basic_pts_cv.ptr<float>(i, 0)[2]<1.5)
        {
            ref_pts_host.emplace_back(pt);
            ref_pts_ids.emplace_back(i);
            n_pts++;
        }
    }

    // start = clock(); //开始时间
    // float3 * ref_pts_host, * selected_pts;
    float3 * selected_pts;
    float3 * ref_pts_cuda;
    float * dist_host;
    float * dist_cuda;
    cudaMalloc((void **)&ref_pts_cuda,    sizeof(float3)  * (n_all_pts));
    cudaMalloc((void **)&dist_cuda,       sizeof(float)   * (n_all_pts));
    dist_host = new float[n_all_pts];
    selected_pts = new float3[ransac_n];
    cudaMemcpy(ref_pts_cuda, &ref_pts_host[0], sizeof(float3) * (n_pts), cudaMemcpyHostToDevice);
    // end = clock();   //结束时间
    // cout<<"time = "<<double(end-start)/CLOCKS_PER_SEC<<"s"<<endl; //* 0.017327s
    float ths = distance_threshold;
    int max_inlies_num=0;
    float best_plane_param[4];
    std::set<int> selected_ids;
    selected_ids.clear();
    int num_no_change = 0;
    std::uniform_int_distribution<uint32_t> int_rand(0, n_pts - 1);
    for(int k=0; k<num_iterations; k++)
    {
        //* 生成随机索引
        if(selected_ids.empty())
        {
            while(selected_ids.size()<ransac_n)
            {
                int id = int_rand(rnd);
                selected_ids.insert(id);
            }
        }
        //* 点选择
        int selected_pts_id=0;
        for(set<int>::iterator it=selected_ids.begin(); it!=selected_ids.end(); it++)  //使用迭代器进行遍历 
        {
            float3 pt;
            pt.x = ref_pts_host[*it].x;
            pt.y = ref_pts_host[*it].y;
            pt.z = ref_pts_host[*it].z;
            if(selected_pts_id<ransac_n)
            {
                selected_pts[selected_pts_id] = pt;
                selected_pts_id++;
            }
            else
            {
                break;
            }
        }
        selected_ids.clear();
        //* 平面拟合
        float M[3];
        cal_plane_param(selected_pts, ransac_n, M);
        int step = 1;
        int inliers_num=0;
        float A = M[1];
        float B = M[2];
        float C = -1;
        float D = M[0];
        float norm_n = sqrt(A*A + B*B + C*C);
        float4 plane_param;
        plane_param.x = A;
        plane_param.y = B;
        plane_param.z = C;
        plane_param.w = D;
        ransac_plane_cal_dist_cuda(n_pts, ref_pts_cuda, dist_cuda, plane_param);
        cudaMemcpy(dist_host, dist_cuda, sizeof(float) * n_pts, cudaMemcpyDeviceToHost);
        for (int i = 0; i < n_pts; i+=step)
        {
            if(dist_host[i]<ths)
            {
                inliers_num++;
            }
        }
        // std::cout<<A<<", "<<B<<", "<<C<<", "<<D<<std::endl;
        if(inliers_num>max_inlies_num)
        {
            max_inlies_num = inliers_num;
            best_plane_param[0] = M[1];
            best_plane_param[1] = M[2];
            best_plane_param[2] = -1;
            best_plane_param[3] = M[0];
            num_no_change = 0;
            std::cout<<"k: "<<k<<", inliers_num: "<<inliers_num<<std::endl;
        }
        else
        {
            num_no_change++;
        }
    }
    printf("max_inlies_num: %d, total_num: %d\n\n", max_inlies_num, n_pts);
    //* 获取inlier index
    int inliers_num=0;
    float A = best_plane_param[0];
    float B = best_plane_param[1];
    float C = best_plane_param[2];
    float D = best_plane_param[3];
    float norm_n = sqrt(A*A + B*B + C*C);
    float4 plane_param;
    plane_param.x = A/D;
    plane_param.y = B/D;
    plane_param.z = C/D;
    plane_param.w = 1;
    ransac_plane_cal_dist_cuda(n_pts, ref_pts_cuda, dist_cuda, plane_param);
    cudaMemcpy(dist_host, dist_cuda, sizeof(float) * n_pts, cudaMemcpyDeviceToHost);
    float Hessien[3][4];
    for(int iter_num=0; iter_num<10; iter_num++)
    {
        for(int i=0;i<3;i++)
        {
            Hessien[i][0] = Hessien[i][1] = Hessien[i][2] = Hessien[i][3] = 0.0;
        }
        for (int i = 0; i < n_pts; i+=1)
        {
            if(dist_host[i]<2*ths)
            {
                const float3 &  pt = ref_pts_host[i];
                float r = plane_param.x*pt.x + plane_param.y*pt.y + plane_param.z*pt.z + 1;
                Hessien[0][0] += pt.x*pt.x; Hessien[0][1] += pt.x*pt.y; Hessien[0][2] += pt.x*pt.z; Hessien[0][3] -= pt.x * r;
                Hessien[1][0] += pt.y*pt.x; Hessien[1][1] += pt.y*pt.y; Hessien[1][2] += pt.y*pt.z; Hessien[1][3] -= pt.y * r;
                Hessien[2][0] += pt.z*pt.x; Hessien[2][1] += pt.z*pt.y; Hessien[2][2] += pt.z*pt.z; Hessien[2][3] -= pt.z * r;
            }
        }
        solve_linear(3, Hessien);
        plane_param.x += Hessien[0][3];
        plane_param.y += Hessien[1][3];
        plane_param.z += Hessien[2][3];
        plane_param.w = 1;
        ransac_plane_cal_dist_cuda(n_pts, ref_pts_cuda, dist_cuda, plane_param);
        cudaMemcpy(dist_host, dist_cuda, sizeof(float) * n_pts, cudaMemcpyDeviceToHost);
        inliers_num = 0;
        for (int i = 0; i < n_pts; i+=1)
        {
            if(dist_host[i]<ths)
            {
                inliers_num++;
            }
        }
        std::cout<<"inliers_num: "<<inliers_num<<std::endl;
    }
    
    cudaMemcpy(ref_pts_cuda, &all_pts_host[0], sizeof(float3) * (n_all_pts), cudaMemcpyHostToDevice);
    ransac_plane_cal_dist_cuda(n_all_pts, ref_pts_cuda, dist_cuda, plane_param);
    cudaMemcpy(dist_host, dist_cuda, sizeof(float) * n_all_pts, cudaMemcpyDeviceToHost);
    // max_inlies_num = inliers_num;
    // cv::Mat inlier_index(max_inlies_num, 1, CV_32FC1, cv::Scalar::all(0));
    std::vector<float> inlier_index_v;
    inliers_num = 0;
    for (int i = 0; i < n_all_pts; i+=1)
    {
        if(dist_host[i]<ths*2)
        {
            // inlier_index.ptr<float>(inliers_num,0)[0] = i;
            inlier_index_v.push_back(i);
            inliers_num++;
        }
    }
    std::cout<<"end ransac_plane"<<std::endl;
    // torch::Tensor output = torch::from_blob(inlier_index.ptr<float>(), /*sizes=*/{max_inlies_num, 1}).clone();
    max_inlies_num = inlier_index_v.size();
    //* hash map 测试
    std::vector <Point> selected_plane(inlier_index_v.size());
    //* 确定点云范围
    float x_min=0.0, x_max=0.0, y_min=0.0, y_max=0.0, z_min=0.0, z_max=0.0;
    for(int i=0; i<inlier_index_v.size(); i++)
    {
        const float3 & inlier_pt = all_pts_host[(int)(inlier_index_v[i])];
        x_min = x_min > inlier_pt.x ? inlier_pt.x : x_min;
        x_max = x_max < inlier_pt.x ? inlier_pt.x : x_max;
        y_min = y_min > inlier_pt.y ? inlier_pt.y : y_min;
        y_max = y_max < inlier_pt.y ? inlier_pt.y : y_max;
        z_min = z_min > inlier_pt.z ? inlier_pt.z : z_min;
        z_max = z_max < inlier_pt.z ? inlier_pt.z : z_max;
        selected_plane[i].coords[0] = inlier_pt.x;
        selected_plane[i].coords[1] = inlier_pt.y;
        selected_plane[i].coords[2] = inlier_pt.z;
    }
    std::cout<<"inlier_index_v.size(): "<<inlier_index_v.size()<<std::endl;
    std::cout<<"x range: ("<<x_min<<", "<<x_max<<"), y range: ("<<y_min<<", "<<y_max<<"), z range: ("<<z_min<<", "<<z_max<<")"<<std::endl;
    torch::Tensor output = torch::from_blob(inlier_index_v.data(), /*sizes=*/{max_inlies_num, 1}).clone();
    cudaFree(ref_pts_cuda);
    cudaFree(dist_cuda);
    // delete [] ref_pts_host;
    delete [] dist_host;
    end = clock();   //结束时间
    cout<<"cal time = "<<float(end-start)/CLOCKS_PER_SEC<<"s"<<endl; //* 0.013133s
    return output;
    // return torch::empty(0);
}

torch::Tensor proj2depth(const torch::Tensor & uv_valid, const torch::Tensor & z_valid, const torch::Tensor & depth)
{
    auto shape = uv_valid.sizes();
    torch::Tensor uv_valid_tmp = uv_valid.clone().to(torch::kCPU).contiguous();
    cv::Mat uv_cv = cv::Mat{shape[0], shape[1], CV_32FC1, uv_valid_tmp.data_ptr<float>()};
    torch::Tensor z_valid_tmp = z_valid.clone().to(torch::kCPU).contiguous();
    cv::Mat z_valid_cv = cv::Mat{shape[0], 1, CV_32FC1, z_valid_tmp.data_ptr<float>()};
    auto dshape = depth.sizes();
    torch::Tensor depth_tmp = depth.clone().to(torch::kCPU).contiguous();
    cv::Mat depth_cv = cv::Mat{dshape[0], dshape[1], CV_32FC1, depth_tmp.data_ptr<float>()};
    for(int i=0;i<shape[0];i++)
    {
        float u_f = uv_cv.at<float>(i,0);
        float v_f = uv_cv.at<float>(i,1);
        float d_f_old = depth_cv.at<float>(int(v_f),int(u_f));
        float d_f_new = z_valid_cv.at<float>(i,0);
        if(d_f_old>d_f_new)
            depth_cv.at<float>(int(v_f),int(u_f)) = d_f_new;
    }
    torch::Tensor output = torch::from_blob(depth_cv.ptr<float>(), /*sizes=*/{dshape[0], dshape[1]}).clone();
    return output;
}

torch::Tensor depth_filter(const torch::Tensor & image_info_torch)
{
    std::cout<<"depth_interp_comp start!"<<std::endl;
    bool debug = true;
    if(debug)
    {
        std::cout<<"Debug mode!"<<std::endl;
    }
    //* image_info_torch torch.Size([8, 1024, 1280]) [r,g,b,e,d，normal] 颜色3、颜色平滑度1、深度1, 法线3
    //* 测试
    // simpletools.depth_interp_comp(torch.ones(4,4))

    //* 形状检查
    auto dshape = image_info_torch.sizes(); //* [C H W]
    if(dshape.size()!=3 || dshape[0]< 1 )
    {
        std::cout<<"orig_depth shape: "<<dshape<<std::endl;
        std::cout<<"Wrong shape, should be the same cxHxW c>=1!\n";
        return torch::empty(0);
    }
    int C = dshape[0], H = dshape[1], W = dshape[2];

    //* 分割和连续化
    torch::Tensor depth_torch = image_info_torch.index_select(0, torch::tensor({0}).to(image_info_torch.device())).to(torch::kCPU).permute({1,2,0}).contiguous();

    if(debug)
    {
        auto rgb_shape = depth_torch.sizes();
        std::cout<<"dshape shape: "<<dshape<<std::endl; //* dshape shape: [8, 1024, 1280]
        std::cout<<"rgb_shape shape: "<<rgb_shape<<std::endl; //*  [1024, 1280, 3]
    }
    
    //* 转为opencv数据 下面并不是复制，而是引用！！！！
    cv::Mat depth_cv = cv::Mat{H, W, CV_32FC(1), depth_torch.data_ptr<float>()};
    //* 定义指针
    float * depth_cuda;
    float * depth_new_host;
    float * depth_new_cuda;
    curandState *rand_states_cuda;
    //* 分配内存
    if(debug)
    {
        std::cout<<"cudaMalloc"<<std::endl;
    }
    cudaMalloc((void **)&depth_cuda,        sizeof(float)  * (H*W));
    cudaMalloc((void **)&depth_new_cuda,    sizeof(float)  * (H*W));
    cudaMalloc((void **)&rand_states_cuda,  sizeof(curandState) * (H*W));


    depth_new_host = new float[H*W];

    //* 复制或者赋值
    if(debug)
    {
        std::cout<<"cudaMemcpy"<<std::endl;
    }

    cudaMemcpy(depth_cuda, depth_torch.data_ptr<float>(), sizeof(float) * (H*W), cudaMemcpyHostToDevice);
    cudaMemcpy(depth_new_cuda, depth_torch.data_ptr<float>(), sizeof(float) * (H*W), cudaMemcpyHostToDevice);

    //* 运行cuda
    if(debug)
    {
        std::cout<<"run depth_interp_comp_cuda"<<std::endl;

    }
    depth_filter_cuda(W, H, depth_cuda,depth_new_cuda, rand_states_cuda);
    cudaMemcpy(depth_new_host, depth_new_cuda, sizeof(float) * W * H, cudaMemcpyDeviceToHost);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    // //* cv 转为 torch
    torch::Tensor output = torch::from_blob(depth_new_host, /*sizes=*/{H, W}).clone();

    // std::cout<<"size: "<<result.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(1,4)}).sizes()<<std::endl;
    
    //* 清除指针
    if(debug)
    {
        std::cout<<"cudaFree"<<std::endl;
    }
    cudaFree(depth_cuda);
    cudaFree(depth_new_cuda);
    cudaFree(rand_states_cuda);


    delete [] depth_new_host;


    //* 返回 return torch::empty(0);
    // return result;
    return output;
}

torch::Tensor depth_interp_comp(const torch::Tensor & image_info_torch, const torch::Tensor & cam_param_torch)
{
    std::cout<<"depth_interp_comp start!"<<std::endl;
    bool debug = true;
    if(debug)
    {
        std::cout<<"Debug mode!"<<std::endl;
        auto dshape = cam_param_torch.sizes();
        std::cout<<"cam_param_torch shape: "<<dshape<<std::endl;
    }
    //* image_info_torch torch.Size([8, 1024, 1280]) [r,g,b,e,d，normal] 颜色3、颜色平滑度1、深度1, 法线3
    //* 测试
    // simpletools.depth_interp_comp(torch.ones(4,4))

    //* 形状检查
    auto dshape = image_info_torch.sizes(); //* [C H W]
    if(dshape.size()!=3 || dshape[0]< 8 )
    {
        std::cout<<"orig_depth shape: "<<dshape<<std::endl;
        std::cout<<"Wrong shape, should be the same cxHxW c>=8!\n";
        return torch::empty(0);
    }
    int C = dshape[0], H = dshape[1], W = dshape[2];

    //* 分割和连续化
    torch::Tensor rgb_torch = image_info_torch.index_select(0, torch::tensor({0,1,2}).to(image_info_torch.device())).to(torch::kCPU).permute({1,2,0}).contiguous();
    torch::Tensor color_smooth_torch = image_info_torch.index_select(0, torch::tensor({3}).to(image_info_torch.device())).to(torch::kCPU).permute({1,2,0}).contiguous();
    torch::Tensor depth_torch = image_info_torch.index_select(0, torch::tensor({4}).to(image_info_torch.device())).to(torch::kCPU).permute({1,2,0}).contiguous();
    torch::Tensor normal_torch = image_info_torch.index_select(0, torch::tensor({5,6,7}).to(image_info_torch.device())).to(torch::kCPU).permute({1,2,0}).contiguous();
    torch::Tensor cam_pts_torch = image_info_torch.index_select(0, torch::tensor({8,9,10}).to(image_info_torch.device())).to(torch::kCPU).permute({1,2,0}).contiguous();
    torch::Tensor cam_param_torch_tmp = cam_param_torch.to(torch::kCPU).contiguous().clone();
    float * cam_param = cam_param_torch_tmp.data_ptr<float>();
    int num_cam_param = cam_param_torch_tmp.sizes()[0];
    if(debug)
    {
        for(int i=0; i<num_cam_param; i++)
        {
            std::cout<<i<<", "<<cam_param[i]<<std::endl;
        }
    }

    if(debug)
    {
        auto rgb_shape = rgb_torch.sizes();
        std::cout<<"dshape shape: "<<dshape<<std::endl; //* dshape shape: [8, 1024, 1280]
        std::cout<<"rgb_shape shape: "<<rgb_shape<<std::endl; //*  [1024, 1280, 3]
    }
    
    //* 转为opencv数据 下面并不是复制，而是引用！！！！
    cv::Mat rgb_cv = cv::Mat{H, W, CV_32FC(3), rgb_torch.data_ptr<float>()}; //* 3通道float
    cv::Mat color_smooth_cv = cv::Mat{H, W, CV_32FC(1), color_smooth_torch.data_ptr<float>()};
    cv::Mat depth_cv = cv::Mat{H, W, CV_32FC(1), depth_torch.data_ptr<float>()};
    cv::Mat normal_cv = cv::Mat{H, W, CV_32FC(3), normal_torch.data_ptr<float>()};
    cv::Mat cam_pts_cv = cv::Mat{H, W, CV_32FC(3), cam_pts_torch.data_ptr<float>()};
    // cv::Mat cam_param_cv = cv::Mat{H, W, CV_32FC(1), cam_param_torch_tmp.data_ptr<float>()};

    cv::Mat depth_new_cv(H, W, CV_32FC1, cv::Scalar::all(0));
    
    //* 定义指针
    float3 * rgb_host;
    float3 * rgb_cuda;
    float * color_smooth_cuda;
    float * depth_cuda;
    float3 * normal_host;
    float3 * normal_cuda;
    float * depth_new_host;
    float * depth_new_cuda;
    float3 * cam_pts_host;
    float3 * cam_pts_cuda;
    float3 * esti_normal_host;
    float3 * esti_normal_cuda;
    curandState *rand_states_cuda;
    float * curve_host;
    float * curve_cuda;
    float * cam_param_cuda;

    //* 分配内存
    if(debug)
    {
        std::cout<<"cudaMalloc"<<std::endl;
    }
    cudaMalloc((void **)&color_smooth_cuda, sizeof(float)  * (H*W));
    cudaMalloc((void **)&depth_cuda,        sizeof(float)  * (H*W));
    cudaMalloc((void **)&depth_new_cuda,    sizeof(float)  * (H*W));
    cudaMalloc((void **)&normal_cuda,       sizeof(float3) * (H*W));
    cudaMalloc((void **)&rgb_cuda,          sizeof(float3) * (H*W));
    cudaMalloc((void **)&cam_pts_cuda,      sizeof(float3) * (H*W));
    cudaMalloc((void **)&esti_normal_cuda,  sizeof(float3) * (H*W));
    cudaMalloc((void **)&rand_states_cuda,  sizeof(curandState) * (H*W));
    cudaMalloc((void **)&curve_cuda,        sizeof(float)  * (H*W));
    cudaMalloc((void **)&cam_param_cuda,    sizeof(float)  * (num_cam_param));

    normal_host = new float3[H*W];
    rgb_host = new float3[H*W];
    cam_pts_host = new float3[H*W];
    esti_normal_host = new float3[H*W];
    depth_new_host = new float[H*W];
    curve_host = new float[H*W];

    //* 复制或者赋值
    if(debug)
    {
        std::cout<<"cudaMemcpy"<<std::endl;
    }
    for (int col = 0; col < W; ++col)
    {
        for (int row = 0; row < H; ++row)
        {
            int center = row * W + col;
            float3 normal;
            normal.x = normal_cv.ptr<float>(row, col)[0];
            normal.y = normal_cv.ptr<float>(row, col)[1];
            normal.z = normal_cv.ptr<float>(row, col)[2];
            normal_host[center] = normal;
            float3 rgb;
            rgb.x = rgb_cv.ptr<float>(row, col)[0];
            rgb.y = rgb_cv.ptr<float>(row, col)[1];
            rgb.z = rgb_cv.ptr<float>(row, col)[2];
            rgb_host[center] = rgb;
            float3 cam_pt;
            cam_pt.x = cam_pts_cv.ptr<float>(row, col)[0];
            cam_pt.y = cam_pts_cv.ptr<float>(row, col)[1];
            cam_pt.z = cam_pts_cv.ptr<float>(row, col)[2];
            cam_pts_host[center] = cam_pt;
            if(debug && (row>=901 && row<905) && col == 1122 && 0)
            {
                // std::cout<<"image_info_torch: "<<image_info_torch.index({torch::indexing::Slice(),row,col})<<std::endl;
                std::cout<<"cam_pt: ("<<cam_pt.x<<", "<<cam_pt.y<<", "<<cam_pt.z<<")"<<std::endl;
            }
        }
    }
    cudaMemcpy(color_smooth_cuda, color_smooth_torch.data_ptr<float>(), sizeof(float) * (H*W), cudaMemcpyHostToDevice);
    cudaMemcpy(depth_cuda, depth_torch.data_ptr<float>(), sizeof(float) * (H*W), cudaMemcpyHostToDevice);
    cudaMemcpy(depth_new_cuda, depth_torch.data_ptr<float>(), sizeof(float) * (H*W), cudaMemcpyHostToDevice);
    cudaMemcpy(normal_cuda, normal_host, sizeof(float3) * (H*W), cudaMemcpyHostToDevice);
    cudaMemcpy(rgb_cuda, rgb_host, sizeof(float3) * (H*W), cudaMemcpyHostToDevice);
    cudaMemcpy(cam_pts_cuda, cam_pts_host, sizeof(float3) * (H*W), cudaMemcpyHostToDevice);
    cudaMemcpy(cam_param_cuda, cam_param, sizeof(float) * (num_cam_param), cudaMemcpyHostToDevice);

    //* 运行cuda
    if(debug)
    {
        std::cout<<"run depth_interp_comp_cuda"<<std::endl;
    }
    depth_interp_comp_cuda(W, H, depth_cuda, color_smooth_cuda, normal_cuda, depth_new_cuda, rand_states_cuda, cam_pts_cuda, esti_normal_cuda, curve_cuda, cam_param_cuda);
    cudaMemcpy(depth_new_host, depth_new_cuda, sizeof(float) * W * H, cudaMemcpyDeviceToHost);
    cudaMemcpy(esti_normal_host, esti_normal_cuda, sizeof(float3) * W * H, cudaMemcpyDeviceToHost);
    cudaMemcpy(curve_host, curve_cuda, sizeof(float) * W * H, cudaMemcpyDeviceToHost);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    torch::Tensor result = torch::zeros({H, W, 5});
    // //* cv 转为 torch
    // torch::Tensor output = torch::from_blob(depth_new_host, /*sizes=*/{H, W}).clone();
    result.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0,1)}) = torch::from_blob(depth_new_host,{H, W, 1}).clone().to(result.device());
    result.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(1,4)}) = torch::from_blob(esti_normal_host,{H, W, 3}).clone().to(result.device());
    result.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(4,5)}) = torch::from_blob(curve_host,{H, W, 1}).clone().to(result.device());

    // std::cout<<"size: "<<result.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(1,4)}).sizes()<<std::endl;
    
    //* 清除指针
    if(debug)
    {
        std::cout<<"cudaFree"<<std::endl;
    }
    cudaFree(color_smooth_cuda);
    cudaFree(depth_cuda);
    cudaFree(normal_cuda);
    cudaFree(rgb_cuda);
    cudaFree(rand_states_cuda);
    cudaFree(cam_pts_cuda);
    cudaFree(esti_normal_cuda);
    cudaFree(curve_cuda);
    cudaFree(cam_param_cuda);


    delete [] normal_host;
    delete [] rgb_host;
    delete [] cam_pts_host;
    delete [] esti_normal_host;
    delete [] depth_new_host;
    delete [] curve_host;

    //* 返回 return torch::empty(0);
    return result;
    // return output;
}

torch::Tensor proj_depth_normal(const torch::Tensor & uvd_normal_torch, int64_t width, int64_t height)
{
    //* 计算像素的法线信息
    //* input: torch.Size([422399, 6]) uvd(3) + normal(3) = 6
    //* uchar* a = mat3D.ptr<uchar>(0,0);//a指向前两维限定下的行首的地址
    auto shape = uvd_normal_torch.sizes();
    torch::Tensor uvd_normal_torch_tmp = uvd_normal_torch.clone().to(torch::kCPU).contiguous();
    cv::Mat uvd_normal_cv = cv::Mat{shape[0], shape[1], CV_32FC1, uvd_normal_torch_tmp.data_ptr<float>()};
    cv::Mat depth_normal_cv(height, width, CV_32FC(4), cv::Scalar(1000)); //* depth(1) + normal(3) = 4
    for(int i=0;i<shape[0];i++)
    {
        float u_f = uvd_normal_cv.ptr<float>(i)[0];
        float v_f = uvd_normal_cv.ptr<float>(i)[1];
        float d_f_new = uvd_normal_cv.ptr<float>(i)[2];
        float d_f_old = depth_normal_cv.ptr<float>(int(v_f),int(u_f))[0];
        if(d_f_old>d_f_new)
        {
            depth_normal_cv.ptr<float>(int(v_f),int(u_f))[0] = d_f_new;
            depth_normal_cv.ptr<float>(int(v_f),int(u_f))[1] = uvd_normal_cv.ptr<float>(i)[3];
            depth_normal_cv.ptr<float>(int(v_f),int(u_f))[2] = uvd_normal_cv.ptr<float>(i)[4];
            depth_normal_cv.ptr<float>(int(v_f),int(u_f))[3] = uvd_normal_cv.ptr<float>(i)[5];
        }
    }
    torch::Tensor output = torch::from_blob(depth_normal_cv.ptr<float>(), /*sizes=*/{height, width,4}).clone();
    return output;
}


TORCH_LIBRARY(simple_tools, m) {
    m.def("proj2depth", proj2depth);
    m.def("depth_interp_comp", depth_interp_comp);
    m.def("proj_depth_normal", proj_depth_normal);
    m.def("depth_filter", depth_filter);
    m.def("ransac_plane", ransac_plane);

}