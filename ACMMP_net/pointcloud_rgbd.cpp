#include "pointcloud_rgbd.hpp"

float3 cross_product(float3 a, float3 b)
{
	float3 c;
	///*c.x = a[1] * b[2] - a[2] * b[1];
	//c.y = a[2] * b[0] - a[0] * b[2];
	//c.z = a[0] * b[1] - a[1] * b[0];*/
	c.x = a.y * b.z - a.z * b.y;
	c.y = a.z * b.x - a.x * b.z;
	c.z = a.x * b.y - a.y * b.x;
	return c;
}

float dot_product(float3 a, float3 b)
{
	float result;
	//result = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
	result = a.x * b.x + a.y * b.y + a.z * b.z;
	return result;
}

float norm2(float3 v)
{
	float result;
	result = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
	return result;
}

void R_from_angle_axis(float angle, float3 u, float R[3][3])
{
	float norm = norm2(u);
	u.x = u.x / norm;
	u.y = u.y / norm;
	u.z = u.z / norm;

	R[0][0] = cos(angle) + u.x * u.x * (1 - cos(angle));
	R[0][1] = u.x * u.y * (1 - cos(angle)) - u.z * sin(angle);
	R[0][2] = u.y * sin(angle) + u.x * u.z * (1 - cos(angle));

	R[1][0] = u.z * sin(angle) + u.x * u.y * (1 - cos(angle));
	R[1][1] = cos(angle) + u.y * u.y * (1 - cos(angle));
	R[1][2] = -u.x * sin(angle) + u.y * u.z * (1 - cos(angle));

	R[2][0] = -u.y * sin(angle) + u.x * u.z * (1 - cos(angle));
	R[2][1] = u.x * sin(angle) + u.y * u.z * (1 - cos(angle));
	R[2][2] = cos(angle) + u.z * u.z * (1 - cos(angle));
}

void calculate_matrix(float3 vectorBefore, float3 vectorAfter, float R[3][3])
{
	float rotationAngle = acos(dot_product(vectorBefore, vectorAfter) / norm2(vectorBefore) / norm2(vectorAfter));
	float3 rotationAxis = cross_product(vectorBefore, vectorAfter);
	R_from_angle_axis(rotationAngle, rotationAxis, R);
}

void cal_Ax(int N, float a[3][4])
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

void fit_plane_params(const float3 * selected_pts, int N, float M[3])
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

    cal_Ax(3, c);

    for (int i = 0; i < 3; i++)
    {
        M[i] = c[i][3];
    }
}

float4 ransac_fit_plane(float3 *ref_pts_host, int n_pts, double distance_threshold, int64_t ransac_n, int64_t num_iterations)
{
	std::random_device rd;
    uint32_t seed = rd();
    std::mt19937 rnd(seed);  // mersenne_twister_engine
	// int n_pts = ref_pts_host.size();
	float3 * selected_pts;
    float3 * ref_pts_cuda;
    float * dist_host;
    float * dist_cuda;

    cudaMalloc((void **)&ref_pts_cuda,    sizeof(float3)  * (n_pts));
    cudaMalloc((void **)&dist_cuda,       sizeof(float)   * (n_pts));
    dist_host = new float[n_pts];
    selected_pts = new float3[ransac_n];

    cudaMemcpy(ref_pts_cuda, ref_pts_host, sizeof(float3) * (n_pts), cudaMemcpyHostToDevice);
    // end = clock();   //结束时间
    // cout<<"time = "<<double(end-start)/CLOCKS_PER_SEC<<"s"<<endl; //* 0.017327s
    float ths = distance_threshold;
    int max_inlies_num=0;
    float best_plane_param[4];
    std::set<int> selected_ids;
    selected_ids.clear();
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
        fit_plane_params(selected_pts, ransac_n, M);
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
        point_to_plane_dist_cuda(n_pts, ref_pts_cuda, dist_cuda, plane_param);
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
            std::cout<<"k: "<<k<<", inliers_num: "<<inliers_num<<std::endl;
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
    point_to_plane_dist_cuda(n_pts, ref_pts_cuda, dist_cuda, plane_param);
    cudaMemcpy(dist_host, dist_cuda, sizeof(float) * n_pts, cudaMemcpyDeviceToHost);
    float Hessien[3][4];

    for(int iter_num=0; iter_num<50; iter_num++)
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
        cal_Ax(3, Hessien);
        plane_param.x += Hessien[0][3];
        plane_param.y += Hessien[1][3];
        plane_param.z += Hessien[2][3];
        plane_param.w = 1;
        point_to_plane_dist_cuda(n_pts, ref_pts_cuda, dist_cuda, plane_param);
        cudaMemcpy(dist_host, dist_cuda, sizeof(float) * n_pts, cudaMemcpyDeviceToHost);
		CheckCUDAError("cudaMemcpy");
        inliers_num = 0;
        for (int i = 0; i < n_pts; i+=1)
        {
            if(dist_host[i]<ths)
            {
                inliers_num++;
            }
        }
        std::cout<<"inliers_num: "<<inliers_num<<std::endl;
		if(max_inlies_num<inliers_num)
		{
			max_inlies_num = inliers_num;
		}
		else
		{
			break;
		}
    }
	cudaFree(ref_pts_cuda);
    cudaFree(dist_cuda);
    delete [] dist_host;
    delete [] selected_pts;
	std::cout<<"end ransac_fit_plane\n";
	return plane_param;
}

Global_map::Global_map()
{
	m_mutex_pts_vec = std::make_shared<std::mutex>();
	m_mutex_img_pose_for_projection = std::make_shared<std::mutex>();
	m_mutex_recent_added_list = std::make_shared<std::mutex>();
	m_mutex_rgb_pts_in_recent_hitted_boxes = std::make_shared<std::mutex>();
	m_mutex_m_box_recent_hitted = std::make_shared<std::mutex>();
	m_mutex_m_voxels_recent_visited_between_keyframes = std::make_shared<std::mutex>();
	m_mutex_pts_last_visited = std::make_shared<std::mutex>();
	// Allocate memory for pointclouds
	m_rgb_pts_vec.reserve(1e9);
	// if (Common_tools::get_total_phy_RAM_size_in_GB() < 12)
	// {
	// 	scope_color(ANSI_COLOR_RED_BOLD);
	// 	std::this_thread::sleep_for(std::chrono::seconds(1));
	// 	cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;
	// 	cout << "I have detected your physical memory smaller than 12GB (currently: " << Common_tools::get_total_phy_RAM_size_in_GB()
	// 		 << "GB). I recommend you to add more physical memory for improving the overall performance of R3LIVE." << endl;
	// 	cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;
	// 	std::this_thread::sleep_for(std::chrono::seconds(5));
	// 	m_rgb_pts_vec.reserve(1e8);
	// }
	// else
	// {
	// 	m_rgb_pts_vec.reserve(1e9);
	// }
	// {
	// 	// scope_color(ANSI_COLOR_RED_BOLD);
	// 	std::this_thread::sleep_for(std::chrono::seconds(1));
	// 	cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;
	// 	cout << "Physical memory: " << Common_tools::get_total_phy_RAM_size_in_GB()<< "GB." << endl;
	// 	cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;
	// 	std::this_thread::sleep_for(std::chrono::seconds(1));
	// 	// scope_color(ANSI_COLOR_RESET);
	// 	// cout << ANSI_COLOR_RESET;
		
	// }
	// m_rgb_pts_in_recent_visited_voxels.reserve( 1e6 );
	int64_t if_start_service = 0;
	// if (if_start_service)
	// {
	// 	m_thread_service = std::make_shared<std::thread>(&Global_map::service_refresh_pts_for_projection, this);
	// }
}

Global_map::~Global_map()
{
	// if(pts_cuda != nullptr)
	// 	cudaFree(pts_cuda);
	// CheckCUDAError("cudaFree");
}

torch::Tensor Global_map::point_cloud_segmentation(const torch::Tensor & image_info_torch, int64_t img_id, bool refresh)
{
	//* 这种投影方式只是单纯考虑点，而没有考虑点的属性
	time_last tim;
	tim.tic();
	if(img_id>=images.size())
	{
		std::cout<<" Too large img_id: "<<img_id<<", expected to less than: "<<images.size()<<std::endl;
		return torch::empty(0);
	}
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
    torch::Tensor rgb_torch = image_info_torch.index_select(0, torch::tensor({0,1,2}).to(image_info_torch.device())).to(torch::kCPU).permute({1,2,0}).contiguous();
    torch::Tensor color_smooth_torch = image_info_torch.index_select(0, torch::tensor({3}).to(image_info_torch.device())).to(torch::kCPU).permute({1,2,0}).contiguous();
    torch::Tensor img_seg_torch = image_info_torch.index_select(0, torch::tensor({4}).to(image_info_torch.device())).to(torch::kCPU).permute({1,2,0}).contiguous();

	//* 转为opencv数据 下面并不是复制，而是引用！！！！
    cv::Mat rgb_cv = cv::Mat{H, W, CV_32FC(3), rgb_torch.data_ptr<float>()}; //* 3通道float
    cv::Mat color_smooth_cv = cv::Mat{H, W, CV_32FC(1), color_smooth_torch.data_ptr<float>()};
    cv::Mat img_seg_cv = cv::Mat{H, W, CV_32FC(1), img_seg_torch.data_ptr<float>()};
	std::unordered_map<float, std::vector<int>> init_img_seg; //* 初始划分
	time_last tim_map;
	tim_map.tic();
	for (int col = 0; col < W; ++col)
    {
        for (int row = 0; row < H; ++row)
        {
            float seg_v = img_seg_cv.ptr<float>(row, col)[0];
			init_img_seg[seg_v].emplace_back(row*W+col);
        }
    }
	tim_map.toc("unordered_map");
	std::vector<float> key_to_erase;
	for (auto iter = init_img_seg.begin(); iter != init_img_seg.end(); ++iter)
	{
		if(iter->second.size()<1e3)
		{
			key_to_erase.emplace_back(iter->first);
		}
    }
	for(int i=0; i<key_to_erase.size(); i++)
	{
		init_img_seg.erase(key_to_erase[i]);
	}
	if(0)
	{
		for (auto iter = init_img_seg.begin(); iter != init_img_seg.end(); ++iter)
		{
			std::cout << iter->first << " " << iter->second.size() << std::endl;
		}
		std::cout<<"init_img_seg.size(): "<<init_img_seg.size()<<std::endl;
		std::vector<float> init_img_seg_show(wh[0]*wh[1], 0.0);
		float tmp_id = 1;
		for (auto iter = init_img_seg.begin(); iter != init_img_seg.end(); ++iter)
		{
			float seg_id = iter->first>0?tmp_id:iter->first;
			for(auto p:iter->second)
			{
				init_img_seg_show[p] = seg_id;
			}
			tmp_id++;
		}
		torch::Tensor init_img_seg_show_torch = torch::from_blob(init_img_seg_show.data(), /*sizes=*/{wh[1], wh[0]}).clone();
		return init_img_seg_show_torch;
	}
	const Image & img = images[img_id];
	if(pts_for_depth.size()==0 || refresh)
	{
		preprocess_for_depth(); //*获得hashmap下的点和在全局下的索引
		//* pts_for_depth.emplace_back(pt); 		pts_for_depth_ids_in_all.emplace_back(i);
	}
	std::vector<float> depth_vct(wh[0]*wh[1], 1000.0);
	std::vector<float3> plane_param_vct(wh[0]*wh[1]);
	float K_tmp[4] = {K[0][0], K[0][2], K[1][1], K[1][2]};
	float T_tmp[12] = {	img.T[0][0], img.T[0][1], img.T[0][2], img.T[0][3],
						img.T[1][0], img.T[1][1], img.T[1][2], img.T[1][3], 
						img.T[2][0], img.T[2][1], img.T[2][2], img.T[2][3], };
	//* cuda
	//* 确定投影点的id，确定那些点能够投影到图像平面
	int opertator_flag = 0;
	std::vector<int> valid_local_ids(pts_for_depth_ids_in_all.size()); //* 局部有效ids
	//* 初次投影得到深度图，二次投影确定深度图与三维点的关联关系
	proj_pts2depth_with_attr(pts_for_depth.data(), valid_local_ids.data(), pts_for_depth.size(), K_tmp, T_tmp, depth_vct.data(), wh[0], wh[1], opertator_flag); //* 7.7824ms
	std::vector<float> depth_new(wh[0]*wh[1], 1000.0);
	depth_remove_outlier_cuda(depth_vct.data(), depth_new.data(), wh[0], wh[1]);
	int num_valid = 0;
	std::vector<int> valid_global_ids;
	for(int i=0; i<pts_for_depth_ids_in_all.size(); i++)
	{
		if(valid_local_ids[i])
		{
			num_valid++;
			valid_global_ids.emplace_back(pts_for_depth_ids_in_all[i]);
		}
	}
	std::vector<float3> valid_pts(valid_global_ids.size());
	std::vector<float3> valid_plane_param(valid_global_ids.size());
	float R_w2c[3][3], R_c2w[3][3];
	float t_w2c[3], t_c2w[3];
	for(int i=0; i<3; i++)
	{
		R_w2c[i][0] = img.T[i][0];
		R_w2c[i][1] = img.T[i][1];
		R_w2c[i][2] = img.T[i][2];
		R_c2w[0][i] = img.T[i][0];
		R_c2w[1][i] = img.T[i][1];
		R_c2w[2][i] = img.T[i][2];
		t_w2c[i] = img.T[i][3];
		t_c2w[i] = -(img.T[0][i]*img.T[0][3] + img.T[1][i]*img.T[1][3] + img.T[2][i]*img.T[2][3]);
	}
	for(int i=0; i<valid_global_ids.size(); i++)
	{
		int id = valid_global_ids[i];
		valid_pts[i].x = m_rgb_pts_vec[id]->m_pos[0];
		valid_pts[i].y = m_rgb_pts_vec[id]->m_pos[1];
		valid_pts[i].z = m_rgb_pts_vec[id]->m_pos[2];
		//* 将平面参数转移到相机坐标系下
		float n_orig[3], n_new[3];
		n_orig[0] = m_rgb_pts_vec[id]->plane_param[0];
		n_orig[1] = m_rgb_pts_vec[id]->plane_param[1];
		n_orig[2] = m_rgb_pts_vec[id]->plane_param[2];
		float scale = 1 + t_c2w[0]*n_orig[0] + t_c2w[1]*n_orig[1] + t_c2w[2]*n_orig[2];
		n_new[0] = (R_w2c[0][0]*n_orig[0] + R_w2c[0][1]*n_orig[1] + R_w2c[0][2]*n_orig[2])/scale;
		n_new[1] = (R_w2c[1][0]*n_orig[0] + R_w2c[1][1]*n_orig[1] + R_w2c[1][2]*n_orig[2])/scale;
		n_new[2] = (R_w2c[2][0]*n_orig[0] + R_w2c[2][1]*n_orig[1] + R_w2c[2][2]*n_orig[2])/scale;
		valid_plane_param[i].x = n_new[0];
		valid_plane_param[i].y = n_new[1];
		valid_plane_param[i].z = n_new[2];

	}
	std::cout<<"num_valid: "<<num_valid<<std::endl;
	for(int i=0; i<plane_param_vct.size(); i++) //* 72.7942ms
	{
		plane_param_vct[i].x=0;
		plane_param_vct[i].y=0;
		plane_param_vct[i].z=0;
	}
	//* 确定三维点与平面点的对应关系
	std::vector<float3> img_to_pc(wh[0]*wh[1]); //* 像素对应的相机坐标系下的三维点坐标
	float3 tmp_p;
	tmp_p.x=0;tmp_p.y=0;tmp_p.z=0;
	for(int i=0;i<img_to_pc.size(); i++)
	{
		img_to_pc[i] = tmp_p;
	}
	for(int i=0; i<valid_pts.size(); i++) //* 72.7942ms
	{
		const float3 & pt_w = valid_pts[i];
		float3 pt_c;
		pt_c.x = T_tmp[0]*pt_w.x + T_tmp[1]*pt_w.y + T_tmp[2] *pt_w.z + T_tmp[3];
		pt_c.y = T_tmp[4]*pt_w.x + T_tmp[5]*pt_w.y + T_tmp[6] *pt_w.z + T_tmp[7];
		pt_c.z = T_tmp[8]*pt_w.x + T_tmp[9]*pt_w.y + T_tmp[10]*pt_w.z + T_tmp[11];
		if(pt_c.z<0.1)
			continue;
		float u,v;
		u = (K_tmp[0]*pt_c.x + K_tmp[1]*pt_c.z)/pt_c.z;
		v = (K_tmp[2]*pt_c.y + K_tmp[3]*pt_c.z)/pt_c.z;
		if(u<0 || v<0 || u>wh[0]-1 || v>wh[1]-1)
			continue;
		int center = (int)((int)v*wh[0] + (int)u);
		if(depth_vct[center]==pt_c.z)
		{
			plane_param_vct[center] = valid_plane_param[i]; //* 局部坐标系下的法线
			img_to_pc[center] = pt_c;
			// std::cout<<"i: "<<i<<", center: "<<center<<std::endl;
		}
	}
	// std::vector<float> cur_fit_mask(wh[0]*wh[1], 0); //* 像素对应的相机坐标系下的三维点坐标
	std::vector<float> depth_fit(wh[0]*wh[1], 0); //* 像素对应的相机坐标系下的三维点坐标
	for (auto iter = init_img_seg.begin(); iter != init_img_seg.end(); ++iter)
	{
		if(iter->first==-1)
		{
			continue;
		}
		std::vector<float3> local_pts;
		for(int i=0; i<iter->second.size(); i++)
		{
			float3 pt = img_to_pc[iter->second[i]];
			if(pt.x>0==0&&pt.y==0&&pt.z==0)
			{
				continue;
			}
			// cur_fit_mask[iter->second[i]] = 1.0;
			local_pts.emplace_back(pt);
		}
		std::cout<<"local_pts.size(): "<<local_pts.size()<<std::endl;
		float4 plane_param = ransac_fit_plane(local_pts.data(), local_pts.size(), 0.01, 10, 100);
		std::vector<int> ids_tmp;
		std::vector<float> d_tmp;
		bool valid_flag = true;
		for(int i=0; i<iter->second.size(); i++)
		{
			int idx = iter->second[i];
			int u = idx%W;
			int v = idx/W;
			float tmp = -1.0/(plane_param.x*(u-K_tmp[1])/K_tmp[0] + plane_param.y*(v-K_tmp[3])/K_tmp[2] + plane_param.z);
			ids_tmp.emplace_back(idx);
			d_tmp.emplace_back(tmp);
			if(tmp>20)
			{
				valid_flag = false;
				break;
			}
			if(depth_new[idx]>0 && fabs(tmp-depth_new[idx])>1)
			{
				valid_flag = false;
				break;
			}
		}
		for(int i=0; i<ids_tmp.size() && valid_flag; i++)
		{
			depth_fit[ids_tmp[i]] = d_tmp[i];
		}
		// break;
    }
	torch::Tensor result = torch::zeros({H, W, 5});
    // //* cv 转为 torch
    // torch::Tensor output = torch::from_blob(depth_new_host, /*sizes=*/{H, W}).clone();
    result.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0,1)}) = torch::from_blob(depth_new.data(),{H, W, 1}).clone().to(result.device());
    result.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(1,2)}) = torch::from_blob(depth_fit.data(),{H, W, 1}).clone().to(result.device());
	return result;

	// depth_comp_with_attr(depth_vct.data(), depth_new.data(), plane_param_vct.data(), K_tmp, wh[0], wh[1]);
    tim.toc("get_depth_with_attr");
	torch::Tensor output = torch::from_blob(depth_new.data(), /*sizes=*/{wh[1], wh[0]}).clone();
	return output;
}

torch::Tensor Global_map::get_depth_with_attr(int64_t img_id, bool refresh)
{
	//* 这种投影方式只是单纯考虑点，而没有考虑点的属性
	time_last tim;
	tim.tic();
	if(img_id>=images.size())
	{
		std::cout<<" Too large img_id: "<<img_id<<", expected to less than: "<<images.size()<<std::endl;
		return torch::empty(0);
	}
	const Image & img = images[img_id];
	if(pts_for_depth.size()==0 || refresh)
	{
		preprocess_for_depth(); //*获得hashmap下的点和在全局下的索引
		//* pts_for_depth.emplace_back(pt); 		pts_for_depth_ids_in_all.emplace_back(i);
	}
	std::vector<float> depth_vct(wh[0]*wh[1], 1000.0);
	std::vector<float3> plane_param_vct(wh[0]*wh[1]);
	float K_tmp[4] = {K[0][0], K[0][2], K[1][1], K[1][2]};
	float T_tmp[12] = {	img.T[0][0], img.T[0][1], img.T[0][2], img.T[0][3],
						img.T[1][0], img.T[1][1], img.T[1][2], img.T[1][3], 
						img.T[2][0], img.T[2][1], img.T[2][2], img.T[2][3], };
	//* cuda
	//* 确定投影点的id，确定那些点能够投影到图像平面
	int opertator_flag = 0;
	std::vector<int> valid_local_ids(pts_for_depth_ids_in_all.size()); //* 局部有效ids
	//* 初次投影得到深度图，二次投影确定深度图与三维点的关联关系
	proj_pts2depth_with_attr(pts_for_depth.data(), valid_local_ids.data(), pts_for_depth.size(), K_tmp, T_tmp, depth_vct.data(), wh[0], wh[1], opertator_flag); //* 7.7824ms
	int num_valid = 0;
	std::vector<int> valid_global_ids;
	for(int i=0; i<pts_for_depth_ids_in_all.size(); i++)
	{
		if(valid_local_ids[i])
		{
			num_valid++;
			valid_global_ids.emplace_back(pts_for_depth_ids_in_all[i]);
		}
	}
	std::vector<float3> valid_pts(valid_global_ids.size());
	std::vector<float3> valid_plane_param(valid_global_ids.size());
	float R_w2c[3][3], R_c2w[3][3];
	float t_w2c[3], t_c2w[3];
	for(int i=0; i<3; i++)
	{
		R_w2c[i][0] = img.T[i][0];
		R_w2c[i][1] = img.T[i][1];
		R_w2c[i][2] = img.T[i][2];
		R_c2w[0][i] = img.T[i][0];
		R_c2w[1][i] = img.T[i][1];
		R_c2w[2][i] = img.T[i][2];
		t_w2c[i] = img.T[i][3];
		t_c2w[i] = -(img.T[0][i]*img.T[0][3] + img.T[1][i]*img.T[1][3] + img.T[2][i]*img.T[2][3]);
	}
	// std::cout<<"T_w2c:"<<std::endl;
	// for(int i=0; i<3; i++)
	// {
	// 	std::cout<<img.T[i][0]<<", "<<img.T[i][1]<<", "<<img.T[i][2]<<", "<<img.T[i][3]<<std::endl;
	// }
	// std::cout<<"R_w2c:"<<std::endl;
	// for(int i=0; i<3; i++)
	// {
	// 	std::cout<<R_w2c[i][0]<<", "<<R_w2c[i][1]<<", "<<R_w2c[i][2]<<std::endl;
	// }
	// std::cout<<"t_c2w:"<<std::endl;
	// std::cout<<t_c2w[0]<<", "<<t_c2w[1]<<", "<<t_c2w[2]<<std::endl;
	for(int i=0; i<valid_global_ids.size(); i++)
	{
		int id = valid_global_ids[i];
		valid_pts[i].x = m_rgb_pts_vec[id]->m_pos[0];
		valid_pts[i].y = m_rgb_pts_vec[id]->m_pos[1];
		valid_pts[i].z = m_rgb_pts_vec[id]->m_pos[2];
		//* 将平面参数转移到相机坐标系下
		float n_orig[3], n_new[3];
		n_orig[0] = m_rgb_pts_vec[id]->plane_param[0];
		n_orig[1] = m_rgb_pts_vec[id]->plane_param[1];
		n_orig[2] = m_rgb_pts_vec[id]->plane_param[2];
		float scale = 1 + t_c2w[0]*n_orig[0] + t_c2w[1]*n_orig[1] + t_c2w[2]*n_orig[2];
		n_new[0] = (R_w2c[0][0]*n_orig[0] + R_w2c[0][1]*n_orig[1] + R_w2c[0][2]*n_orig[2])/scale;
		n_new[1] = (R_w2c[1][0]*n_orig[0] + R_w2c[1][1]*n_orig[1] + R_w2c[1][2]*n_orig[2])/scale;
		n_new[2] = (R_w2c[2][0]*n_orig[0] + R_w2c[2][1]*n_orig[1] + R_w2c[2][2]*n_orig[2])/scale;
		valid_plane_param[i].x = n_new[0];
		valid_plane_param[i].y = n_new[1];
		valid_plane_param[i].z = n_new[2];

	}
	std::cout<<"num_valid: "<<num_valid<<std::endl;
	for(int i=0; i<plane_param_vct.size(); i++) //* 72.7942ms
	{
		plane_param_vct[i].x=0;
		plane_param_vct[i].y=0;
		plane_param_vct[i].z=0;
	}
	//* 确定三维点与平面点的对应关系
	for(int i=0; i<valid_pts.size(); i++) //* 72.7942ms
	{
		const float3 & pt_w = valid_pts[i];
		float3 pt_c;
		pt_c.x = T_tmp[0]*pt_w.x + T_tmp[1]*pt_w.y + T_tmp[2] *pt_w.z + T_tmp[3];
		pt_c.y = T_tmp[4]*pt_w.x + T_tmp[5]*pt_w.y + T_tmp[6] *pt_w.z + T_tmp[7];
		pt_c.z = T_tmp[8]*pt_w.x + T_tmp[9]*pt_w.y + T_tmp[10]*pt_w.z + T_tmp[11];
		if(pt_c.z<0.1)
			continue;
		float u,v;
		u = (K_tmp[0]*pt_c.x + K_tmp[1]*pt_c.z)/pt_c.z;
		v = (K_tmp[2]*pt_c.y + K_tmp[3]*pt_c.z)/pt_c.z;
		if(u<0 || v<0 || u>wh[0]-1 || v>wh[1]-1)
			continue;
		int center = (int)((int)v*wh[0] + (int)u);
		if(depth_vct[center]==pt_c.z)
		{
			plane_param_vct[center] = valid_plane_param[i]; //* 局部坐标系下的法线
			// std::cout<<"i: "<<i<<", center: "<<center<<std::endl;
		}
	}
	std::vector<float> depth_new(wh[0]*wh[1], 1000.0);
	depth_comp_with_attr(depth_vct.data(), depth_new.data(), plane_param_vct.data(), K_tmp, wh[0], wh[1]);
    tim.toc("get_depth_with_attr");
	torch::Tensor output = torch::from_blob(depth_new.data(), /*sizes=*/{wh[1], wh[0]}).clone();
	return output;
}

void Global_map::ikdtree_test()
{
	std::cout<<"Kdtree test!!!!"<<std::endl;
	//* 点云初始化
	ref_pts.reset(new PointCloudXYZINormal());
	// std::cout << "RAND_MAX:" << RAND_MAX << std::endl;
	srand((unsigned)time(NULL));
	// srand((unsigned)time(NULL));
	for (int i = 0; i < 50; i++)
	{
		// std::cout << (rand()/double(RAND_MAX)) << " "; //生成[0,1]范围内的随机数
		PointType point;
		point.x = rand()/double(RAND_MAX);
		point.y = rand()/double(RAND_MAX);
		point.z = rand()/double(RAND_MAX);
		ref_pts->push_back(point);
		// std::cout<<"point: "<<point<<std::endl;
	}
	std::cout << std::endl;
	// for()
	std::cout<<"ref_pts->points.size(): "<<ref_pts->points.size()<<std::endl;;
	if (ikdtree.Root_Node == nullptr)
	{
		ikdtree.set_downsample_param(0.4);
		ikdtree.Build(ref_pts->points);
		std::cout << "~~~~~~~ Initialize Map iKD-Tree ! ~~~~~~~" << std::endl;
		// continue;
		PointType pointSel_tmpt;
		pointSel_tmpt.x = rand()/double(RAND_MAX);
		pointSel_tmpt.y = rand()/double(RAND_MAX);
		pointSel_tmpt.z = rand()/double(RAND_MAX);
		std::vector<float> pointSearchSqDis_surf;
		PointVector points_near;
		ikdtree.Nearest_Search(pointSel_tmpt, NUM_MATCH_POINTS, points_near, pointSearchSqDis_surf);
		std::cout<<"point (x, y, z): "<<pointSel_tmpt.x<<", "<<pointSel_tmpt.y<<", "<<pointSel_tmpt.x<<std::endl;
		for(int i=0;i<pointSearchSqDis_surf.size();i++)
		{
			std::cout<<i<<", dist: "<<pointSearchSqDis_surf[i]<<std::endl;
			std::cout<<"(x, y, z): "<<points_near[i].x<<", "<<points_near[i].y<<", "<<points_near[i].x<<std::endl;
		}
		float max_distance = pointSearchSqDis_surf[NUM_MATCH_POINTS - 1];
	}
}

float4 Global_map::kdtree_fit_plane(float3 *ref_pts_host, int n_pts, double distance_threshold, int64_t ransac_n, int64_t num_iterations)
{
	std::random_device rd;
    uint32_t seed = rd();
    std::mt19937 rnd(seed);  // mersenne_twister_engine
	// int n_pts = ref_pts_host.size();
	// time_last tim;
	// tim.tic();
	std::cout<<"Create ikdtree"<<std::endl;
	ref_pts = std::make_shared<PointCloudXYZINormal>();
	time_last tim;
	tim.tic();
    for(int i=0;i<n_pts;i++)
    {
        PointType point;
        point.x = ref_pts_host[i].x;
        point.y = ref_pts_host[i].y;
        point.z = ref_pts_host[i].z;
        ref_pts->push_back(point);
    }
	tim.toc("data trans");
    if(ikdtree.Root_Node != nullptr)
    {
        ikdtree.myreset();
    }
    if (ikdtree.Root_Node == nullptr)
    {
		tim.tic();
        ikdtree.set_downsample_param(0.4);
        ikdtree.Build(ref_pts->points);
		tim.toc("ikdtree build");
		std::cout << "~~~~~~~ Initialize Map iKD-Tree ! ~~~~~~~" << std::endl;
        std::cout<<"The num of input pts: "<<ref_pts->points.size()<<std::endl;
        // std::cout<<"The num of kdtree pts: "<<ikdtree.size()<<std::endl;
        if(0) //* just a test
        {
            // continue;
            PointType pointSel_tmpt;
            pointSel_tmpt.x = rand()/double(RAND_MAX);
            pointSel_tmpt.y = rand()/double(RAND_MAX);
            pointSel_tmpt.z = rand()/double(RAND_MAX);
            std::vector<float> pointSearchSqDis_surf;
            PointVector points_near;
			tim.tic();
            ikdtree.Nearest_Search(pointSel_tmpt, NUM_MATCH_POINTS, points_near, pointSearchSqDis_surf);
			tim.toc("Nearest_Search");
            std::cout<<"point (x, y, z): "<<pointSel_tmpt.x<<", "<<pointSel_tmpt.y<<", "<<pointSel_tmpt.x<<std::endl;
			std::cout<<"points_near.size(): "<<points_near.size()<<std::endl;
            // for(int i=0;i<pointSearchSqDis_surf.size();i++)
            // {
            //     std::cout<<i<<", dist: "<<pointSearchSqDis_surf[i]<<std::endl;
            //     std::cout<<"(x, y, z): "<<points_near[i].x<<", "<<points_near[i].y<<", "<<points_near[i].x<<std::endl;
            // }
			BoxPointType boxpoint;
			boxpoint.vertex_min[0] = pointSel_tmpt.x - 10;
			boxpoint.vertex_min[1] = pointSel_tmpt.y - 10;
			boxpoint.vertex_min[2] = pointSel_tmpt.z - 10;
			boxpoint.vertex_max[0] = pointSel_tmpt.x + 10;
			boxpoint.vertex_max[1] = pointSel_tmpt.y + 10;
			boxpoint.vertex_max[2] = pointSel_tmpt.z + 10;
			tim.tic();
			ikdtree.Search_by_range(ikdtree.Root_Node, boxpoint, points_near);
			tim.toc("Search_by_range");
			std::cout<<"points_near.size(): "<<points_near.size()<<std::endl;
        }
	}
	// return;

	float3 * selected_pts;
    float3 * ref_pts_cuda;
    float * dist_host;
    float * dist_cuda;

    cudaMalloc((void **)&ref_pts_cuda,    sizeof(float3)  * (n_pts));
    cudaMalloc((void **)&dist_cuda,       sizeof(float)   * (n_pts));
    dist_host = new float[n_pts];
    selected_pts = new float3[ransac_n];

    cudaMemcpy(ref_pts_cuda, ref_pts_host, sizeof(float3) * (n_pts), cudaMemcpyHostToDevice);
    // end = clock();   //结束时间
    // cout<<"time = "<<double(end-start)/CLOCKS_PER_SEC<<"s"<<endl; //* 0.017327s
    float ths = distance_threshold;
    int max_inlies_num=0;
    float4 best_plane_param;
    std::set<int> selected_ids;
    selected_ids.clear();
    std::uniform_int_distribution<uint32_t> int_rand(0, n_pts - 1);
	float radius = 0.5;
    for(int k=0; k<num_iterations; k++)
    {
        //* 首先随机选择一点
		int init_id = int_rand(rnd);
		PointType pointSel_tmpt;
		pointSel_tmpt.x = ref_pts_host[init_id].x;
		pointSel_tmpt.y = ref_pts_host[init_id].y;
		pointSel_tmpt.z = ref_pts_host[init_id].z;
		// std::vector<float> pointSearchSqDis_surf;
		PointVector points_near;
		// tim.tic();
		// ikdtree.Nearest_Search(pointSel_tmpt, NUM_MATCH_POINTS, points_near, pointSearchSqDis_surf);
		// tim.toc("Nearest_Search");
		// std::cout<<"point (x, y, z): "<<pointSel_tmpt.x<<", "<<pointSel_tmpt.y<<", "<<pointSel_tmpt.x<<std::endl;
		// std::cout<<"points_near.size(): "<<points_near.size()<<std::endl;
		BoxPointType boxpoint;
		boxpoint.vertex_min[0] = pointSel_tmpt.x - radius;
		boxpoint.vertex_min[1] = pointSel_tmpt.y - radius;
		boxpoint.vertex_min[2] = pointSel_tmpt.z - radius;
		boxpoint.vertex_max[0] = pointSel_tmpt.x + radius;
		boxpoint.vertex_max[1] = pointSel_tmpt.y + radius;
		boxpoint.vertex_max[2] = pointSel_tmpt.z + radius;
		// tim.tic();
		ikdtree.Search_by_range(ikdtree.Root_Node, boxpoint, points_near);
		// tim.toc("Search_by_range");
		// std::cout<<"points_near.size(): "<<points_near.size()<<std::endl;
		int num_cur_pts = points_near.size();
		if(ransac_n>num_cur_pts)
		{
			continue;
		}
		std::uniform_int_distribution<uint32_t> cur_rand(0, num_cur_pts - 1);
		// return;
		//* 生成随机索引
        if(selected_ids.empty())
        {
            while(selected_ids.size()<ransac_n)
            {
                int id = cur_rand(rnd);
                selected_ids.insert(id);
            }
        }
        //* 点选择
        int selected_pts_id=0;
        for(set<int>::iterator it=selected_ids.begin(); it!=selected_ids.end(); it++)  //使用迭代器进行遍历 
        {
            float3 pt;
            pt.x = points_near[*it].x;
            pt.y = points_near[*it].y;
            pt.z = points_near[*it].z;
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
        fit_plane_params(selected_pts, ransac_n, M);
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
        point_to_plane_dist_cuda(n_pts, ref_pts_cuda, dist_cuda, plane_param);
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
			best_plane_param.x = M[1]/M[0];
			best_plane_param.y = M[2]/M[0];
			best_plane_param.z = -1/M[0];
			best_plane_param.w = 1;
            std::cout<<"k: "<<k<<", inliers_num: "<<inliers_num<<std::endl;
        }

    }
    printf("max_inlies_num: %d, total_num: %d\n\n", max_inlies_num, n_pts);
    //* 获取inlier index
    int inliers_num=0, last_inlier_num=0;
    float4 plane_param = best_plane_param;
    point_to_plane_dist_cuda(n_pts, ref_pts_cuda, dist_cuda, plane_param);
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
            if(dist_host[i]<4*ths)
            {
                const float3 &  pt = ref_pts_host[i];
                float r = plane_param.x*pt.x + plane_param.y*pt.y + plane_param.z*pt.z + 1;
                Hessien[0][0] += pt.x*pt.x; Hessien[0][1] += pt.x*pt.y; Hessien[0][2] += pt.x*pt.z; Hessien[0][3] -= pt.x * r;
                Hessien[1][0] += pt.y*pt.x; Hessien[1][1] += pt.y*pt.y; Hessien[1][2] += pt.y*pt.z; Hessien[1][3] -= pt.y * r;
                Hessien[2][0] += pt.z*pt.x; Hessien[2][1] += pt.z*pt.y; Hessien[2][2] += pt.z*pt.z; Hessien[2][3] -= pt.z * r;
            }
        }
        cal_Ax(3, Hessien);
        plane_param.x += Hessien[0][3];
        plane_param.y += Hessien[1][3];
        plane_param.z += Hessien[2][3];
        plane_param.w = 1;
        point_to_plane_dist_cuda(n_pts, ref_pts_cuda, dist_cuda, plane_param);
        cudaMemcpy(dist_host, dist_cuda, sizeof(float) * n_pts, cudaMemcpyDeviceToHost);
		CheckCUDAError("cudaMemcpy");
        inliers_num = 0;
        for (int i = 0; i < n_pts; i+=1)
        {
            if(dist_host[i]<ths)
            {
                inliers_num++;
            }
        }
        std::cout<<"iter_num: "<<iter_num<<", inliers_num: "<<inliers_num<<std::endl;
		if(max_inlies_num<inliers_num)
		{
			max_inlies_num = inliers_num;
			best_plane_param = plane_param;
		}
		if(fabs(last_inlier_num - inliers_num)<200)
		{
			best_plane_param = plane_param;
			break;
		}
		last_inlier_num = inliers_num;
    }
	cudaFree(ref_pts_cuda);
    cudaFree(dist_cuda);
    // delete [] ref_pts_host;
    delete [] dist_host;
    delete [] selected_pts;
	std::cout<<"Clear ikdtree"<<std::endl;
	ikdtree.myreset();
	std::cout<<"end ransac_fit_plane\n";
	return best_plane_param;
}

torch::Tensor Global_map::ransac_fit_ground_plane(bool refresh, double distance_threshold, int64_t ransac_n, int64_t num_iterations)
{
    // 使用kdtree+ransac确定地面所在的点云，并计算这些点的平面参数：n^T*P + 1 = 0;
	if(pts_for_depth.size()==0 || refresh)
	{
		preprocess_for_depth();
	}
	if(pts_for_depth.size()==0)
	{
		std::cout<<"No points!"<<std::endl;
		return torch::empty(0);
	}
	// std::vector<float3> & all_pts_host = pts_for_depth;
    std::random_device rd;
    uint32_t seed = rd();
    std::mt19937 rnd(seed);  // mersenne_twister_engine
    clock_t start,end, start_fit,end_fit; //定义clock_t变量
    start = clock(); //开始时间
    bool debug = true;
    srand((unsigned)time(NULL));
    // std::cout << (rand()/double(RAND_MAX)) << " "; //生成[0,1]范围内的随机数
    //* basic_pts [N ,3+n]
    int n_all_pts = pts_for_depth.size();
    if(n_all_pts<NUM_MATCH_POINTS)
    {
        std::cout<<"Too few pts!!"<<std::endl;
        return torch::empty(0);
    }
    //* 直通滤波，为了提取地面,提取地面之后，再将地面用kdtree表示，直接处理点云
    int n_pts = 0;
    std::vector<float3> ref_pts_host;
    std::vector<int> ref_pts_ids;
    for(int i=0; i<n_all_pts; i++)
    {
        float3 pt = pts_for_depth[i];
        if(pt.z<1.5)
        {
            ref_pts_host.emplace_back(pt);
            ref_pts_ids.emplace_back(i);
            n_pts++;
        }
    }
    // start = clock(); //开始时间
    float3 * ref_pts_cuda;
    float * dist_host;
    float * dist_cuda;
	dist_host = new float[n_all_pts];
    cudaMalloc((void **)&ref_pts_cuda,    sizeof(float3)  * (n_all_pts));
    cudaMalloc((void **)&dist_cuda,       sizeof(float)   * (n_all_pts));
	float4 plane_param;

	// plane_param = ransac_fit_plane(ref_pts_host.data(), ref_pts_host.size(), distance_threshold, ransac_n, num_iterations);

	plane_param = kdtree_fit_plane(ref_pts_host.data(), ref_pts_host.size(), distance_threshold, ransac_n, num_iterations);

	cudaMemcpy(ref_pts_cuda, pts_for_depth.data(), sizeof(float3) * (n_all_pts), cudaMemcpyHostToDevice);
    CheckCUDAError("cudaMemcpy");

    // point_to_plane_dist_cuda(n_all_pts, ref_pts_cuda, dist_cuda, plane_param);
    point_to_plane_sign_dist_cuda(n_all_pts, ref_pts_cuda, dist_cuda, plane_param);
    CheckCUDAError("point_to_plane_dist_cuda");

    cudaMemcpy(dist_host, dist_cuda, sizeof(float) * n_all_pts, cudaMemcpyDeviceToHost);
    CheckCUDAError("cudaMemcpy");

    std::vector<float> inlier_index_v;
    std::vector<float> edge_index_v;
    int inliers_num = 0;
    for (int i = 0; i < n_all_pts; i+=1)
    {
        if(dist_host[i]>-distance_threshold*2 && dist_host[i]<distance_threshold*15)
        {
            
			if(dist_host[i]<distance_threshold*2)
			{
				inlier_index_v.push_back(i);
				inliers_num++;
				int id_in_all = pts_for_depth_ids_in_all[i];
				m_rgb_pts_vec[id_in_all]->plane_param[0] = plane_param.x;
				m_rgb_pts_vec[id_in_all]->plane_param[1] = plane_param.y;
				m_rgb_pts_vec[id_in_all]->plane_param[2] = plane_param.z;
				m_rgb_pts_vec[id_in_all]->pt_attr = (pt_attr_set)1;
			}
			if(dist_host[i]>distance_threshold*8)
			{
				edge_index_v.emplace_back(i);
			}
        }
    }
	ground_plane_param = plane_param;
	float3 plane_normal;
	plane_normal.x = plane_param.x;
	plane_normal.y = plane_param.y;
	plane_normal.z = plane_param.z;
	float3 v_z;
	v_z.x =0; v_z.y=0; v_z.z=1.0;
	float R[3][3]; //* 将平面旋转到垂直于z轴
	calculate_matrix(plane_normal, v_z, R);
	float3 plane_normal_rotated;
	plane_normal_rotated.x = R[0][0]*plane_normal.x + R[0][1]*plane_normal.y + R[0][2]*plane_normal.z;
	plane_normal_rotated.y = R[1][0]*plane_normal.x + R[1][1]*plane_normal.y + R[1][2]*plane_normal.z;
	plane_normal_rotated.z = R[2][0]*plane_normal.x + R[2][1]*plane_normal.y + R[2][2]*plane_normal.z;
	std::cout<<"After rotate: "<<plane_normal_rotated.x<<", "<<plane_normal_rotated.y<<", "<<plane_normal_rotated.z<<std::endl;
	if(debug && 0)
	{
		float3 v1;
		v1.x = R[0][0]*plane_normal.x + R[0][1]*plane_normal.y + R[0][2]*plane_normal.z;
		v1.y = R[1][0]*plane_normal.x + R[1][1]*plane_normal.y + R[1][2]*plane_normal.z;
		v1.z = R[2][0]*plane_normal.x + R[2][1]*plane_normal.y + R[2][2]*plane_normal.z;
		std::cout<<"After rotate: "<<v1.x<<", "<<v1.y<<", "<<v1.z<<std::endl;
		for (int i = 0; i < inlier_index_v.size() && i<10; i+=1)
		{
			int id = inlier_index_v[i];
			float3 pt = pts_for_depth[id];
			float3 v1;
			v1.x = R[0][0]*pt.x + R[0][1]*pt.y + R[0][2]*pt.z;
			v1.y = R[1][0]*pt.x + R[1][1]*pt.y + R[1][2]*pt.z;
			v1.z = R[2][0]*pt.x + R[2][1]*pt.y + R[2][2]*pt.z;
			std::cout<<"After rotate: "<<v1.x<<", "<<v1.y<<", "<<v1.z<<std::endl;
		}
	}
	//* 先将点云旋转到z平面上
	std::vector<float3> pts_rotated(inlier_index_v.size());
	float grid_size = 1.0;
	Hash_map_3d< long, std::vector<int> > hashmap_tmp;
	//* 较大区域的划分去掉稀疏点
	for (int i = 0; i < inlier_index_v.size(); i+=1)
	{
		int id = inlier_index_v[i];
		const float3 & pt = pts_for_depth[id];
		pts_rotated[i].x = R[0][0]*pt.x + R[0][1]*pt.y + R[0][2]*pt.z;
		pts_rotated[i].y = R[1][0]*pt.x + R[1][1]*pt.y + R[1][2]*pt.z;
		pts_rotated[i].z = R[2][0]*pt.x + R[2][1]*pt.y + R[2][2]*pt.z;
		int box_x = std::round(pts_rotated[i].x / grid_size); //* 1.0m
		int box_y = std::round(pts_rotated[i].y / grid_size);
		int box_z = std::round(pts_rotated[i].z / grid_size);
		if (!hashmap_tmp.if_exist(box_x, box_y, box_z))
		{
			hashmap_tmp.insert(box_x, box_y, box_z, std::vector<int>()); //* m_map_3d_hash_map[x][y][z] = target;
		}
		hashmap_tmp.m_map_3d_hash_map[box_x][box_y][box_z].emplace_back(i);
	}
	std::vector<int> ids_to_save;
	int num_to_del = 50;
	for(auto it : hashmap_tmp.m_map_3d_hash_map) //* <x,<y,<z,std::vector<int>>>>
		for(auto it_it: it.second) //* <y,<z,std::vector<int>>>
			for( auto it_it_it: it_it.second ) //* <z,std::vector<int>>
			{
				if(it_it_it.second.size()>num_to_del)
				{
					ids_to_save.insert(ids_to_save.end(), it_it_it.second.begin(), it_it_it.second.end());
				}
			}
	std::cout<<"ids_to_save.size(): "<<ids_to_save.size()<<std::endl;
	
	//* 计算图像边界
	std::vector<float> inlier_index_v_filter;
	float x_min=0, x_max=0, y_min=0, y_max=0;
	for (int i = 0; i < ids_to_save.size(); i+=1)
	{
		int id = inlier_index_v[ids_to_save[i]];
		inlier_index_v_filter.emplace_back(id);
		const float3 & pt = pts_rotated[i];
		x_min = x_min > pt.x ? pt.x : x_min;
		x_max = x_max < pt.x ? pt.x : x_max;
		y_min = y_min > pt.y ? pt.y : y_min;
		y_max = y_max < pt.y ? pt.y : y_max;
	}
	//* 确定旋转后的边界，以0.01m划分
	float voxel_size = 0.01;
	int img_width = std::ceil((x_max-x_min)/voxel_size);
	int img_height = std::ceil((y_max-y_min)/voxel_size);
	std::vector<float> img(img_width*img_height,0.0);
	std::cout<<"x range: "<<x_min<<", "<<x_max<<std::endl;
	std::cout<<"y range: "<<y_min<<", "<<y_max<<std::endl;
	std::cout<<"img size: "<<img_width<<", "<<img_height<<std::endl;
	//* 下面几句会引起错误： double free or corruption (out) 不知道为什么，难道太大了？？？
	// std::vector<float> img_test(img_width*img_height, 0);
	// for (int i = 0; i < pts_rotated.size(); i+=1)
	// {
	// 	const float3 & pt = pts_rotated[i];
	// 	int px = (int)pt.x;
	// 	int py = (int)pt.y;
	// 	int center = py*img_width + px;
	// 	if(center<img_width*img_height)
	// 		img_test[py*img_width + px] = 1;
	// }
    // torch::Tensor output = torch::from_blob(img_test.data(), /*sizes=*/{img_height, img_width}).clone();

	// getchar();
    std::cout<<"end ransac_plane"<<std::endl;
    // torch::Tensor output = torch::from_blob(inlier_index.ptr<float>(), /*sizes=*/{max_inlies_num, 1}).clone();
    int max_inlies_num = inlier_index_v_filter.size();
    std::cout<<"inlier_index_v_filter.size(): "<<inlier_index_v_filter.size()<<std::endl;
    std::cout<<"inlier_index_v.size(): "<<inlier_index_v.size()<<std::endl;
	data_for_return.edge_index_v = edge_index_v;
	data_for_return.inlier_index_v = inlier_index_v;
	data_for_return.inlier_index_v_filter = inlier_index_v_filter;

	//* edge_index_v  inlier_index_v_filter  inlier_index_v
    torch::Tensor output = torch::from_blob(inlier_index_v.data(), /*sizes=*/{inlier_index_v.size(), 1}).clone();
    cudaFree(ref_pts_cuda);
    cudaFree(dist_cuda);
    // delete [] ref_pts_host;
    delete [] dist_host;
    end = clock();   //结束时间
    cout<<"cal time = "<<float(end-start)/CLOCKS_PER_SEC<<"s"<<endl; //* 0.013133s
    return output;
    // return torch::empty(0);
}

torch::Tensor Global_map::enrich_ground(const torch::Tensor & ground_pts_torch, const torch::Tensor & edge_pts_torch, const torch::Tensor & plane_param_torch)
{
    // 使用kdtree+ransac确定地面所在的点云，并计算这些点的平面参数：n^T*P + 1 = 0;
	// std::vector<float3> & all_pts_host = pts_for_depth;
	bool debug = true;
	auto shape = ground_pts_torch.sizes();
    if(shape.size()!=2 || shape[1]<3)
    {
        std::cout<<"Wrong shape of pts!! Should be Nx3"<<std::endl;
		return torch::empty(0);
    }
    int n_pts = shape[0];
	torch::Tensor ground_pts = ground_pts_torch.clone().to(torch::kCPU).contiguous();
	torch::Tensor edge_pts = edge_pts_torch.clone().to(torch::kCPU).contiguous();
	torch::Tensor plane_param = plane_param_torch.clone().to(torch::kCPU).contiguous();
	auto edge_pts_shape = edge_pts.sizes();
    cv::Mat ground_pts_cv = cv::Mat{shape[0], shape[1], CV_32FC1, ground_pts.data_ptr<float>()};
    cv::Mat edge_pts_cv = cv::Mat{edge_pts_shape[0], edge_pts_shape[1], CV_32FC1, edge_pts.data_ptr<float>()};
    cv::Mat plane_param_cv = cv::Mat{4, 1, CV_32FC1, plane_param.data_ptr<float>()};
	if(debug)
	{
		cout<<"n_pts: "<<n_pts<<endl;
		for(int i=0; i<4; i++)
		{
			std::cout<<plane_param_cv.at<float>(i,0)<<", ";
		}
		std::cout<<std::endl;
	}
	float3 plane_normal;
	plane_normal.x = plane_param_cv.at<float>(0,0);
	plane_normal.y = plane_param_cv.at<float>(1,0);
	plane_normal.z = plane_param_cv.at<float>(2,0);
	float3 v_z;
	v_z.x =0; v_z.y=0; v_z.z=1.0;
	float R[3][3]; //* 将平面旋转到垂直于z轴
	calculate_matrix(plane_normal, v_z, R);
	float3 plane_normal_rotated;
	plane_normal_rotated.x = R[0][0]*plane_normal.x + R[0][1]*plane_normal.y + R[0][2]*plane_normal.z;
	plane_normal_rotated.y = R[1][0]*plane_normal.x + R[1][1]*plane_normal.y + R[1][2]*plane_normal.z;
	plane_normal_rotated.z = R[2][0]*plane_normal.x + R[2][1]*plane_normal.y + R[2][2]*plane_normal.z;
	// std::cout<<"After rotate: "<<plane_normal_rotated.x<<", "<<plane_normal_rotated.y<<", "<<plane_normal_rotated.z<<std::endl;
	if(debug)
	{
		// float3 plane_normal_rotated;
		// plane_normal_rotated.x = R[0][0]*plane_normal.x + R[0][1]*plane_normal.y + R[0][2]*plane_normal.z;
		// plane_normal_rotated.y = R[1][0]*plane_normal.x + R[1][1]*plane_normal.y + R[1][2]*plane_normal.z;
		// plane_normal_rotated.z = R[2][0]*plane_normal.x + R[2][1]*plane_normal.y + R[2][2]*plane_normal.z;
		std::cout<<"After rotate: "<<plane_normal_rotated.x<<", "<<plane_normal_rotated.y<<", "<<plane_normal_rotated.z<<std::endl;
	}
	
	float grid_size = 1.0;
	Hash_map_3d< long, std::vector<int> > hashmap_tmp;
	//* 较大区域的划分去掉稀疏点
	for (int i = 0; i < edge_pts_shape[0]; i+=1)
	{
		int box_x = std::round(edge_pts_cv.at<float>(i,0) / grid_size); //* 1.0m
		int box_y = std::round(edge_pts_cv.at<float>(i,1) / grid_size);
		int box_z = std::round(edge_pts_cv.at<float>(i,2) / grid_size);
		if (!hashmap_tmp.if_exist(box_x, box_y, box_z))
		{
			hashmap_tmp.insert(box_x, box_y, box_z, std::vector<int>()); //* m_map_3d_hash_map[x][y][z] = target;
		}
		hashmap_tmp.m_map_3d_hash_map[box_x][box_y][box_z].emplace_back(i);
	}
	std::vector<int> ids_to_save;
	int num_to_del = 50;
	for(auto it : hashmap_tmp.m_map_3d_hash_map) //* <x,<y,<z,std::vector<int>>>>
		for(auto it_it: it.second) //* <y,<z,std::vector<int>>>
			for( auto it_it_it: it_it.second ) //* <z,std::vector<int>>
			{
				if(it_it_it.second.size()>num_to_del)
				{
					ids_to_save.insert(ids_to_save.end(), it_it_it.second.begin(), it_it_it.second.end());
				}
			}
	std::cout<<"edge_pts_cv size: "<<edge_pts_shape[0]<<std::endl;
	std::cout<<"ids_to_save.size(): "<<ids_to_save.size()<<std::endl;
	std::vector<float> edge_pts_ids_filter;
	for (int i = 0; i < ids_to_save.size(); i+=1)
	{
		edge_pts_ids_filter.emplace_back(ids_to_save[i]);
	}
    // torch::Tensor output = torch::from_blob(edge_pts_ids_filter.data(), /*sizes=*/{edge_pts_ids_filter.size(), 1}).clone();
	// return output;
	std::vector<float3> ground_pts_rotated(shape[0]);
	std::vector<float3> edge_pts_rotated(edge_pts_ids_filter.size());
	float x_min=0, x_max=0, y_min=0, y_max=0;
	for (int i = 0; i < edge_pts_ids_filter.size(); i+=1)
	{
		int id = edge_pts_ids_filter[i];
		float3 pt;
		pt.x = edge_pts_cv.at<float>(id,0);
		pt.y = edge_pts_cv.at<float>(id,1);
		pt.z = edge_pts_cv.at<float>(id,2);
		edge_pts_rotated[i].x = R[0][0]*pt.x + R[0][1]*pt.y + R[0][2]*pt.z;
		edge_pts_rotated[i].y = R[1][0]*pt.x + R[1][1]*pt.y + R[1][2]*pt.z;
		edge_pts_rotated[i].z = R[2][0]*pt.x + R[2][1]*pt.y + R[2][2]*pt.z;
		x_min = x_min > edge_pts_rotated[i].x ? edge_pts_rotated[i].x : x_min;
		x_max = x_max < edge_pts_rotated[i].x ? edge_pts_rotated[i].x : x_max;
		y_min = y_min > edge_pts_rotated[i].y ? edge_pts_rotated[i].y : y_min;
		y_max = y_max < edge_pts_rotated[i].y ? edge_pts_rotated[i].y : y_max;
	}
	for (int i = 0; i < ground_pts_rotated.size(); i+=1)
	{
		float3 pt;
		pt.x = ground_pts_cv.at<float>(i,0);
		pt.y = ground_pts_cv.at<float>(i,1);
		pt.z = ground_pts_cv.at<float>(i,2);
		ground_pts_rotated[i].x = R[0][0]*pt.x + R[0][1]*pt.y + R[0][2]*pt.z;
		ground_pts_rotated[i].y = R[1][0]*pt.x + R[1][1]*pt.y + R[1][2]*pt.z;
		ground_pts_rotated[i].z = R[2][0]*pt.x + R[2][1]*pt.y + R[2][2]*pt.z;
		x_min = x_min > ground_pts_rotated[i].x ? ground_pts_rotated[i].x : x_min;
		x_max = x_max < ground_pts_rotated[i].x ? ground_pts_rotated[i].x : x_max;
		y_min = y_min > ground_pts_rotated[i].y ? ground_pts_rotated[i].y : y_min;
		y_max = y_max < ground_pts_rotated[i].y ? ground_pts_rotated[i].y : y_max;
		if(debug && i<10)
		{
			std::cout<<ground_pts_rotated[i].x<<", "<<ground_pts_rotated[i].y<<", "<<ground_pts_rotated[i].z<<std::endl;
		}
	}
	float voxel_size = 0.1;
	int img_width = std::ceil((x_max-x_min)/voxel_size);
	int img_height = std::ceil((y_max-y_min)/voxel_size);
	// cv::Mat img(img_height, img_width, CV_32FC1, cv::Scalar::all(0));
	std::vector<float> img(img_width*img_height,0.0);
	std::cout<<"x range: "<<x_min<<", "<<x_max<<std::endl;
	std::cout<<"y range: "<<y_min<<", "<<y_max<<std::endl;
	std::cout<<"img size: "<<img_width<<", "<<img_height<<std::endl;
	// //* 下面几句会引起错误： double free or corruption (out) 不知道为什么，难道太大了？？？
	// // std::vector<float> img_test(img_width*img_height, 0);
	for (int i = 0; i < ground_pts_rotated.size(); i+=1)
	{
		float3 pt = ground_pts_rotated[i];
		int px = (int)((pt.x-x_min)/voxel_size);
		int py = (int)((pt.y-y_min)/voxel_size);
		if(px<img_width && py<img_height)
		{
			// img.at<float>(py, px) = 1.0;
			img[py*img_width + px] = 1.0;
		}
	}
	std::cout<<"edge_pts_rotated.size(): "<<edge_pts_rotated.size()<<std::endl;
	for (int i = 0; i < edge_pts_rotated.size(); i+=1)
	{
		float3 pt = edge_pts_rotated[i];
		int px = (int)((pt.x-x_min)/voxel_size);
		int py = (int)((pt.y-y_min)/voxel_size);
		if(px<img_width && py<img_height)
		{
			// img.at<float>(py, px) = 2.0;
			img[py*img_width + px] = 2.0;
		}
	}
	std::vector<float> img_comp = img;
	for(int row=0; row<img_height || row<5; row++)
	{
		for(int col=0; col<img_width; col++)
		{
			if(img_comp[row*img_width + col]>0)
				continue;
			bool valid_flag[8];
			bool edge_flag[8];
			for(int i=0; i<8; i++)
			{
				valid_flag[i] = false;
				edge_flag[i] = false;
			}
			//* 右
			for(int i=1; i<img_width; i++)
			{
				int new_row = row;
				int new_col = col + i;
				if(new_col>=0 && new_row>=0 && new_col<img_width && new_row<img_height)
				{
					if(img[new_row*img_width + new_col]>0)
					{
						if(img[new_row*img_width + new_col]>1)
						{
							edge_flag[0] = true;
						}
						valid_flag[0] = true;
						break;
					}
				}
				else
				{
					break;
				}
			}
			//* 右上
			for(int i=1; i<img_width; i++)
			{
				int new_row = row - i;
				int new_col = col + i;
				if(new_col>=0 && new_row>=0 && new_col<img_width && new_row<img_height)
				{
					if(img[new_row*img_width + new_col]>0)
					{
						if(img[new_row*img_width + new_col]>1)
						{
							edge_flag[1] = true;
						}
						valid_flag[1] = true;
						break;
					}
				}
				else
				{
					break;
				}
			}
			//* 上
			for(int i=1; i<img_width; i++)
			{
				int new_row = row - i;
				int new_col = col ;
				if(new_col>=0 && new_row>=0 && new_col<img_width && new_row<img_height)
				{
					if(img[new_row*img_width + new_col]>0)
					{
						if(img[new_row*img_width + new_col]>1)
						{
							edge_flag[2] = true;
						}
						valid_flag[2] = true;
						break;
					}
				}
				else
				{
					break;
				}
			}
			//* 左上
			for(int i=1; i<img_width; i++)
			{
				int new_row = row - i;
				int new_col = col - i;
				if(new_col>=0 && new_row>=0 && new_col<img_width && new_row<img_height)
				{
					if(img[new_row*img_width + new_col]>0)
					{
						if(img[new_row*img_width + new_col]>1)
						{
							edge_flag[3] = true;
						}
						valid_flag[3] = true;
						break;
					}
				}
				else
				{
					break;
				}
			}
			//* 左
			for(int i=1; i<img_width; i++)
			{
				int new_row = row ;
				int new_col = col - i;
				if(new_col>=0 && new_row>=0 && new_col<img_width && new_row<img_height)
				{
					if(img[new_row*img_width + new_col]>0)
					{
						if(img[new_row*img_width + new_col]>1)
						{
							edge_flag[4] = true;
						}
						valid_flag[4] = true;
						break;
					}
				}
				else
				{
					break;
				}
			}
			//* 左下
			for(int i=1; i<img_width; i++)
			{
				int new_row = row + i;
				int new_col = col - i;
				if(new_col>=0 && new_row>=0 && new_col<img_width && new_row<img_height)
				{
					if(img[new_row*img_width + new_col]>0)
					{
						if(img[new_row*img_width + new_col]>1)
						{
							edge_flag[5] = true;
						}
						valid_flag[5] = true;
						break;
					}
				}
				else
				{
					break;
				}
			}
			//* 下
			for(int i=1; i<img_width; i++)
			{
				int new_row = row + i;
				int new_col = col ;
				if(new_col>=0 && new_row>=0 && new_col<img_width && new_row<img_height)
				{
					if(img[new_row*img_width + new_col]>0)
					{
						if(img[new_row*img_width + new_col]>1)
						{
							edge_flag[6] = true;
						}
						valid_flag[6] = true;
						break;
					}
				}
				else
				{
					break;
				}
			}
			//* 右下
			for(int i=1; i<img_width; i++)
			{
				int new_row = row + i;
				int new_col = col + i;
				if(new_col>=0 && new_row>=0 && new_col<img_width && new_row<img_height)
				{
					if(img[new_row*img_width + new_col]>0)
					{
						if(img[new_row*img_width + new_col]>1)
						{
							edge_flag[7] = true;
						}
						valid_flag[7] = true;
						break;
					}
				}
				else
				{
					break;
				}
			}
			bool valid = true;
			for(int i=0; i<8; i++)
			{
				if(!valid_flag[i])
				{
					valid = false;
					break;
				}
			}
			bool edge_valid = false;
			for(int i=0; i<8; i++)
			{
				if(!edge_flag[i])
				{
					edge_valid = true;
					break;
				}
			}
			if(valid && edge_valid)
			{
				img_comp[row*img_width + col] = 1.0;
			}
			if(!edge_valid)
			{
				img_comp[row*img_width + col] = 3.0;
			}
		}
	}
	std::vector<float> img_comp_filter = img_comp;
	for(int row=0; row<img_height || row<5; row++)
	{
		for(int col=0; col<img_width; col++)
		{
			if(img_comp[row*img_width + col]<3)
				continue;
			int new_rows[8] = {1,1,0,-1,-1,-1,0,1};
			int new_cols[8] = {0,-1,-1,-1,0,1,1,1};
			for(int i=0; i<8; i++)
			{
				int new_row = row + new_rows[i];
				int new_col = col + new_cols[i];
				if(new_col>=0 && new_row>=0 && new_col<img_width && new_row<img_height)
				{
					if(img_comp[new_row*img_width + new_col]==1)
					{
						img_comp_filter[new_row*img_width + new_col] = 3.0;
					}
				}
			}
		}
	}
	float hres_voxel_size = 0.01;
	int hres_img_width = std::ceil((x_max-x_min)/hres_voxel_size);
	int hres_img_height = std::ceil((y_max-y_min)/hres_voxel_size);
	// cv::Mat img(img_height, img_width, CV_32FC1, cv::Scalar::all(0));
	std::vector<float> hres_img(hres_img_width*hres_img_height,0.0);
	float scale = voxel_size / hres_voxel_size;
	for(int row=0; row<hres_img_height; row++)
	{
		for(int col=0; col<hres_img_width; col++)
		{
			int new_row = (int)(row/scale);
			int new_col = (int)(col/scale);
			if(new_col>=0 && new_row>=0 && new_col<img_width && new_row<img_height)
			{
				if(img_comp_filter[new_row*img_width + new_col]==1)
				{
					hres_img[row*hres_img_width + col] = 1.0;
				}
			}
			
		}
	}
	for (int i = 0; i < ground_pts_rotated.size(); i+=1)
	{
		float3 pt = ground_pts_rotated[i];
		int px = (int)((pt.x-x_min)/hres_voxel_size);
		int py = (int)((pt.y-y_min)/hres_voxel_size);
		if(px<hres_img_width && py<hres_img_height)
		{
			// img.at<float>(py, px) = 1.0;
			hres_img[py*hres_img_width + px] = 2.0;
		}
	}
	for (int i = 0; i < edge_pts_rotated.size(); i+=1)
	{
		float3 pt = edge_pts_rotated[i];
		int px = (int)((pt.x-x_min)/hres_voxel_size);
		int py = (int)((pt.y-y_min)/hres_voxel_size);
		if(px<hres_img_width && py<hres_img_height)
		{
			// img.at<float>(py, px) = 1.0;
			hres_img[py*hres_img_width + px] = 3.0;
		}
	}
	std::vector<float> hres_img_comp = hres_img;
	int radius = 10;
	for(int row=0; row<hres_img_height; row++)
	{
		for(int col=0; col<hres_img_width; col++)
		{
			if(hres_img_comp[row*hres_img_width + col]>0)
				continue;
			bool valid_flag[8];
			bool dists[8];
			bool edge_flag[8];
			for(int i=0; i<8; i++)
			{
				valid_flag[i] = false;
				edge_flag[i] = false;
				dists[i] = 0.0;
			}
			//* 右
			for(int i=1; i<radius; i++)
			{
				int new_row = row;
				int new_col = col + i;
				if(new_col>=0 && new_row>=0 && new_col<hres_img_width && new_row<hres_img_height)
				{
					if(hres_img[new_row*hres_img_width + new_col]>0)
					{
						if(hres_img[new_row*hres_img_width + new_col]>2)
						{
							edge_flag[0] = true;
						}
						valid_flag[0] = true;
						dists[0] = i;
						break;
						
					}
				}
				else
				{
					break;
				}
			}
			//* 右上
			for(int i=1; i<radius; i++)
			{
				int new_row = row - i;
				int new_col = col + i;
				if(new_col>=0 && new_row>=0 && new_col<hres_img_width && new_row<hres_img_height)
				{
					if(hres_img[new_row*hres_img_width + new_col]>0)
					{
						if(hres_img[new_row*hres_img_width + new_col]>2)
						{
							edge_flag[1] = true;
						}
						valid_flag[1] = true;
						dists[1] = i;
						break;
					}
				}
				else
				{
					break;
				}
			}
			//* 上
			for(int i=1; i<radius; i++)
			{
				int new_row = row - i;
				int new_col = col ;
				if(new_col>=0 && new_row>=0 && new_col<hres_img_width && new_row<hres_img_height)
				{
					if(hres_img[new_row*hres_img_width + new_col]>0)
					{
						if(hres_img[new_row*hres_img_width + new_col]>2)
						{
							edge_flag[2] = true;
						}
						valid_flag[2] = true;
						dists[2] = i;
						break;
					}
				}
				else
				{
					break;
				}
			}
			//* 左上
			for(int i=1; i<radius; i++)
			{
				int new_row = row - i;
				int new_col = col - i;
				if(new_col>=0 && new_row>=0 && new_col<hres_img_width && new_row<hres_img_height)
				{
					if(hres_img[new_row*hres_img_width + new_col]>0)
					{
						if(hres_img[new_row*hres_img_width + new_col]>2)
						{
							edge_flag[3] = true;
						}
						valid_flag[3] = true;
						dists[3] = i;
						break;
					}
				}
				else
				{
					break;
				}
			}
			//* 左
			for(int i=1; i<radius; i++)
			{
				int new_row = row ;
				int new_col = col - i;
				if(new_col>=0 && new_row>=0 && new_col<hres_img_width && new_row<hres_img_height)
				{
					if(hres_img[new_row*hres_img_width + new_col]>0)
					{
						if(hres_img[new_row*hres_img_width + new_col]>2)
						{
							edge_flag[4] = true;
						}
						valid_flag[4] = true;
						dists[4] = i;
						break;
					}
				}
				else
				{
					break;
				}
			}
			//* 左下
			for(int i=1; i<radius; i++)
			{
				int new_row = row + i;
				int new_col = col - i;
				if(new_col>=0 && new_row>=0 && new_col<hres_img_width && new_row<hres_img_height)
				{
					if(hres_img[new_row*hres_img_width + new_col]>0)
					{
						if(hres_img[new_row*hres_img_width + new_col]>2)
						{
							edge_flag[5] = true;
						}
						valid_flag[5] = true;
						dists[5] = i;
						break;
					}
				}
				else
				{
					break;
				}
			}
			//* 下
			for(int i=1; i<radius; i++)
			{
				int new_row = row + i;
				int new_col = col ;
				if(new_col>=0 && new_row>=0 && new_col<hres_img_width && new_row<hres_img_height)
				{
					if(hres_img[new_row*hres_img_width + new_col]>0)
					{
						if(hres_img[new_row*hres_img_width + new_col]>2)
						{
							edge_flag[6] = true;
						}
						valid_flag[6] = true;
						dists[6] = i;
						break;
					}
				}
				else
				{
					break;
				}
			}
			//* 右下
			for(int i=1; i<radius; i++)
			{
				int new_row = row + i;
				int new_col = col + i;
				if(new_col>=0 && new_row>=0 && new_col<hres_img_width && new_row<hres_img_height)
				{
					if(hres_img[new_row*hres_img_width + new_col]>0)
					{
						if(hres_img[new_row*hres_img_width + new_col]>2)
						{
							edge_flag[7] = true;
						}
						valid_flag[7] = true;
						dists[7] = i;
						break;
					}
				}
				else
				{
					break;
				}
			}
			bool valid = true;
			for(int i=0; i<8; i++)
			{
				if(!valid_flag[i])
				{
					valid = false;
					break;
				}
			}
			bool edge_valid = false;
			for(int i=0; i<8; i++)
			{
				if(!edge_flag[i])
				{
					edge_valid = true;
					break;
				}
			}
			if(valid && edge_valid)
			{
				hres_img_comp[row*hres_img_width + col] = 4.0;
				continue;
			}
			int near_num=0;
			for(int i=0; i<8; i++)
			{
				if(dists[i]>0 && dists[i]<5)
				{
					near_num++;
				}
			}
			if(near_num>5)
			{
				hres_img_comp[row*hres_img_width + col] = 5.0;
			}
		}
	}
	
	float z_value = -1.0/plane_normal_rotated.z;
	std::cout<<"z_value: "<<z_value<<std::endl;
	std::vector<float3> new_ground_pts;
	for(int row=0; row<hres_img_height; row++)
	{
		for(int col=0; col<hres_img_width; col++)
		{
			if(hres_img_comp[row*hres_img_width + col]<1)
				continue;
			float3 pt, pt_r;
			pt.x = x_min + col*hres_voxel_size;
			pt.y = y_min + row*hres_voxel_size;
			pt.z = z_value;
			pt_r.x = R[0][0]*pt.x + R[1][0]*pt.y + R[2][0]*pt.z;
			pt_r.y = R[0][1]*pt.x + R[1][1]*pt.y + R[2][1]*pt.z;
			pt_r.z = R[0][2]*pt.x + R[1][2]*pt.y + R[2][2]*pt.z;
			new_ground_pts.emplace_back(pt_r);
		}
	}
	cv::Mat new_ground_pts_cv(new_ground_pts.size(), 3, CV_32FC1, cv::Scalar::all(0));
	for(int i=0; i<new_ground_pts.size(); i++)
	{
		new_ground_pts_cv.at<float>(i,0) = new_ground_pts[i].x;
		new_ground_pts_cv.at<float>(i,1) = new_ground_pts[i].y;
		new_ground_pts_cv.at<float>(i,2) = new_ground_pts[i].z;
	}
	std::cout<<"Return"<<std::endl;
    torch::Tensor output = torch::from_blob(new_ground_pts_cv.ptr<float>(), /*sizes=*/{new_ground_pts.size(), 3}).clone();
    // torch::Tensor output = torch::from_blob(img.data(), /*sizes=*/{hres_img_height, img_width}).clone();
	torch::Tensor result = torch::zeros({img_height, img_width, 5});
    // //* cv 转为 torch
    // torch::Tensor output = torch::from_blob(hres_img_comp.data(), /*sizes=*/{hres_img_height, hres_img_width}).clone();
    result.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0,1)}) = torch::from_blob(img.data(),{img_height, img_width, 1}).clone().to(result.device());
    result.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(1,2)}) = torch::from_blob(img_comp_filter.data(),{img_height, img_width, 1}).clone().to(result.device());
	// return result;
	return output;

    // std::random_device rd;
    // uint32_t seed = rd();
    // std::mt19937 rnd(seed);  // mersenne_twister_engine
    // clock_t start,end, start_fit,end_fit; //定义clock_t变量
    // start = clock(); //开始时间
    // srand((unsigned)time(NULL));
    // int n_all_pts = pts_for_depth.size();
    // if(n_all_pts<NUM_MATCH_POINTS)
    // {
    //     std::cout<<"Too few pts!!"<<std::endl;
    //     return torch::empty(0);
    // }
    // //* 直通滤波，为了提取地面,提取地面之后，再将地面用kdtree表示，直接处理点云
    // int n_pts = 0;
    // std::vector<float3> ref_pts_host;
    // std::vector<int> ref_pts_ids;
    // for(int i=0; i<n_all_pts; i++)
    // {
    //     float3 pt = pts_for_depth[i];
    //     if(pt.z<1.5)
    //     {
    //         ref_pts_host.emplace_back(pt);
    //         ref_pts_ids.emplace_back(i);
    //         n_pts++;
    //     }
    // }
    // // start = clock(); //开始时间
    // float3 * ref_pts_cuda;
    // float * dist_host;
    // float * dist_cuda;
	// dist_host = new float[n_all_pts];
    // cudaMalloc((void **)&ref_pts_cuda,    sizeof(float3)  * (n_all_pts));
    // cudaMalloc((void **)&dist_cuda,       sizeof(float)   * (n_all_pts));
	// float4 plane_param;
	// // plane_param = ransac_fit_plane(ref_pts_host.data(), ref_pts_host.size(), distance_threshold, ransac_n, num_iterations);
	// plane_param = kdtree_fit_plane(ref_pts_host.data(), ref_pts_host.size(), distance_threshold, ransac_n, num_iterations);
	// cudaMemcpy(ref_pts_cuda, pts_for_depth.data(), sizeof(float3) * (n_all_pts), cudaMemcpyHostToDevice);
    // CheckCUDAError("cudaMemcpy");
    // // point_to_plane_dist_cuda(n_all_pts, ref_pts_cuda, dist_cuda, plane_param);
    // point_to_plane_sign_dist_cuda(n_all_pts, ref_pts_cuda, dist_cuda, plane_param);
    // CheckCUDAError("point_to_plane_dist_cuda");
    // cudaMemcpy(dist_host, dist_cuda, sizeof(float) * n_all_pts, cudaMemcpyDeviceToHost);
    // CheckCUDAError("cudaMemcpy");
    // std::vector<float> inlier_index_v;
    // std::vector<float> edge_index_v;
    // int inliers_num = 0;
    // for (int i = 0; i < n_all_pts; i+=1)
    // {
    //     if(dist_host[i]>-distance_threshold*2 && dist_host[i]<distance_threshold*15)
    //     {
            
	// 		if(dist_host[i]<distance_threshold*2)
	// 		{
	// 			inlier_index_v.push_back(i);
	// 			inliers_num++;
	// 			int id_in_all = pts_for_depth_ids_in_all[i];
	// 			m_rgb_pts_vec[id_in_all]->plane_param[0] = plane_param.x;
	// 			m_rgb_pts_vec[id_in_all]->plane_param[1] = plane_param.y;
	// 			m_rgb_pts_vec[id_in_all]->plane_param[2] = plane_param.z;
	// 			m_rgb_pts_vec[id_in_all]->pt_attr = (pt_attr_set)1;
	// 		}
	// 		if(dist_host[i]>distance_threshold*8)
	// 		{
	// 			edge_index_v.emplace_back(i);
	// 		}
    //     }
    // }
	// ground_plane_param = plane_param;
	// float3 plane_normal;
	// plane_normal.x = plane_param.x;
	// plane_normal.y = plane_param.y;
	// plane_normal.z = plane_param.z;
	// float3 v_z;
	// v_z.x =0; v_z.y=0; v_z.z=1.0;
	// float R[3][3]; //* 将平面旋转到垂直于z轴
	// calculate_matrix(plane_normal, v_z, R);
	// float3 plane_normal_rotated;
	// plane_normal_rotated.x = R[0][0]*plane_normal.x + R[0][1]*plane_normal.y + R[0][2]*plane_normal.z;
	// plane_normal_rotated.y = R[1][0]*plane_normal.x + R[1][1]*plane_normal.y + R[1][2]*plane_normal.z;
	// plane_normal_rotated.z = R[2][0]*plane_normal.x + R[2][1]*plane_normal.y + R[2][2]*plane_normal.z;
	// std::cout<<"After rotate: "<<plane_normal_rotated.x<<", "<<plane_normal_rotated.y<<", "<<plane_normal_rotated.z<<std::endl;
	// if(debug && 0)
	// {
	// 	float3 v1;
	// 	v1.x = R[0][0]*plane_normal.x + R[0][1]*plane_normal.y + R[0][2]*plane_normal.z;
	// 	v1.y = R[1][0]*plane_normal.x + R[1][1]*plane_normal.y + R[1][2]*plane_normal.z;
	// 	v1.z = R[2][0]*plane_normal.x + R[2][1]*plane_normal.y + R[2][2]*plane_normal.z;
	// 	std::cout<<"After rotate: "<<v1.x<<", "<<v1.y<<", "<<v1.z<<std::endl;
	// 	for (int i = 0; i < inlier_index_v.size() && i<10; i+=1)
	// 	{
	// 		int id = inlier_index_v[i];
	// 		float3 pt = pts_for_depth[id];
	// 		float3 v1;
	// 		v1.x = R[0][0]*pt.x + R[0][1]*pt.y + R[0][2]*pt.z;
	// 		v1.y = R[1][0]*pt.x + R[1][1]*pt.y + R[1][2]*pt.z;
	// 		v1.z = R[2][0]*pt.x + R[2][1]*pt.y + R[2][2]*pt.z;
	// 		std::cout<<"After rotate: "<<v1.x<<", "<<v1.y<<", "<<v1.z<<std::endl;
	// 	}
	// }
	// //* 先将点云旋转到z平面上
	// std::vector<float3> pts_rotated(inlier_index_v.size());
	// float grid_size = 1.0;
	// Hash_map_3d< long, std::vector<int> > hashmap_tmp;
	// //* 较大区域的划分去掉稀疏点
	// for (int i = 0; i < inlier_index_v.size(); i+=1)
	// {
	// 	int id = inlier_index_v[i];
	// 	const float3 & pt = pts_for_depth[id];
	// 	pts_rotated[i].x = R[0][0]*pt.x + R[0][1]*pt.y + R[0][2]*pt.z;
	// 	pts_rotated[i].y = R[1][0]*pt.x + R[1][1]*pt.y + R[1][2]*pt.z;
	// 	pts_rotated[i].z = R[2][0]*pt.x + R[2][1]*pt.y + R[2][2]*pt.z;
	// 	int box_x = std::round(pts_rotated[i].x / grid_size); //* 1.0m
	// 	int box_y = std::round(pts_rotated[i].y / grid_size);
	// 	int box_z = std::round(pts_rotated[i].z / grid_size);
	// 	if (!hashmap_tmp.if_exist(box_x, box_y, box_z))
	// 	{
	// 		hashmap_tmp.insert(box_x, box_y, box_z, std::vector<int>()); //* m_map_3d_hash_map[x][y][z] = target;
	// 	}
	// 	hashmap_tmp.m_map_3d_hash_map[box_x][box_y][box_z].emplace_back(i);
	// }
	// std::vector<int> ids_to_save;
	// int num_to_del = 50;
	// for(auto it : hashmap_tmp.m_map_3d_hash_map) //* <x,<y,<z,std::vector<int>>>>
	// 	for(auto it_it: it.second) //* <y,<z,std::vector<int>>>
	// 		for( auto it_it_it: it_it.second ) //* <z,std::vector<int>>
	// 		{
	// 			if(it_it_it.second.size()>num_to_del)
	// 			{
	// 				ids_to_save.insert(ids_to_save.end(), it_it_it.second.begin(), it_it_it.second.end());
	// 			}
	// 		}
	// std::cout<<"ids_to_save.size(): "<<ids_to_save.size()<<std::endl;
	
	// //* 计算图像边界
	// std::vector<float> inlier_index_v_filter;
	// float x_min=0, x_max=0, y_min=0, y_max=0;
	// for (int i = 0; i < ids_to_save.size(); i+=1)
	// {
	// 	int id = inlier_index_v[ids_to_save[i]];
	// 	inlier_index_v_filter.emplace_back(id);
	// 	const float3 & pt = pts_rotated[i];
	// 	x_min = x_min > pt.x ? pt.x : x_min;
	// 	x_max = x_max < pt.x ? pt.x : x_max;
	// 	y_min = y_min > pt.y ? pt.y : y_min;
	// 	y_max = y_max < pt.y ? pt.y : y_max;
	// }
	// //* 确定旋转后的边界，以0.01m划分
	// float voxel_size = 0.01;
	// int img_width = std::ceil((x_max-x_min)/voxel_size);
	// int img_height = std::ceil((y_max-y_min)/voxel_size);
	// std::vector<float> img(img_width*img_height,0.0);
	// std::cout<<"x range: "<<x_min<<", "<<x_max<<std::endl;
	// std::cout<<"y range: "<<y_min<<", "<<y_max<<std::endl;
	// std::cout<<"img size: "<<img_width<<", "<<img_height<<std::endl;
	// //* 下面几句会引起错误： double free or corruption (out) 不知道为什么，难道太大了？？？
	// // std::vector<float> img_test(img_width*img_height, 0);
	// // for (int i = 0; i < pts_rotated.size(); i+=1)
	// // {
	// // 	const float3 & pt = pts_rotated[i];
	// // 	int px = (int)pt.x;
	// // 	int py = (int)pt.y;
	// // 	int center = py*img_width + px;
	// // 	if(center<img_width*img_height)
	// // 		img_test[py*img_width + px] = 1;
	// // }
    // // torch::Tensor output = torch::from_blob(img_test.data(), /*sizes=*/{img_height, img_width}).clone();

	// // getchar();
    // std::cout<<"end ransac_plane"<<std::endl;
    // // torch::Tensor output = torch::from_blob(inlier_index.ptr<float>(), /*sizes=*/{max_inlies_num, 1}).clone();
    // int max_inlies_num = inlier_index_v_filter.size();
    // std::cout<<"inlier_index_v_filter.size(): "<<inlier_index_v_filter.size()<<std::endl;
    // std::cout<<"inlier_index_v.size(): "<<inlier_index_v.size()<<std::endl;
	// data_for_return.edge_index_v = edge_index_v;
	// data_for_return.inlier_index_v = inlier_index_v;
	// data_for_return.inlier_index_v_filter = inlier_index_v_filter;

	// //* edge_index_v  inlier_index_v_filter  inlier_index_v
    // torch::Tensor output = torch::from_blob(inlier_index_v.data(), /*sizes=*/{inlier_index_v.size(), 1}).clone();
    // cudaFree(ref_pts_cuda);
    // cudaFree(dist_cuda);
    // // delete [] ref_pts_host;
    // delete [] dist_host;
    // end = clock();   //结束时间
    // cout<<"cal time = "<<float(end-start)/CLOCKS_PER_SEC<<"s"<<endl; //* 0.013133s
    // return output;
    return torch::empty(0);
}

torch::Tensor Global_map::bilinear_interplote_depth(const torch::Tensor & depth_torch_in)
{
	// time_last tim;
	torch::Tensor depth_torch = depth_torch_in.clone().to(torch::kCPU).contiguous();
	float K_tmp[4] = {K[0][0], K[0][2], K[1][1], K[1][2]};
	// tim.tic();
	bilinear_interplote_depth_comp(depth_torch.data_ptr<float>(), K_tmp, wh[0], wh[1]); //* 7.7824ms
	// cout << "bilinear_interplote_depth cost time = " << tim.toc() << "ms" << endl;
    // torch::Tensor output = torch::from_blob(depth_vct.data(), /*sizes=*/{wh[1], wh[0]}).clone();
	return depth_torch;
}

torch::Tensor Global_map::bilinear_interplote_depth_id(int64_t img_id, bool refresh)
{
	// time_last tim;
	if(img_id>=images.size())
	{
		std::cout<<" Too large img_id: "<<img_id<<", expected to less than: "<<images.size()<<std::endl;
		return torch::empty(0);
	}
	torch::Tensor depth_torch = get_depth(img_id, false);
	float K_tmp[4] = {K[0][0], K[0][2], K[1][1], K[1][2]};
	// tim.tic();
	bilinear_interplote_depth_comp(depth_torch.data_ptr<float>(), K_tmp, wh[0], wh[1]); //* 7.7824ms
	// cout << "bilinear_interplote_depth_id cost time = " << tim.toc() << "ms" << endl;
    // torch::Tensor output = torch::from_blob(depth_vct.data(), /*sizes=*/{wh[1], wh[0]}).clone();
	return depth_torch;
}

void Global_map::preprocess_for_depth()
{
	// time_last tim;
	// tim.tic();
	long pt_size = m_rgb_pts_vec.size();
	pts_for_depth.clear();
	pts_for_depth_ids_in_all.clear();
	for (long i = pt_size - 1; i >= 0; i--)
	{
		if (i % 1000 == 0)
		{
			cout << ANSI_DELETE_CURRENT_LINE << "Converting pts to float3 " << (int)((pt_size - 1 - i) * 100.0 / (pt_size - 1)) << " % ...";
			fflush(stdout);
		}

		if (m_rgb_pts_vec[i] == nullptr)
		{
			continue;
		}
		// PointXYZRGBA pt;
		float3 pt;
		pt.x = m_rgb_pts_vec[i]->m_pos[0];
		pt.y = m_rgb_pts_vec[i]->m_pos[1];
		pt.z = m_rgb_pts_vec[i]->m_pos[2];
		pts_for_depth.emplace_back(pt);
		pts_for_depth_ids_in_all.emplace_back(i);
	}
	cout << ANSI_DELETE_CURRENT_LINE << "Converting pts to float3 100% ..." << endl;
	cout << "Total have " << pts_for_depth.size() << " points." << endl;
	// cout << "Convert to preprocess_for_depth cost time = " << tim.toc() << "ms" << endl;
}

torch::Tensor Global_map::get_ext(int64_t img_id)
{
	if(img_id>=images.size())
	{
		std::cout<<" Too large img_id: "<<img_id<<", expected to less than: "<<images.size()<<std::endl;
		return torch::empty(0);
	}
	const Image & img = images[img_id];
	cv::Mat T_cv(4, 4, CV_32FC1, cv::Scalar::all(0));
	for(int i=0; i<4; i++)
    {
		for(int j=0; j<4; j++)
		{
			T_cv.ptr<float>(i, j)[0] = img.T[i][j];
		}
    }
	torch::Tensor output = torch::from_blob(T_cv.ptr<float>(), /*sizes=*/{4,4}).clone();
	return output;
}

torch::Tensor Global_map::get_image(int64_t img_id)
{
	if(img_id>=images.size())
	{
		std::cout<<" Too large img_id: "<<img_id<<", expected to less than: "<<images.size()<<std::endl;
		return torch::empty(0);
	}
	cv::Mat image_uint8 = images[img_id].read_color_img();
	cv::Mat image;
	image_uint8.convertTo(image, CV_32FC3);
	torch::Tensor output = torch::from_blob(image.ptr<float>(), /*sizes=*/{wh[1], wh[0], 3}).clone();
	return output;
}

torch::Tensor Global_map::get_depth(int64_t img_id, bool refresh)
{
	//* 这种投影方式只是单纯考虑点，而没有考虑点的属性
	// time_last tim;
	if(img_id>=images.size())
	{
		std::cout<<" Too large img_id: "<<img_id<<", expected to less than: "<<images.size()<<std::endl;
		return torch::empty(0);
	}
	const Image & img = images[img_id];
	if(pts_for_depth.size()==0 || refresh)
	{
		preprocess_for_depth();
	}
	std::vector<float> depth_vct(wh[0]*wh[1], 1000.0);
	float K_tmp[4] = {K[0][0], K[0][2], K[1][1], K[1][2]};
	float T_tmp[12] = {	img.T[0][0], img.T[0][1], img.T[0][2], img.T[0][3],
						img.T[1][0], img.T[1][1], img.T[1][2], img.T[1][3], 
						img.T[2][0], img.T[2][1], img.T[2][2], img.T[2][3], };
	// tim.tic();
	//* cuda
	proj_pts2depth(pts_for_depth.data(), pts_for_depth.size(), K_tmp, T_tmp, depth_vct.data(), wh[0], wh[1]); //* 7.7824ms
	
	//* c++
	// for(int i=0; i<pts_for_depth.size(); i++) //* 72.7942ms
	// {
	// 	const float3 & pt_w = pts_for_depth[i];
	// 	float3 pt_c;
	// 	pt_c.x = T_tmp[0]*pt_w.x + T_tmp[1]*pt_w.y + T_tmp[2] *pt_w.z + T_tmp[3];
	// 	pt_c.y = T_tmp[4]*pt_w.x + T_tmp[5]*pt_w.y + T_tmp[6] *pt_w.z + T_tmp[7];
	// 	pt_c.z = T_tmp[8]*pt_w.x + T_tmp[9]*pt_w.y + T_tmp[10]*pt_w.z + T_tmp[11];
	// 	if(pt_c.z<0.1)
	// 		continue;
	// 	float u,v;
	// 	u = (K_tmp[0]*pt_c.x + K_tmp[1]*pt_c.z)/pt_c.z;
	// 	v = (K_tmp[2]*pt_c.y + K_tmp[3]*pt_c.z)/pt_c.z;
	// 	if(u<0 || v<0 || u>wh[0]-1 || v>wh[1]-1)
	// 		continue;
	// 	int center = (int)((int)v*wh[0] + (int)u);
	// 	if(depth_vct[center]>pt_c.z)
	// 	{
	// 		depth_vct[center] = pt_c.z;
	// 	}
	// }
	// cout << "get_depth cost time = " << tim.toc() << "ms" << endl;
    torch::Tensor output = torch::from_blob(depth_vct.data(), /*sizes=*/{wh[1], wh[0]}).clone();
	return output;
}

void Global_map::delete_pt(float x, float y, float z)
{
	int grid_x = std::round(x / m_minimum_pts_size); //* 0.01m
	int grid_y = std::round(y / m_minimum_pts_size);
	int grid_z = std::round(z / m_minimum_pts_size);
	int box_x = std::round(x / m_voxel_resolution); //* 0.1m
	int box_y = std::round(y / m_voxel_resolution);
	int box_z = std::round(z / m_voxel_resolution);
	std::shared_ptr<RGB_pts> pt_rgb_tmp = nullptr;
	if(m_hashmap_3d_pts.get(grid_x, grid_y, grid_z, &pt_rgb_tmp))
	{
		int m_pt_index = pt_rgb_tmp->m_pt_index;
		m_hashmap_voxels.m_map_3d_hash_map[box_x][box_y][box_z]->erase_pt(m_pt_index);
		m_hashmap_3d_pts.erase(grid_x, grid_y, grid_z);
		m_rgb_pts_vec[m_pt_index] = nullptr;
	}
	else
	{
		return;
	}
}

void Global_map::append_points_to_global_map(const torch::Tensor & src_pts_torch)
{
	// time_last tim;
	// tim.tic();
	std::vector<std::shared_ptr<RGB_pts>> *pts_added_vec = nullptr;
	int step = 1;
	float added_time = 0.0;
    auto shape = src_pts_torch.sizes();
    if(shape.size()!=2 || shape[1]<6)
    {
        std::cout<<"Wrong shape of pts!! Should be Nx6"<<std::endl;
		return;
    }
    int n_pts = shape[0];
	cout<<"n_pts: "<<n_pts<<endl;
	torch::Tensor src_pts = src_pts_torch.clone().to(torch::kCPU).contiguous();
    cv::Mat src_pts_cv = cv::Mat{shape[0], shape[1], CV_32FC1, src_pts.data_ptr<float>()};
	m_in_appending_pts = 1;
	if (pts_added_vec != nullptr) //* 用于保存最近添加的点
		pts_added_vec->clear();
	std::unordered_set<std::shared_ptr<RGB_Voxel>> voxels_recent_visited; //* 最近访问的大格子，无序集合
	if (m_recent_visited_voxel_activated_time == 0)
		//* 0 最近访问大格子的保留时间，0表示不保留，直接清空，只用最新帧的结果
		voxels_recent_visited.clear();
	else
	{
		// m_mutex_m_box_recent_hitted->lock();
		// voxels_recent_visited = m_voxels_recent_visited;
		// m_mutex_m_box_recent_hitted->unlock();
		for (Voxel_set_iterator it = voxels_recent_visited.begin(); it != voxels_recent_visited.end();)
		{
			//* 最近访问的大格子中超过保留时间的被清除
			if (added_time - (*it)->m_last_visited_time > m_recent_visited_voxel_activated_time)
			{
				it = voxels_recent_visited.erase(it);
				continue;
			}
			it++;
		}
		cout << "Restored voxel number = " << voxels_recent_visited.size() << endl;
	}
	int number_of_voxels_before_add = voxels_recent_visited.size();
	// step = 4;
	for (int pt_idx = 0; pt_idx < n_pts; pt_idx += step)
	{
		if (pt_idx % 1000 == 0)
		{
			cout << ANSI_DELETE_CURRENT_LINE << "Inserting points " << (int)((pt_idx) * 100.0 / (n_pts)) << " % ...";
			fflush(stdout);
		}
		// cout<<"pt_idx: "<<pt_idx<<endl;
		int add = 1;
		int grid_x = std::round(src_pts_cv.at<float>(pt_idx,0) / m_minimum_pts_size); //* 0.01m
		int grid_y = std::round(src_pts_cv.at<float>(pt_idx,1) / m_minimum_pts_size);
		int grid_z = std::round(src_pts_cv.at<float>(pt_idx,2) / m_minimum_pts_size);
		int box_x = std::round(src_pts_cv.at<float>(pt_idx,0) / m_voxel_resolution); //* 0.1m
		int box_y = std::round(src_pts_cv.at<float>(pt_idx,1) / m_voxel_resolution);
		int box_z = std::round(src_pts_cv.at<float>(pt_idx,2) / m_voxel_resolution);
		//* 用一个map嵌套来构造网格，牛
		//* m_hashmap_3d_pts将空间划分为很多小的格子，每个小格子的数据就是一个rgb点
		//* 小格子是具有唯一性的，每个格子只容纳一个点，这样可以保证数据的稀疏性！
		//* m_hashmap_voxels将空间分为较大的格子，每个大格子包含多个指向rgb点的指针和上次访问时间,
		//* template <typename data_type = float, typename T = void *>
		//* std::unordered_map<data_type, std::unordered_map<data_type, std::unordered_map<data_type, T>>>
		//* Hash_map_3d< long, RGB_pt_ptr >   m_hashmap_3d_pts;
		//* 如果当前点对应的小格子已经存在，说明这个小格子中已经有一个点，这个格子中不再添加新数据，将格子中的点加入最近访问点集pts_added_vec中
		if (m_hashmap_3d_pts.if_exist(grid_x, grid_y, grid_z))
		{
			add = 0;
			//* 引用计数 4
			if (pts_added_vec != nullptr)
				pts_added_vec->push_back(m_hashmap_3d_pts.m_map_3d_hash_map[grid_x][grid_y][grid_z]); //* 相当于复制了一份指针,多了一个引用计数
		}
		//* 如果该点对应的大格子不存在，则创建，否则返回大格子的指针
		RGB_voxel_ptr box_ptr; //* 0
		if (!m_hashmap_voxels.if_exist(box_x, box_y, box_z))
		{
			box_ptr = std::make_shared<RGB_Voxel>();			   //* 1
			box_ptr->pose = vec_3(box_x*m_voxel_resolution, box_y*m_voxel_resolution, box_z*m_voxel_resolution);
			m_hashmap_voxels.insert(box_x, box_y, box_z, box_ptr); //* m_map_3d_hash_map[x][y][z] = target;
		}
		else
			box_ptr = m_hashmap_voxels.m_map_3d_hash_map[box_x][box_y][box_z];
		//* voxels_recent_visited保存最近访问的大格子信息
		voxels_recent_visited.insert(box_ptr); //* 增加一个引用计数 3
		//* voxels_recent_visited 保存最近被访问的体素box的指针，每个体素中包含多个指向rgb点的指针和上次访问时间,每个rgb点包含很多信息
		//* 这里保存的只是指针，所以下面box_ptr修改的数据就是保存的指针指向的数据，可以提前将指针保存，而不需要数据修改结束之后再保存指针，与直接保存数据不同！！！
		// voxels_recent_visited_between_keyframes.emplace(box_ptr); //* 向容器中添加新元素，效率比 insert() 方法高。
		box_ptr->m_last_visited_time = added_time;
		//* 记录大格子最近被访问的时间
		if (add == 0)
			continue;
		//* 强引用, 用来记录当前有多少个存活的 shared_ptrs 正持有该对象. 共享的对象会在最后一个强引用离开的时候销毁( 也可能释放).
		//* 传递指针比传递数据更加高效，虽然创建数据的函数的生命周期结束了，但是还有指向数据的指针，只要这些指针还在，数据就不会被销毁！！
		std::shared_ptr<RGB_pts> pt_rgb = std::make_shared<RGB_pts>(); //* 1
		pt_rgb->set_pos(vec_3(src_pts_cv.at<float>(pt_idx,0), src_pts_cv.at<float>(pt_idx,1), src_pts_cv.at<float>(pt_idx,2)));
		pt_rgb->update_rgb(vec_3(src_pts_cv.at<float>(pt_idx,3), src_pts_cv.at<float>(pt_idx,4), src_pts_cv.at<float>(pt_idx,5)));
		pt_rgb->m_pt_index = m_rgb_pts_vec.size(); //* 每个点的唯一标识
		m_rgb_pts_vec.push_back(pt_rgb); //* 记录新加入的点信息 //* 2
		m_hashmap_3d_pts.insert(grid_x, grid_y, grid_z, pt_rgb); //* 往小格子中加入点 //* 3
		box_ptr->add_pt(pt_rgb);								 //* 往大格子中加入点 //* 4
		//* 简单debug
		if(debug && 0)
		{
			std::shared_ptr<RGB_pts> pt_rgb_tmp = nullptr;
			if(m_hashmap_3d_pts.get(grid_x, grid_y, grid_z, &pt_rgb_tmp))
			{
				std::cout<<"exist: "<<grid_x<<", "<<grid_y<<", "<<grid_z<<std::endl;
				std::cout<<"m_pt_index: "<<pt_rgb_tmp->m_pt_index<<std::endl;
			}
			else
			{
				std::cout<<"not exist: "<<grid_x<<", "<<grid_y<<", "<<grid_z<<std::endl;
			}
			m_hashmap_3d_pts.erase(grid_x, grid_y, grid_z);
			std::cout<<"After erase"<<std::endl;
			if(m_hashmap_3d_pts.get(grid_x, grid_y, grid_z, &pt_rgb_tmp))
			{
				std::cout<<"exist: "<<grid_x<<", "<<grid_y<<", "<<grid_z<<std::endl;
				std::cout<<"m_pt_index: "<<pt_rgb_tmp->m_pt_index<<std::endl;
			}
			else
			{
				std::cout<<"not exist: "<<grid_x<<", "<<grid_y<<", "<<grid_z<<std::endl;
			}
			// debug = false;
		}
		//* 删除的时候vector里面的指针设置为空，hashmap中直接删除
		if (pts_added_vec != nullptr)
			pts_added_vec->push_back(pt_rgb); //* 将新的点加入最近访问的点集中 //* 5
	}
	cout << ANSI_DELETE_CURRENT_LINE << "Inserting points 100% ..." << endl;
	// cout<<"m_hashmap_3d_pts.total_size():"<<m_hashmap_3d_pts.total_size()<<endl;//* 不断增长
	//* m_rgb_pts_vec 保存所有点的信息，用于发布彩色点云和保存点云
	cout<<"m_rgb_pts_vec.size(): "<<m_rgb_pts_vec.size() << std::endl;
	m_in_appending_pts = 0; //* 添加点结束
	m_mutex_m_box_recent_hitted->lock();
	m_voxels_recent_visited = voxels_recent_visited;
	m_mutex_m_box_recent_hitted->unlock();
	// cout << "Save torch to hashmap cost time = " << tim.toc() << "ms" << endl;
	// return (m_voxels_recent_visited.size() - number_of_voxels_before_add);
}

torch::Tensor Global_map::get_pc()
{
	// time_last tim;
	// tim.tic();
	cout << "Return Rgb points " << endl;
	long pt_size = m_rgb_pts_vec.size();
	cv::Mat return_pts_cv(pt_size, 6, CV_32FC1, cv::Scalar::all(0));
	long pt_count = 0;
	for (long i = pt_size - 1; i > 0; i--)
	{
		if (i % 1000 == 0)
		{
			cout << ANSI_DELETE_CURRENT_LINE << "Saving offline map " << (int)((pt_size - 1 - i) * 100.0 / (pt_size - 1)) << " % ...";
			fflush(stdout);
		}
		if (m_rgb_pts_vec[i] == nullptr)
		{
			continue;
		}
		return_pts_cv.at<float>(pt_count, 0) = m_rgb_pts_vec[i]->m_pos[0];
		return_pts_cv.at<float>(pt_count, 1) = m_rgb_pts_vec[i]->m_pos[1];
		return_pts_cv.at<float>(pt_count, 2) = m_rgb_pts_vec[i]->m_pos[2];
		return_pts_cv.at<float>(pt_count, 3) = m_rgb_pts_vec[i]->m_rgb[0];
		return_pts_cv.at<float>(pt_count, 4) = m_rgb_pts_vec[i]->m_rgb[1];
		return_pts_cv.at<float>(pt_count, 5) = m_rgb_pts_vec[i]->m_rgb[2];
		pt_count++;
	}
	if(pts_for_depth.size()!=pt_count)
	{
		preprocess_for_depth();
	}
	cout << ANSI_DELETE_CURRENT_LINE << "Saving offline map 100% ..." << endl;
	cout << "Total have " << pt_count << " points." << endl;
	// cout << "Convert to torch cost time = " << tim.toc() << "ms" << endl;
    torch::Tensor output = torch::from_blob(return_pts_cv.ptr<float>(), /*sizes=*/{pt_count, 6}).clone();
	return output;
}

bool Global_map::is_busy()
{
	return m_in_appending_pts;
}

void Global_map::save_to_pcd(std::string dir_name, std::string _file_name, int save_pts_with_views)
{
	// time_last tim;
	// Common_tools::create_dir(dir_name);
	std::string file_name = std::string(dir_name).append(_file_name);
	scope_color(ANSI_COLOR_BLUE_BOLD);
	cout << "Save Rgb points to " << file_name << endl;
	fflush(stdout);
	PointCloud<PointXYZRGBA> pc_rgb;
	long pt_size = m_rgb_pts_vec.size();
	pc_rgb.resize(pt_size);
	long pt_count = 0;
	for (long i = pt_size - 1; i > 0; i--)
	//for (int i = 0; i  <  pt_size; i++)
	{
		if (i % 1000 == 0)
		{
			cout << ANSI_DELETE_CURRENT_LINE << "Saving offline map " << (int)((pt_size - 1 - i) * 100.0 / (pt_size - 1)) << " % ...";
			fflush(stdout);
		}

		if (m_rgb_pts_vec[i] == nullptr)
		{
			continue;
		}

		if (m_rgb_pts_vec[i]->m_N_rgb < save_pts_with_views)
		{
			continue;
		}
		PointXYZRGBA pt;
		pc_rgb.points[pt_count].x = m_rgb_pts_vec[i]->m_pos[0];
		pc_rgb.points[pt_count].y = m_rgb_pts_vec[i]->m_pos[1];
		pc_rgb.points[pt_count].z = m_rgb_pts_vec[i]->m_pos[2];
		pc_rgb.points[pt_count].r = m_rgb_pts_vec[i]->m_rgb[2];
		pc_rgb.points[pt_count].g = m_rgb_pts_vec[i]->m_rgb[1];
		pc_rgb.points[pt_count].b = m_rgb_pts_vec[i]->m_rgb[0];
		pt_count++;
	}
	cout << ANSI_DELETE_CURRENT_LINE << "Saving offline map 100% ..." << endl;
	pc_rgb.resize(pt_count);
	cout << "Total have " << pt_count << " points." << endl;
	// tim.tic();
	// pcl::io::savePCDFileBinary(std::string(file_name).append(".pcd"), pc_rgb);
	// pcl::io::savePLYFile(std::string(file_name).append(".ply"), pc_rgb);
	// cout << "Now write to: " << std::string(file_name).append(".ply") << endl;
	// cout << "Save PCD cost time = " << tim.toc() << endl;
}

void Global_map::save_and_display_pointcloud(std::string dir_name, std::string file_name, int save_pts_with_views)
{
	save_to_pcd(dir_name, file_name, save_pts_with_views);
	scope_color(ANSI_COLOR_WHITE_BOLD);
	cout << "========================================================" << endl;
	cout << "Open pcl_viewer to display point cloud, close the viewer's window to continue mapping process ^_^" << endl;
	cout << "========================================================" << endl;
	system(std::string("pcl_viewer ").append(dir_name).append("/rgb_pt.pcd").c_str());
}

void Global_map::clear()
{
	m_rgb_pts_vec.clear();
}

void Global_map::set_minmum_dis(float minimum_dis)
{
	m_hashmap_3d_pts.clear();
	m_minimum_pts_size = minimum_dis;
}
