
#include "main.h"
#include "ACMMP.h"
#include "pointcloud_rgbd.hpp"


void GenerateSampleList(const std::string &dense_folder, std::vector<Problem> &problems)
{
    std::string cluster_list_path = dense_folder + std::string("/pair.txt");

    problems.clear();

    std::ifstream file(cluster_list_path);

    int num_images;
    file >> num_images;

    for (int i = 0; i < num_images; ++i) {
        Problem problem;
        problem.src_image_ids.clear();
        file >> problem.ref_image_id;

        int num_src_images;
        file >> num_src_images;
        for (int j = 0; j < num_src_images; ++j) {
            int id;
            float score;
            file >> id >> score;
            if (score <= 0.0f) {
                continue;
            }
            problem.src_image_ids.push_back(id);
        }
        problems.push_back(problem);
    }
}

int ComputeMultiScaleSettings(const std::string &dense_folder, std::vector<Problem> &problems)
{
    int max_num_downscale = -1;
    int size_bound = 1000;
    PatchMatchParams pmp;
    std::string image_folder = dense_folder + std::string("/images");
    size_t num_images = problems.size();
    for (size_t i = 0; i < num_images; ++i)
    {
        std::stringstream image_path;
        image_path << image_folder << "/" << std::setw(8) << std::setfill('0') << problems[i].ref_image_id << ".jpg";
        cv::Mat_<uint8_t> image_uint = cv::imread(image_path.str(), cv::IMREAD_GRAYSCALE);
        int rows = image_uint.rows;
        int cols = image_uint.cols;
        int max_size = std::max(rows, cols);
        if (max_size > pmp.max_image_size)
        {
            max_size = pmp.max_image_size;
        }
        problems[i].max_image_size = max_size;
        int k = 0;
        while (max_size > size_bound)
        {
            max_size /= 2;
            k++;
        }
        if (k > max_num_downscale)
        {
            max_num_downscale = k;
        }
        problems[i].num_downscale = k;
    }
    return max_num_downscale;
}

void ProcessProblem(const std::string &dense_folder, const std::vector<Problem> &problems, const int idx, bool geom_consistency, bool planar_prior, bool hierarchy, bool multi_geometrty=false)
{
    const Problem problem = problems[idx];
    std::cout << "Processing image " << std::setw(8) << std::setfill('0') << problem.ref_image_id << "..." << std::endl;
    cudaSetDevice(0);
    std::stringstream result_path;
    result_path << dense_folder << "/ACMMP" << "/2333_" << std::setw(8) << std::setfill('0') << problem.ref_image_id;
    std::string result_folder = result_path.str();
    mkdir(result_folder.c_str(), 0777);

    ACMMP acmmp;
    if (geom_consistency) {
        acmmp.SetGeomConsistencyParams(multi_geometrty);
    }
    if (hierarchy) {
        acmmp.SetHierarchyParams();
    }

    std::cout<<"acmmp.InuputInitialization"<<std::endl;
    acmmp.InuputInitialization(dense_folder, problems, idx);
    std::cout<<"acmmp.CudaSpaceInitialization"<<std::endl;
    acmmp.CudaSpaceInitialization(dense_folder, problem);
    std::cout<<"acmmp.RunPatchMatch"<<std::endl;
    acmmp.RunPatchMatch();

    const int width = acmmp.GetReferenceImageWidth();
    const int height = acmmp.GetReferenceImageHeight();

    cv::Mat_<float> depths = cv::Mat::zeros(height, width, CV_32FC1);
    cv::Mat_<cv::Vec3f> normals = cv::Mat::zeros(height, width, CV_32FC3);
    cv::Mat_<float> costs = cv::Mat::zeros(height, width, CV_32FC1);

    for (int col = 0; col < width; ++col) {
        for (int row = 0; row < height; ++row) {
            int center = row * width + col;
            float4 plane_hypothesis = acmmp.GetPlaneHypothesis(center);
            depths(row, col) = plane_hypothesis.w;
            normals(row, col) = cv::Vec3f(plane_hypothesis.x, plane_hypothesis.y, plane_hypothesis.z);
            costs(row, col) = acmmp.GetCost(center);
        }
    }

    if (planar_prior) {
        std::cout << "Run Planar Prior Assisted PatchMatch MVS ..." << std::endl;
        acmmp.SetPlanarPriorParams();

        const cv::Rect imageRC(0, 0, width, height);
        std::vector<cv::Point> support2DPoints;

        acmmp.GetSupportPoints(support2DPoints);
        const auto triangles = acmmp.DelaunayTriangulation(imageRC, support2DPoints);
        cv::Mat refImage = acmmp.GetReferenceImage().clone();
        std::vector<cv::Mat> mbgr(3);
        mbgr[0] = refImage.clone();
        mbgr[1] = refImage.clone();
        mbgr[2] = refImage.clone();
        cv::Mat srcImage;
        cv::merge(mbgr, srcImage);
        for (const auto triangle : triangles) {
            if (imageRC.contains(triangle.pt1) && imageRC.contains(triangle.pt2) && imageRC.contains(triangle.pt3)) {
                cv::line(srcImage, triangle.pt1, triangle.pt2, cv::Scalar(0, 0, 255));
                cv::line(srcImage, triangle.pt1, triangle.pt3, cv::Scalar(0, 0, 255));
                cv::line(srcImage, triangle.pt2, triangle.pt3, cv::Scalar(0, 0, 255));
            }
        }
        std::string triangulation_path = result_folder + "/triangulation.png";
        cv::imwrite(triangulation_path, srcImage);

        cv::Mat_<float> mask_tri = cv::Mat::zeros(height, width, CV_32FC1);
        std::vector<float4> planeParams_tri;
        planeParams_tri.clear();

        uint32_t idx = 0;
        for (const auto triangle : triangles) {
            if (imageRC.contains(triangle.pt1) && imageRC.contains(triangle.pt2) && imageRC.contains(triangle.pt3)) {
                float L01 = sqrt(pow(triangle.pt1.x - triangle.pt2.x, 2) + pow(triangle.pt1.y - triangle.pt2.y, 2));
                float L02 = sqrt(pow(triangle.pt1.x - triangle.pt3.x, 2) + pow(triangle.pt1.y - triangle.pt3.y, 2));
                float L12 = sqrt(pow(triangle.pt2.x - triangle.pt3.x, 2) + pow(triangle.pt2.y - triangle.pt3.y, 2));

                float max_edge_length = std::max(L01, std::max(L02, L12));
                float step = 1.0 / max_edge_length;

                for (float p = 0; p < 1.0; p += step) {
                    for (float q = 0; q < 1.0 - p; q += step) {
                        int x = p * triangle.pt1.x + q * triangle.pt2.x + (1.0 - p - q) * triangle.pt3.x;
                        int y = p * triangle.pt1.y + q * triangle.pt2.y + (1.0 - p - q) * triangle.pt3.y;
                        mask_tri(y, x) = idx + 1.0; // To distinguish from the label of non-triangulated areas
                    }
                }

                // estimate plane parameter
                float4 n4 = acmmp.GetPriorPlaneParams(triangle, depths);
                planeParams_tri.push_back(n4);
                idx++;
            }
        }

        cv::Mat_<float> priordepths = cv::Mat::zeros(height, width, CV_32FC1);
        for (int i = 0; i < width; ++i) {
            for (int j = 0; j < height; ++j) {
                if (mask_tri(j, i) > 0) {
                    float d = acmmp.GetDepthFromPlaneParam(planeParams_tri[mask_tri(j, i) - 1], i, j);
                    if (d <= acmmp.GetMaxDepth() && d >= acmmp.GetMinDepth()) {
                        priordepths(j, i) = d;
                    }
                    else {
                        mask_tri(j, i) = 0;
                    }
                }
            }
        }
        // std::string depth_path = result_folder + "/depths_prior.dmb";
        //  writeDepthDmb(depth_path, priordepths);

        acmmp.CudaPlanarPriorInitialization(planeParams_tri, mask_tri);
        acmmp.RunPatchMatch();

        for (int col = 0; col < width; ++col) {
            for (int row = 0; row < height; ++row) {
                int center = row * width + col;
                float4 plane_hypothesis = acmmp.GetPlaneHypothesis(center);
                depths(row, col) = plane_hypothesis.w;
                normals(row, col) = cv::Vec3f(plane_hypothesis.x, plane_hypothesis.y, plane_hypothesis.z);
                costs(row, col) = acmmp.GetCost(center);
            }
        }
    }

    std::string suffix = "/depths.dmb";
    if (geom_consistency) {
        suffix = "/depths_geom.dmb";
    }
    std::string depth_path = result_folder + suffix;
    std::string normal_path = result_folder + "/normals.dmb";
    std::string cost_path = result_folder + "/costs.dmb";
    writeDepthDmb(depth_path, depths);
    writeNormalDmb(normal_path, normals);
    writeDepthDmb(cost_path, costs);
    std::cout << "Processing image " << std::setw(8) << std::setfill('0') << problem.ref_image_id << " done!" << std::endl;
}

void JointBilateralUpsampling(const std::string &dense_folder, const Problem &problem, int acmmp_size)
{
    std::stringstream result_path;
    result_path << dense_folder << "/ACMMP" << "/2333_" << std::setw(8) << std::setfill('0') << problem.ref_image_id;
    std::string result_folder = result_path.str();
    std::string depth_path = result_folder + "/depths_geom.dmb";
    cv::Mat_<float> ref_depth;
    readDepthDmb(depth_path, ref_depth);

    std::string image_folder = dense_folder + std::string("/images");
    std::stringstream image_path;
    image_path << image_folder << "/" << std::setw(8) << std::setfill('0') << problem.ref_image_id << ".jpg";
    cv::Mat_<uint8_t> image_uint = cv::imread(image_path.str(), cv::IMREAD_GRAYSCALE);
    cv::Mat image_float;
    image_uint.convertTo(image_float, CV_32FC1);
    const float factor_x = static_cast<float>(acmmp_size) / image_float.cols;
    const float factor_y = static_cast<float>(acmmp_size) / image_float.rows;
    const float factor = std::min(factor_x, factor_y);

    const int new_cols = std::round(image_float.cols * factor);
    const int new_rows = std::round(image_float.rows * factor);
    cv::Mat scaled_image_float;
    cv::resize(image_float, scaled_image_float, cv::Size(new_cols,new_rows), 0, 0, cv::INTER_LINEAR);

    std::cout << "Run JBU for image " << problem.ref_image_id <<  ".jpg" << std::endl;
    RunJBU(scaled_image_float, ref_depth, dense_folder, problem );
}

void RunFusion(std::string &dense_folder, const std::vector<Problem> &problems, bool geom_consistency)
{
    size_t num_images = problems.size();
    std::string image_folder = dense_folder + std::string("/images");
    std::string cam_folder = dense_folder + std::string("/cams");

    std::vector<cv::Mat> images;
    std::vector<Camera> cameras;
    std::vector<cv::Mat_<float>> depths;
    std::vector<cv::Mat_<cv::Vec3f>> normals;
    std::vector<cv::Mat> masks;
    images.clear();
    cameras.clear();
    depths.clear();
    normals.clear();
    masks.clear();

    for (size_t i = 0; i < num_images; ++i) {
        std::cout << "Reading image " << std::setw(8) << std::setfill('0') << i << "..." << std::endl;
        std::stringstream image_path;
        image_path << image_folder << "/" << std::setw(8) << std::setfill('0') << problems[i].ref_image_id << ".jpg";
        cv::Mat_<cv::Vec3b> image = cv::imread (image_path.str(), cv::IMREAD_COLOR);
        std::stringstream cam_path;
        cam_path << cam_folder << "/" << std::setw(8) << std::setfill('0') << problems[i].ref_image_id << "_cam.txt";
        Camera camera = ReadCamera(cam_path.str());

        std::stringstream result_path;
        result_path << dense_folder << "/ACMMP" << "/2333_" << std::setw(8) << std::setfill('0') << problems[i].ref_image_id;
        std::string result_folder = result_path.str();
        std::string suffix = "/depths.dmb";
        if (geom_consistency) {
            suffix = "/depths_geom.dmb";
        }
        std::string depth_path = result_folder + suffix;
        std::string normal_path = result_folder + "/normals.dmb";
        cv::Mat_<float> depth;
        cv::Mat_<cv::Vec3f> normal;
        readDepthDmb(depth_path, depth);
        readNormalDmb(normal_path, normal);

        cv::Mat_<cv::Vec3b> scaled_image;
        RescaleImageAndCamera(image, scaled_image, depth, camera);
        images.push_back(scaled_image);
        cameras.push_back(camera);
        depths.push_back(depth);
        normals.push_back(normal);
        cv::Mat mask = cv::Mat::zeros(depth.rows, depth.cols, CV_8UC1);
        masks.push_back(mask);
    }
    std::cout<< "images.size(): "<<images.size()<<std::endl;
    bool single_proj = false;
    if(images.size()<2)
    {
        std::cout<<"Only one image, no fuse"<<std::endl;
        single_proj = true;
        // return;
    }

    std::vector<PointList> PointCloud;
    PointCloud.clear();

    for (size_t i = 0; i < num_images; ++i) {
        std::cout << "Fusing image " << std::setw(8) << std::setfill('0') << i << "..." << std::endl;
        const int cols = depths[i].cols;
        const int rows = depths[i].rows;
        int num_ngb = problems[i].src_image_ids.size();
        std::vector<int2> used_list(num_ngb, make_int2(-1, -1));
        for (int r =0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                if (masks[i].at<uchar>(r, c) == 1)
                    continue;
                float ref_depth = depths[i].at<float>(r, c);
                cv::Vec3f ref_normal = normals[i].at<cv::Vec3f>(r, c);

                if (ref_depth <= 0.0)
                    continue;

                float3 PointX = Get3DPointonWorld(c, r, ref_depth, cameras[i]);
                float3 consistent_Point = PointX;
                cv::Vec3f consistent_normal = ref_normal;
                float consistent_Color[3] = {(float)images[i].at<cv::Vec3b>(r, c)[0], (float)images[i].at<cv::Vec3b>(r, c)[1], (float)images[i].at<cv::Vec3b>(r, c)[2]};
                if(single_proj)
                {
                    PointList point3D;
                    point3D.coord = consistent_Point;
                    point3D.normal = make_float3(consistent_normal[0], consistent_normal[1], consistent_normal[2]);
                    point3D.color = make_float3(consistent_Color[0], consistent_Color[1], consistent_Color[2]);
                    PointCloud.push_back(point3D);
                    continue;
                }
                int num_consistent = 0;
                float dynamic_consistency = 0;

                for (int j = 0; j < num_ngb; ++j) {
                    int src_id = problems[i].src_image_ids[j];
                    const int src_cols = depths[src_id].cols;
                    const int src_rows = depths[src_id].rows;
                    float2 point;
                    float proj_depth;
                    ProjectonCamera(PointX, cameras[src_id], point, proj_depth);
                    int src_r = int(point.y + 0.5f);
                    int src_c = int(point.x + 0.5f);
                    if (src_c >= 0 && src_c < src_cols && src_r >= 0 && src_r < src_rows) {
                        if (masks[src_id].at<uchar>(src_r, src_c) == 1)
                            continue;

                        float src_depth = depths[src_id].at<float>(src_r, src_c);
                        cv::Vec3f src_normal = normals[src_id].at<cv::Vec3f>(src_r, src_c);
                        if (src_depth <= 0.0)
                            continue;

                        float3 tmp_X = Get3DPointonWorld(src_c, src_r, src_depth, cameras[src_id]);
                        float2 tmp_pt;
                        ProjectonCamera(tmp_X, cameras[i], tmp_pt, proj_depth);
                        float reproj_error = sqrt(pow(c - tmp_pt.x, 2) + pow(r - tmp_pt.y, 2));
                        float relative_depth_diff = fabs(proj_depth - ref_depth) / ref_depth;
                        float angle = GetAngle(ref_normal, src_normal);

                        if (reproj_error < 2.0f && relative_depth_diff < 0.01f && angle < 0.174533f) {
                           /* consistent_Point.x += tmp_X.x;
                            consistent_Point.y += tmp_X.y;
                            consistent_Point.z += tmp_X.z;
                            consistent_normal = consistent_normal + src_normal;
                            consistent_Color[0] += images[src_id].at<cv::Vec3b>(src_r, src_c)[0];
                            consistent_Color[1] += images[src_id].at<cv::Vec3b>(src_r, src_c)[1];
                            consistent_Color[2] += images[src_id].at<cv::Vec3b>(src_r, src_c)[2];*/
                           

                            used_list[j].x = src_c;
                            used_list[j].y = src_r;

                            float tmp_index = reproj_error + 200 * relative_depth_diff + angle * 10;
                            float cons = exp(-tmp_index);
                            dynamic_consistency += exp(-tmp_index);
                            num_consistent++;
                        }
                    }
                }

                if (num_consistent >= 1 && (dynamic_consistency > 0.3 * num_consistent)) {
                    /*consistent_Point.x /= (num_consistent + 1.0f);
                    consistent_Point.y /= (num_consistent + 1.0f);
                    consistent_Point.z /= (num_consistent + 1.0f);
                    consistent_normal /= (num_consistent + 1.0f);
                    consistent_Color[2] /= (num_consistent + 1.0f);*/

                    PointList point3D;
                    point3D.coord = consistent_Point;
                    point3D.normal = make_float3(consistent_normal[0], consistent_normal[1], consistent_normal[2]);
                    point3D.color = make_float3(consistent_Color[0], consistent_Color[1], consistent_Color[2]);
                    PointCloud.push_back(point3D);

                    for (int j = 0; j < num_ngb; ++j) {
                        if (used_list[j].x == -1)
                            continue;
                        masks[problems[i].src_image_ids[j]].at<uchar>(used_list[j].y, used_list[j].x) = 1;
                    }
                }
            }
        }
    }

    std::string ply_path = dense_folder + "/ACMMP/ACMMP_model.ply";
    StoreColorPlyFileBinaryPointCloud (ply_path, PointCloud);
    std::cout<<"Saving to "<<ply_path<<std::endl;
}

//* 返回值不能是int
//* error: static assertion failed: You tried to register a kernel with an unsupported integral output type. Please use int64_t instead.
void maintest()
{
    std::string dense_folder = "/home/zhujun/catkin_ws/src/r3live-master/r3live_output/data_for_mesh_front/acmmp";
    std::vector<Problem> problems;
    //* 获取参考图和源图的id
    GenerateSampleList(dense_folder, problems);
    std::string output_folder = dense_folder + std::string("/ACMMP");
    mkdir(output_folder.c_str(), 0777);
    size_t num_images = problems.size();
    std::cout << "There are " << num_images << " problems needed to be processed!" << std::endl;
    //* 最大尺寸大于1000就会被下采样，每次缩小为原来1/2
    int max_num_downscale = ComputeMultiScaleSettings(dense_folder, problems);
    int flag = 0;
    int geom_iterations = 2;
    bool geom_consistency = false;
    bool planar_prior = false;
    bool hierarchy = false;
    bool multi_geometry = false;
    while (max_num_downscale >= 0)
    {
        std::cout << "Scale: " << max_num_downscale << std::endl;
        for (size_t i = 0; i < num_images; ++i)
        {
            if (problems[i].num_downscale >= 0)
            {
                problems[i].cur_image_size = problems[i].max_image_size / pow(2, problems[i].num_downscale);
                problems[i].num_downscale--;
            }
        }
        std::cout<<"problems[0].cur_image_size: "<<problems[0].cur_image_size<<std::endl;
        if (flag == 0)
        {
            flag = 1;
            geom_consistency = false;
            planar_prior = true;
            for (size_t i = 0; i < num_images; ++i)
            {
                ProcessProblem(dense_folder, problems, i, geom_consistency, planar_prior, hierarchy);
            }
            geom_consistency = true;
            planar_prior = false;
            for (int geom_iter = 0; geom_iter < geom_iterations; ++geom_iter)
            {
                if (geom_iter == 0) 
                    multi_geometry = false;
                else 
                    multi_geometry = true;
                for (size_t i = 0; i < num_images; ++i)
                {
                    ProcessProblem(dense_folder,  problems, i, geom_consistency, planar_prior, hierarchy, multi_geometry);
                }
            }
        }
        else 
        {
            for (size_t i = 0; i < num_images; ++i)
            {
                JointBilateralUpsampling(dense_folder, problems[i], problems[i].cur_image_size);
            }
            hierarchy = true;
            geom_consistency = false;
            planar_prior = true;
            for (size_t i = 0; i < num_images; ++i)
            {
                ProcessProblem(dense_folder,  problems, i, geom_consistency, planar_prior, hierarchy);
            }
            hierarchy = false;
            geom_consistency = true;
            planar_prior = false;
            for (int geom_iter = 0; geom_iter < geom_iterations; ++geom_iter)
            {
                if (geom_iter == 0)
                    multi_geometry = false;
                else
                    multi_geometry = true;
                for (size_t i = 0; i < num_images; ++i)
                {
                    ProcessProblem(dense_folder,  problems, i, geom_consistency, planar_prior, hierarchy, multi_geometry);
                }
            }
        }
        max_num_downscale--;
    }
    geom_consistency = true;
    RunFusion(dense_folder, problems, geom_consistency);

}

void acmmp_init(const torch::Tensor &image_torch, const torch::Tensor & K_torch, const torch::Tensor & ext_torch)
{
    //* GPU to cv::Mat 先换到cpu，再contiguous ！！！ 否则指针读取顺序会出问题
    //* 输入tensor为多张灰度图 [N, c, h, w] float 灰度图c为1
    //* K_torch 内参 [3，3]
    //* ext_torch 外参 [N, 4, 4]
    auto sizes = image_torch.sizes();
    auto device = image_torch.device();
    auto ndim = sizes.size();
    auto dtype = image_torch.dtype();
    auto ext_sizes = ext_torch.sizes();
    auto K_sizes = K_torch.sizes();
    std::cout<<"input image_torch sizes: "<<sizes<<std::endl; //* [5, 1002, 1253, 3]
    std::cout<<"ndim: "<<sizes.size()<<std::endl; //* 4
    std::cout<<"device: "<<device<<std::endl; //* cuda:0
    std::cout<<"dtype: "<<dtype<<std::endl; //* float
    if(ext_sizes.size()!=3 || ext_sizes[0]!=sizes[0] || K_sizes[0]!=3 || K_sizes[1]!=3 || K_sizes.size()!=2)
    {
        std::cout<<"Extrinsic size error or K error!"<<std::endl;
        std::cout<<"input ext sizes: "<<ext_sizes<<std::endl; //* [5, 1002, 1253, 3]
        return;
    }
    std::vector<cv::Mat> images;
    std::vector<cv::Mat> exts;
    std::cout<<"K_torch:\n"<<K_torch<<std::endl;
    cv::Mat K =  cv::Mat{3, 3, CV_32FC1, K_torch.to(torch::kCPU).contiguous().data_ptr<float>()};
    std::cout<<"K:\n"<<K<<std::endl;
    int N=-1,H=-1,W=-1,C=-1;
    if(ndim == 4)
    {
        N = sizes[0]; C = sizes[1]; H = sizes[2]; W = sizes[3];
    }
    for(int i=0;i<N;i++)
    {
        //* 将tensor转为cv::Mat。 先转换到cpu，不然会出错，并要连续化
        torch::Tensor img_i = image_torch.index_select(0,torch::tensor({i}).to(image_torch.device())).to(torch::kCPU).permute({0,2,3,1}).contiguous();
        cv::Mat img =  cv::Mat{H, W, CV_32FC(C), img_i.data_ptr<float>()};
        if(i==0)
        {
            auto size_i = img_i.sizes();
            std::cout<<"img_i sizes: "<<size_i<<std::endl;
            // std::cout<<"img_i : "<<img_i.squeeze()<<std::endl;
            std::cout << "img.rows: "<< img.rows << std::endl
                    << "img.cols: "<< img.cols << std::endl
                    << "img.channels(): "<< img.channels() << std::endl;
        }
        images.push_back(img);
        torch::Tensor ext_cpu = ext_torch.index_select(0,torch::tensor({i}).to(ext_torch.device())).to(torch::kCPU).contiguous();
        cv::Mat ext = cv::Mat{4, 4, CV_32FC(1), ext_cpu.data_ptr<float>()};
        exts.push_back(ext);
        std::cout<<ext<<std::endl;
    }
    std::cout<<"images.size(): "<<images.size()<<std::endl;
}


void tensorToMat(const torch::Tensor &image_torch)
{
    //* 输入tensor为多张灰度图 [N, c, h, w] float 灰度图c为1
    //* K 内参 [3，3]
    //* ext 外参 [N, 4, 4]
    std::vector<cv::Mat> images;
    auto sizes = image_torch.sizes();
    auto device = image_torch.device();
    auto ndim = sizes.size();
    auto dtype = image_torch.dtype();
    std::cout<<"input image_torch sizes: "<<sizes<<std::endl; //* [5, 1002, 1253, 3]
    std::cout<<"ndim: "<<sizes.size()<<std::endl; //* 4
    std::cout<<"device: "<<device<<std::endl; //* cuda:0
    std::cout<<"dtype: "<<dtype<<std::endl; //* float
    int N=-1,H=-1,W=-1,C=-1;
    if(ndim == 4)
    {
        N = sizes[0]; C = sizes[1]; H = sizes[2]; W = sizes[3];
    }
    
    for(int i=0;i<N;i++)
    {
        //* 先转换到cpu，不然会出错，并要连续化
        torch::Tensor img_i = image_torch.index_select(0,torch::tensor({i}).to(image_torch.device())).to(torch::kCPU).permute({0,2,3,1}).contiguous();
        auto size_i = img_i.sizes();
        std::cout<<"img_i sizes: "<<size_i<<std::endl;
        // std::cout<<"img_i : "<<img_i.squeeze()<<std::endl;
        cv::Mat img =  cv::Mat{H, W, CV_32FC(C), img_i.data_ptr<float>()};
        std::cout << "img.rows: "<< img.rows << std::endl
                << "img.cols: "<< img.cols << std::endl
                << "img.channels(): "<< img.channels() << std::endl;
        images.push_back(img);
        // std::cout<<img<<std::endl;
    }
    std::cout<<"images.size(): "<<images.size()<<std::endl;
    // std::cout<<"torch::tensor({0,3}): "<<torch::tensor({0,3})<<std::endl;
    // std::cout<<"torch::Tensor({0,3}): "<<torch::Tensor({0,3})<<std::endl;
    //*  no matching function for call to ‘at::Tensor::Tensor(<brace-enclosed initializer list>)
    //* torch.Tensor是主要的tensor类，所有的tensor都是torch.Tensor的实例。torch.Tensor是torch.FloatTensor的别名
}

torch::Tensor proj2depth(const torch::Tensor &image_torch, const torch::Tensor & K_torch)
{

}

TORCH_LIBRARY(acmmp, m) {
    //* 类并不需要暴露全部成员给python，部分也行！   
    m.class_<ACMMP>("ACMMP")
        .def(torch::init())
        .def("acmmp_init_test", &ACMMP::acmmp_init_test)
        .def("GetDepth", &ACMMP::GetDepth)
        .def("build_basic_tree", &ACMMP::build_basic_tree)
        .def("align_pts", &ACMMP::align_pts)
        .def("GetUsedMask", &ACMMP::GetUsedMask)
        .def("GetCosts", &ACMMP::GetCosts)
        .def("GetDepthEdge", &ACMMP::GetDepthEdge)
    ;
    m.class_<Global_map>("Global_map")
        .def(torch::init())
        .def("append_points_to_global_map", &Global_map::append_points_to_global_map)
        .def("get_pc", &Global_map::get_pc)
        .def("set_resolution", &Global_map::set_resolution)
        .def("cur_resolution", &Global_map::cur_resolution)
        .def("str_test", &Global_map::str_test)
        .def("set_K_wh", &Global_map::set_K_wh)
        .def("add_img", &Global_map::add_img)
        .def("get_depth", &Global_map::get_depth)
        .def("get_image", &Global_map::get_image)
        .def("get_ext", &Global_map::get_ext)
        .def("get_K", &Global_map::get_K)
        .def("bilinear_interplote_depth", &Global_map::bilinear_interplote_depth)
        .def("bilinear_interplote_depth_id", &Global_map::bilinear_interplote_depth_id)
        .def("ransac_fit_ground_plane", &Global_map::ransac_fit_ground_plane)
        .def("get_depth_with_attr", &Global_map::get_depth_with_attr)
        .def("point_cloud_segmentation", &Global_map::point_cloud_segmentation)
        .def("get_ground_plane_param", &Global_map::get_ground_plane_param)
        .def("return_data", &Global_map::return_data)
        .def("enrich_ground", &Global_map::enrich_ground)
    ;
    m.class_<Params>("Params")
        // The following line registers the contructor of our MyStackClass
        // class that takes a single `std::vector<std::string>` argument,
        // i.e. it exposes the C++ method `MyStackClass(std::vector<T> init)`.
        // Currently, we do not support registering overloaded
        // constructors, so for now you can only `def()` one instance of
        // `torch::init`.
        .def(torch::init())
        // The next line registers a stateless (i.e. no captures) C++ lambda
        // function as a method. Note that a lambda function must take a
        // `c10::intrusive_ptr<YourClass>` (or some const/ref version of that)
        // as the first argument. Other arguments can be whatever you want.
        // .def("top", [](const c10::intrusive_ptr<Params>& self) {return self->stack_.back();})
        // The following four lines expose methods of the MyStackClass<std::string>
        // class as-is. `torch::class_` will automatically examine the
        // argument and return types of the passed-in method pointers and
        // expose these to Python and TorchScript accordingly. Finally, notice
        // that we must take the *address* of the fully-qualified method name,
        // i.e. use the unary `&` operator, due to C++ typing rules.
        .def("test", &Params::test)
    ;
    m.def("maintest", maintest);
    m.def("tensorToMat", tensorToMat);
    m.def("acmmp_init", acmmp_init);
    m.def("proj2depth", proj2depth);

}