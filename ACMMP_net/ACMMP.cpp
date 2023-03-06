#include "ACMMP.h"

#include <cstdarg>

void struct_test(struct Params params)
{
    int b = params.b;
    int h = params.h;
    int w = params.w;
    int p = params.p;
    int npts = params.npts;
    int ndepth = params.ndepth;
    int nsrc = params.nsrc;
    int height = params.height;
    int width = params.width;
    printf("Struct test!!!\n");
    printf("b: %d width: %d\n", b, width);
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

Camera ReadCamera(const std::string &cam_path)
{
    Camera camera;
    std::ifstream file(cam_path);

    std::string line;
    file >> line;

    for (int i = 0; i < 3; ++i)
    {
        file >> camera.R[3 * i + 0] >> camera.R[3 * i + 1] >> camera.R[3 * i + 2] >> camera.t[i];
    }

    float tmp[4];
    file >> tmp[0] >> tmp[1] >> tmp[2] >> tmp[3];
    file >> line;

    for (int i = 0; i < 3; ++i)
    {
        file >> camera.K[3 * i + 0] >> camera.K[3 * i + 1] >> camera.K[3 * i + 2];
    }

    float depth_num;
    float interval;
    file >> camera.depth_min >> interval >> depth_num >> camera.depth_max;
    if (camera.depth_max < interval)
    {
        camera.depth_max = interval;
    }

    return camera;
}

void RescaleImageAndCamera(cv::Mat_<cv::Vec3b> &src, cv::Mat_<cv::Vec3b> &dst, cv::Mat_<float> &depth, Camera &camera)
{
    const int cols = depth.cols;
    const int rows = depth.rows;

    if (cols == src.cols && rows == src.rows)
    {
        dst = src.clone();
        return;
    }

    const float scale_x = cols / static_cast<float>(src.cols);
    const float scale_y = rows / static_cast<float>(src.rows);

    cv::resize(src, dst, cv::Size(cols, rows), 0, 0, cv::INTER_LINEAR);

    camera.K[0] *= scale_x;
    camera.K[2] *= scale_x;
    camera.K[4] *= scale_y;
    camera.K[5] *= scale_y;
    camera.width = cols;
    camera.height = rows;
}

float3 Get3DPointonWorld(const int x, const int y, const float depth, const Camera camera)
{
    float3 pointX;
    float3 tmpX;
    // Reprojection
    pointX.x = depth * (x - camera.K[2]) / camera.K[0];
    pointX.y = depth * (y - camera.K[5]) / camera.K[4];
    pointX.z = depth;

    // Rotation
    tmpX.x = camera.R[0] * pointX.x + camera.R[3] * pointX.y + camera.R[6] * pointX.z;
    tmpX.y = camera.R[1] * pointX.x + camera.R[4] * pointX.y + camera.R[7] * pointX.z;
    tmpX.z = camera.R[2] * pointX.x + camera.R[5] * pointX.y + camera.R[8] * pointX.z;

    // Transformation
    float3 C;
    C.x = -(camera.R[0] * camera.t[0] + camera.R[3] * camera.t[1] + camera.R[6] * camera.t[2]);
    C.y = -(camera.R[1] * camera.t[0] + camera.R[4] * camera.t[1] + camera.R[7] * camera.t[2]);
    C.z = -(camera.R[2] * camera.t[0] + camera.R[5] * camera.t[1] + camera.R[8] * camera.t[2]);
    pointX.x = tmpX.x + C.x;
    pointX.y = tmpX.y + C.y;
    pointX.z = tmpX.z + C.z;

    return pointX;
}

float3 Get3DPointonRefCam(const int x, const int y, const float depth, const Camera camera)
{
    float3 pointX;
    // Reprojection
    pointX.x = depth * (x - camera.K[2]) / camera.K[0];
    pointX.y = depth * (y - camera.K[5]) / camera.K[4];
    pointX.z = depth;

    return pointX;
}

void ProjectonCamera(const float3 PointX, const Camera camera, float2 &point, float &depth)
{
    float3 tmp;
    tmp.x = camera.R[0] * PointX.x + camera.R[1] * PointX.y + camera.R[2] * PointX.z + camera.t[0];
    tmp.y = camera.R[3] * PointX.x + camera.R[4] * PointX.y + camera.R[5] * PointX.z + camera.t[1];
    tmp.z = camera.R[6] * PointX.x + camera.R[7] * PointX.y + camera.R[8] * PointX.z + camera.t[2];

    depth = camera.K[6] * tmp.x + camera.K[7] * tmp.y + camera.K[8] * tmp.z;
    point.x = (camera.K[0] * tmp.x + camera.K[1] * tmp.y + camera.K[2] * tmp.z) / depth;
    point.y = (camera.K[3] * tmp.x + camera.K[4] * tmp.y + camera.K[5] * tmp.z) / depth;
}

float GetAngle(const cv::Vec3f &v1, const cv::Vec3f &v2)
{
    float dot_product = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
    float angle = acosf(dot_product);
    // if angle is not a number the dot product was 1 and thus the two vectors should be identical --> return 0
    if (angle != angle)
        return 0.0f;

    return angle;
}

int readDepthDmb(const std::string file_path, cv::Mat_<float> &depth)
{
    FILE *inimage;
    inimage = fopen(file_path.c_str(), "rb");
    if (!inimage)
    {
        std::cout << "Error opening file " << file_path << std::endl;
        return -1;
    }

    int32_t type, h, w, nb;

    type = -1;

    fread(&type, sizeof(int32_t), 1, inimage);
    fread(&h, sizeof(int32_t), 1, inimage);
    fread(&w, sizeof(int32_t), 1, inimage);
    fread(&nb, sizeof(int32_t), 1, inimage);

    if (type != 1)
    {
        fclose(inimage);
        return -1;
    }

    int32_t dataSize = h * w * nb;

    float *data;
    data = (float *)malloc(sizeof(float) * dataSize);
    fread(data, sizeof(float), dataSize, inimage);

    depth = cv::Mat(h, w, CV_32F, data);

    fclose(inimage);
    return 0;
}

int writeDepthDmb(const std::string file_path, const cv::Mat_<float> depth)
{
    FILE *outimage;
    outimage = fopen(file_path.c_str(), "wb");
    if (!outimage)
    {
        std::cout << "Error opening file " << file_path << std::endl;
    }

    int32_t type = 1;
    int32_t h = depth.rows;
    int32_t w = depth.cols;
    int32_t nb = 1;

    fwrite(&type, sizeof(int32_t), 1, outimage);
    fwrite(&h, sizeof(int32_t), 1, outimage);
    fwrite(&w, sizeof(int32_t), 1, outimage);
    fwrite(&nb, sizeof(int32_t), 1, outimage);

    float *data = (float *)depth.data;

    int32_t datasize = w * h * nb;
    fwrite(data, sizeof(float), datasize, outimage);

    fclose(outimage);
    return 0;
}

int readNormalDmb(const std::string file_path, cv::Mat_<cv::Vec3f> &normal)
{
    FILE *inimage;
    inimage = fopen(file_path.c_str(), "rb");
    if (!inimage)
    {
        std::cout << "Error opening file " << file_path << std::endl;
        return -1;
    }

    int32_t type, h, w, nb;

    type = -1;

    fread(&type, sizeof(int32_t), 1, inimage);
    fread(&h, sizeof(int32_t), 1, inimage);
    fread(&w, sizeof(int32_t), 1, inimage);
    fread(&nb, sizeof(int32_t), 1, inimage);

    if (type != 1)
    {
        fclose(inimage);
        return -1;
    }

    int32_t dataSize = h * w * nb;

    float *data;
    data = (float *)malloc(sizeof(float) * dataSize);
    fread(data, sizeof(float), dataSize, inimage);

    normal = cv::Mat(h, w, CV_32FC3, data);

    fclose(inimage);
    return 0;
}

int writeNormalDmb(const std::string file_path, const cv::Mat_<cv::Vec3f> normal)
{
    FILE *outimage;
    outimage = fopen(file_path.c_str(), "wb");
    if (!outimage)
    {
        std::cout << "Error opening file " << file_path << std::endl;
    }

    int32_t type = 1; // float
    int32_t h = normal.rows;
    int32_t w = normal.cols;
    int32_t nb = 3;

    fwrite(&type, sizeof(int32_t), 1, outimage);
    fwrite(&h, sizeof(int32_t), 1, outimage);
    fwrite(&w, sizeof(int32_t), 1, outimage);
    fwrite(&nb, sizeof(int32_t), 1, outimage);

    float *data = (float *)normal.data;

    int32_t datasize = w * h * nb;
    fwrite(data, sizeof(float), datasize, outimage);

    fclose(outimage);
    return 0;
}

void StoreColorPlyFileBinaryPointCloud(const std::string &plyFilePath, const std::vector<PointList> &pc)
{
    std::cout << "store 3D points to ply file" << std::endl;

    FILE *outputPly;
    outputPly = fopen(plyFilePath.c_str(), "wb");

    /*write header*/
    fprintf(outputPly, "ply\n");
    fprintf(outputPly, "format binary_little_endian 1.0\n");
    fprintf(outputPly, "element vertex %d\n", pc.size());
    fprintf(outputPly, "property float x\n");
    fprintf(outputPly, "property float y\n");
    fprintf(outputPly, "property float z\n");
    fprintf(outputPly, "property float nx\n");
    fprintf(outputPly, "property float ny\n");
    fprintf(outputPly, "property float nz\n");
    fprintf(outputPly, "property uchar red\n");
    fprintf(outputPly, "property uchar green\n");
    fprintf(outputPly, "property uchar blue\n");
    fprintf(outputPly, "end_header\n");

    // write data
#pragma omp parallel for
    for (size_t i = 0; i < pc.size(); i++)
    {
        const PointList &p = pc[i];
        float3 X = p.coord;
        const float3 normal = p.normal;
        const float3 color = p.color;
        const char b_color = (int)color.x;
        const char g_color = (int)color.y;
        const char r_color = (int)color.z;

        if (!(X.x < FLT_MAX && X.x > -FLT_MAX) || !(X.y < FLT_MAX && X.y > -FLT_MAX) || !(X.z < FLT_MAX && X.z >= -FLT_MAX))
        {
            X.x = 0.0f;
            X.y = 0.0f;
            X.z = 0.0f;
        }
#pragma omp critical
        {
            fwrite(&X.x, sizeof(X.x), 1, outputPly);
            fwrite(&X.y, sizeof(X.y), 1, outputPly);
            fwrite(&X.z, sizeof(X.z), 1, outputPly);
            fwrite(&normal.x, sizeof(normal.x), 1, outputPly);
            fwrite(&normal.y, sizeof(normal.y), 1, outputPly);
            fwrite(&normal.z, sizeof(normal.z), 1, outputPly);
            fwrite(&r_color, sizeof(char), 1, outputPly);
            fwrite(&g_color, sizeof(char), 1, outputPly);
            fwrite(&b_color, sizeof(char), 1, outputPly);
        }
    }
    fclose(outputPly);
}

static float GetDisparity(const Camera &camera, const int2 &p, const float &depth)
{
    float point3D[3];
    point3D[0] = depth * (p.x - camera.K[2]) / camera.K[0];
    point3D[1] = depth * (p.y - camera.K[5]) / camera.K[4];
    point3D[3] = depth;

    return std::sqrt(point3D[0] * point3D[0] + point3D[1] * point3D[1] + point3D[2] * point3D[2]);
}

inline bool file_or_path_exist (const std::string& name) {
    struct stat buffer;
    bool result = stat(name.c_str(), &buffer) == 0;
    if(!result)
    {
        std::cout<<name<<" not exist!!"<<std::endl;
    }
    return result; 
}

ACMMP::ACMMP() 
{
    std::cout<<" ACMMP created!"<<std::endl;
    // getchar();
}

ACMMP::~ACMMP()
{
    std::cout<<"delete ACMMP!"<<std::endl;
    delete[] plane_hypotheses_host;
    delete[] costs_host;

    for (int i = 0; i < num_images; ++i)
    {
        cudaDestroyTextureObject(texture_objects_host.images[i]);
        cudaFreeArray(cuArray[i]);
    }
    cudaFree(texture_objects_cuda);
    cudaFree(cameras_cuda);
    cudaFree(plane_hypotheses_cuda);
    cudaFree(costs_cuda);
    cudaFree(pre_costs_cuda);
    cudaFree(rand_states_cuda);
    cudaFree(selected_views_cuda);
    cudaFree(depths_cuda);
    //* TODO
    cudaFree(depth_edge_mask_cuda);
    delete[] depth_edge_mask_host;

    if (params.geom_consistency)
    {
        for (int i = 0; i < num_images; ++i)
        {
            cudaDestroyTextureObject(texture_depths_host.images[i]);
            cudaFreeArray(cuDepthArray[i]);
        }
        cudaFree(texture_depths_cuda);
    }

    if (params.hierarchy)
    {
        delete[] scaled_plane_hypotheses_host;
        delete[] pre_costs_host;

        cudaFree(scaled_plane_hypotheses_cuda);
        cudaFree(pre_costs_cuda);
    }

    if (params.planar_prior)
    {
        delete[] prior_planes_host;
        delete[] plane_masks_host;

        cudaFree(prior_planes_cuda);
        cudaFree(plane_masks_cuda);
    }
}

void ACMMP::CudaSpaceInitialization(const std::string &dense_folder, const Problem &problem)
{
    num_images = (int)images.size();

    for (int i = 0; i < num_images; ++i)
    {
        //* 纹理对象创建之前首先要分别对纹理资源和纹理对象属性进行确定，分别对应cudaResoruceDesc和cudaTextureDesc；然后即可利用cudaCreateTextureObject来创建纹理对象
        int rows = images[i].rows;
        int cols = images[i].cols;

        //* 通道数据格式描述，最多四个通道，每个通道的数据格式
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
        //* 返回一个通道描述符，其中，x,y,z,和w 是返回值每个部分的位数，f 是格式
        //* The x,y,z,w are the number of bits in the x,y,z dimensions and 'w'. In your example the 'x' data is 32bits and the other dimensions aren't used.
        //* (The 'w' is used to make the math easier for applying transformations to 3d data)
        cudaMallocArray(&cuArray[i], &channelDesc, cols, rows);
        //* cudaArray *cuArray[MAX_IMAGES];
        //* 根据cudaChannelFormatDesc 结构desc 分配一个CUDA 数组
        cudaMemcpy2DToArray(cuArray[i], 0, 0, images[i].ptr<float>(), images[i].step[0], cols * sizeof(float), rows, cudaMemcpyHostToDevice);
        //* dst wOffset hOffset src_addr spitch width height kind 	 spitch：src指向的数组在内存中以字节形式的宽度，width<=spitch
        //* step[0] 一行元素的字节数： width*channels*bits/8 //* https://www.cnblogs.com/narjaja/p/10300878.html

        // * 资源描述
        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(cudaResourceDesc)); //* 将地址&resDesc的前sizeof(cudaResourceDesc)个字节全部置为0
        resDesc.resType = cudaResourceTypeArray;
        // resType 指定对应设备内存的形式，主要包括
        // cudaResourceTypeArray(二维纹理内存和二维纹理对象）
        // cudaResourceTypeMipmappedArray（不常用）
        // cudaResourceTypeLinear（一维纹理内存和一维纹理对象）
        // cudaResourceTypePitch2D（一维纹理内存和二维纹理对象）
        resDesc.res.array.array = cuArray[i]; //* 指定需要绑定的二维纹理内存
        // res是一个枚举变量，针对不同内存也有不同的形式
        // cudaResourceTypeArray 对应 res.array.array
        // cudaResourceTypeMipmappedArray 对应res.mipmap.mipmap
        // cudaResourceTypeLinear 对应 res.linear.devPtr（同时还需要设置res.linear.sizeInBytes和res.linear.desc）
        // cudaResourceTypePitch2D 对应 res.pitch2D.devPtr(同时需要设定res.pitch2D.pitchInBytes,res.pitch2D.width,res.pitch2D.height,res.pitch2D.ddesc)

        // * 纹理描述
        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(cudaTextureDesc));
        texDesc.addressMode[0] = cudaAddressModeWrap;
        texDesc.addressMode[1] = cudaAddressModeWrap;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;
        // * 创建纹理对象
        cudaCreateTextureObject(&(texture_objects_host.images[i]), &resDesc, &texDesc, NULL);
    }
    cudaMalloc((void **)&texture_objects_cuda, sizeof(cudaTextureObjects));
    cudaMemcpy(texture_objects_cuda, &texture_objects_host, sizeof(cudaTextureObjects), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&cameras_cuda, sizeof(Camera) * (num_images));
    cudaMemcpy(cameras_cuda, &cameras[0], sizeof(Camera) * (num_images), cudaMemcpyHostToDevice);

    plane_hypotheses_host = new float4[cameras[0].height * cameras[0].width];
    cudaMalloc((void **)&plane_hypotheses_cuda, sizeof(float4) * (cameras[0].height * cameras[0].width));

    costs_host = new float[cameras[0].height * cameras[0].width];
    cudaMalloc((void **)&costs_cuda, sizeof(float) * (cameras[0].height * cameras[0].width));
    cudaMalloc((void **)&pre_costs_cuda, sizeof(float) * (cameras[0].height * cameras[0].width));

    cudaMalloc((void **)&rand_states_cuda, sizeof(curandState) * (cameras[0].height * cameras[0].width));
    cudaMalloc((void **)&selected_views_cuda, sizeof(unsigned int) * (cameras[0].height * cameras[0].width));

    cudaMalloc((void **)&depths_cuda, sizeof(float) * (cameras[0].height * cameras[0].width));
    //* TODO 将初始深度复制到gpu
    cudaMemcpy(depths_cuda, init_depth.ptr<float>(), sizeof(float) * (cameras[0].height * cameras[0].width), cudaMemcpyHostToDevice);
    //* TODO 保存edge mask
    cudaMalloc((void **)&depth_edge_mask_cuda, sizeof(float) * (cameras[0].height * cameras[0].width));
    cudaMemset(depth_edge_mask_cuda, 0, sizeof(float) * (cameras[0].height * cameras[0].width));
    depth_edge_mask_host = new float[cameras[0].height * cameras[0].width];


    if (params.geom_consistency)
    {
        for (int i = 0; i < num_images; ++i)
        {
            int rows = depths[i].rows;
            int cols = depths[i].cols;

            cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
            cudaMallocArray(&cuDepthArray[i], &channelDesc, cols, rows);
            cudaMemcpy2DToArray(cuDepthArray[i], 0, 0, depths[i].ptr<float>(), depths[i].step[0], cols * sizeof(float), rows, cudaMemcpyHostToDevice);

            struct cudaResourceDesc resDesc;
            memset(&resDesc, 0, sizeof(cudaResourceDesc));
            resDesc.resType = cudaResourceTypeArray;
            resDesc.res.array.array = cuDepthArray[i];

            struct cudaTextureDesc texDesc;
            memset(&texDesc, 0, sizeof(cudaTextureDesc));
            texDesc.addressMode[0] = cudaAddressModeWrap;
            texDesc.addressMode[1] = cudaAddressModeWrap;
            texDesc.filterMode = cudaFilterModeLinear;
            texDesc.readMode = cudaReadModeElementType;
            texDesc.normalizedCoords = 0;

            cudaCreateTextureObject(&(texture_depths_host.images[i]), &resDesc, &texDesc, NULL);
        }
        cudaMalloc((void **)&texture_depths_cuda, sizeof(cudaTextureObjects));
        cudaMemcpy(texture_depths_cuda, &texture_depths_host, sizeof(cudaTextureObjects), cudaMemcpyHostToDevice);

        std::stringstream result_path;
        result_path << dense_folder << "/ACMMP"
                    << "/2333_" << std::setw(8) << std::setfill('0') << problem.ref_image_id;
        std::string result_folder = result_path.str();
        std::string suffix = "/depths.dmb";
        if (params.multi_geometry)
        {
            suffix = "/depths_geom.dmb";
        }
        std::string depth_path = result_folder + suffix;
        std::string normal_path = result_folder + "/normals.dmb";
        std::string cost_path = result_folder + "/costs.dmb";
        cv::Mat_<float> ref_depth;
        cv::Mat_<cv::Vec3f> ref_normal;
        cv::Mat_<float> ref_cost;
        readDepthDmb(depth_path, ref_depth);
        depths.push_back(ref_depth);
        readNormalDmb(normal_path, ref_normal);
        readDepthDmb(cost_path, ref_cost);
        int width = ref_depth.cols;
        int height = ref_depth.rows;
        for (int col = 0; col < width; ++col)
        {
            for (int row = 0; row < height; ++row)
            {
                int center = row * width + col;
                float4 plane_hypothesis;
                plane_hypothesis.x = ref_normal(row, col)[0];
                plane_hypothesis.y = ref_normal(row, col)[1];
                plane_hypothesis.z = ref_normal(row, col)[2];
                plane_hypothesis.w = ref_depth(row, col);
                plane_hypotheses_host[center] = plane_hypothesis;
                costs_host[center] = ref_cost(row, col);
            }
        }
        cudaMemcpy(plane_hypotheses_cuda, plane_hypotheses_host, sizeof(float4) * width * height, cudaMemcpyHostToDevice);
        cudaMemcpy(costs_cuda, costs_host, sizeof(float) * width * height, cudaMemcpyHostToDevice);
    }

    if (params.hierarchy)
    {
        std::stringstream result_path;
        result_path << dense_folder << "/ACMMP" << "/2333_" << std::setw(8) << std::setfill('0') << problem.ref_image_id;
        std::string result_folder = result_path.str();
        std::string depth_path = result_folder + "/depths.dmb";
        std::string normal_path = result_folder + "/normals.dmb";
        std::string cost_path = result_folder + "/costs.dmb";
        cv::Mat_<float> ref_depth;
        cv::Mat_<cv::Vec3f> ref_normal;
        cv::Mat_<float> ref_cost;
        readDepthDmb(depth_path, ref_depth);
        depths.push_back(ref_depth);
        readNormalDmb(normal_path, ref_normal);
        readDepthDmb(cost_path, ref_cost);
        int width = ref_normal.cols;
        int height = ref_normal.rows;
        scaled_plane_hypotheses_host = new float4[height * width];
        cudaMalloc((void **)&scaled_plane_hypotheses_cuda, sizeof(float4) * height * width);
        pre_costs_host = new float[height * width];
        cudaMalloc((void **)&pre_costs_cuda, sizeof(float) * cameras[0].height * cameras[0].width);
        if (width != images[0].rows || height != images[0].cols)
        {
            params.upsample = true;
            params.scaled_cols = width;
            params.scaled_rows = height;
        }
        else
        {
            params.upsample = false;
        }
        for (int col = 0; col < width; ++col)
        {
            for (int row = 0; row < height; ++row)
            {
                int center = row * width + col;
                float4 plane_hypothesis;
                plane_hypothesis.x = ref_normal(row, col)[0];
                plane_hypothesis.y = ref_normal(row, col)[1];
                plane_hypothesis.z = ref_normal(row, col)[2];
                if (params.upsample)
                {
                    plane_hypothesis.w = ref_cost(row, col);
                }
                else
                {
                    plane_hypothesis.w = ref_depth(row, col);
                }
                scaled_plane_hypotheses_host[center] = plane_hypothesis;
            }
        }

        for (int col = 0; col < cameras[0].width; ++col)
        {
            for (int row = 0; row < cameras[0].height; ++row)
            {
                int center = row * cameras[0].width + col;
                float4 plane_hypothesis;
                plane_hypothesis.w = ref_depth(row, col);
                plane_hypotheses_host[center] = plane_hypothesis;
            }
        }

        cudaMemcpy(scaled_plane_hypotheses_cuda, scaled_plane_hypotheses_host, sizeof(float4) * height * width, cudaMemcpyHostToDevice);
        cudaMemcpy(plane_hypotheses_cuda, plane_hypotheses_host, sizeof(float4) * cameras[0].width * cameras[0].height, cudaMemcpyHostToDevice);
    }
}

void ACMMP::kdtree_test()
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

void ACMMP::build_basic_tree(const torch::Tensor & basic_pts)
{
    //* basic_pts [N ,3+n]
    auto shape = basic_pts.sizes();
    if(shape.size()!=2 || shape[1]<3)
    {
        std::cout<<"Wrong shape of pts!!"<<std::endl;
        return;
    }
    int n_pts = shape[0];
    if(n_pts<NUM_MATCH_POINTS)
    {
        std::cout<<"Too few pts!!"<<std::endl;
        return;
    }
    // ref_pts.reset(new PointCloudXYZINormal());
    ref_pts = std::make_shared<PointCloudXYZINormal>();
    src_pts = std::make_shared<PointCloudXYZINormal>();
    src_pts_updated = std::make_shared<PointCloudXYZINormal>();
    laserCloudOri = std::make_shared<PointCloudXYZINormal>();
    coeffSel  = std::make_shared<PointCloudXYZINormal>();
    for(int i=0;i<n_pts;i++)
    {
        PointType point;
        point.x = basic_pts[i][0].item<float>();
        point.y = basic_pts[i][1].item<float>();
        point.z = basic_pts[i][2].item<float>();
        ref_pts->push_back(point);
    }
    if(ikdtree.Root_Node != nullptr)
    {
        ikdtree.myreset();
    }
    if (ikdtree.Root_Node == nullptr)
    {
        ikdtree.set_downsample_param(0.4);
        ikdtree.Build(ref_pts->points);
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
            ikdtree.Nearest_Search(pointSel_tmpt, NUM_MATCH_POINTS, points_near, pointSearchSqDis_surf);
            std::cout<<"point (x, y, z): "<<pointSel_tmpt.x<<", "<<pointSel_tmpt.y<<", "<<pointSel_tmpt.x<<std::endl;
            for(int i=0;i<pointSearchSqDis_surf.size();i++)
            {
                std::cout<<i<<", dist: "<<pointSearchSqDis_surf[i]<<std::endl;
                std::cout<<"(x, y, z): "<<points_near[i].x<<", "<<points_near[i].y<<", "<<points_near[i].x<<std::endl;
            }
        }
    }

}

torch::Tensor ACMMP::align_pts(const torch::Tensor & src_pts_torch, const torch::Tensor & T_torch)
{
    torch::Tensor T_torch_new = torch::eye(4);
    auto shape = src_pts_torch.sizes();
    if(shape.size()!=2 || shape[1]<3)
    {
        std::cout<<"Wrong shape of pts!!"<<std::endl;
        return T_torch_new;
    }
    int n_pts = shape[0];
    if(n_pts<NUM_MATCH_POINTS)
    {
        std::cout<<"Too few pts!!"<<std::endl;
        return T_torch_new;
    }
    src_pts->clear();
    for(int i=0;i<n_pts;i++)
    {
        PointType point;
        point.x = src_pts_torch[i][0].item<float>();
        point.y = src_pts_torch[i][1].item<float>();
        point.z = src_pts_torch[i][2].item<float>();
        src_pts->push_back(point);
    }
    std::cout<<"src_pc->points.size(): "<<src_pts->points.size()<<std::endl;
    src_pts_updated->resize(src_pts->points.size());
    point_selected_surf.resize(src_pts->points.size(), 0);				 //* 每个点是否是面点
    // std::cout<<"init T: "<<T_torch<<std::endl;
    for(int i=0;i<3;i++)
    {
        state.rot_end(i,0) = T_torch[i][0].item<float>();
        state.rot_end(i,1) = T_torch[i][1].item<float>();
        state.rot_end(i,2) = T_torch[i][2].item<float>();
        state.pos_end(i)   = T_torch[i][3].item<float>();
    }
    std::cout<<"state.rot_end: \n"<<state.rot_end<<std::endl;
    std::cout<<"state.pos_end: \n"<<state.pos_end<<std::endl;
    laserCloudOri->clear(); //* 存放旋转之后的点
	coeffSel->clear();
    findCorrespondingSurfFeatures(0,false, state);
    std::cout<<"coeffSel->points.size(): "<<coeffSel->points.size()<<std::endl;
    std::cout<<"laserCloudOri->points.size(): "<<laserCloudOri->points.size()<<std::endl;
    int laserCloudSelNum = laserCloudOri->points.size();
    Eigen::Matrix<double, 3, 6> J_p_xi;
    Eigen::Matrix<double, 1, 6> J_xi;
    Eigen::Matrix<double, 6, 6> H;
    Eigen::Matrix<double, 6, 1> b;
    Eigen::Matrix<double, 6, 1> delta_xi;
    J_p_xi.setZero();
    H.setZero();
    // I_STATE.setIdentity();
    b.setZero();
    J_xi.setZero();
    delta_xi.setZero();
    J_p_xi.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    for(int i=0;i<laserCloudSelNum;i++)
    {
        const PointType &norm_p = coeffSel->points[i];
        Eigen::Vector3d norm_vec(norm_p.x, norm_p.y, norm_p.z);						  //* 当前点对应的平面法向量
        Eigen::Matrix3d point_crossmat;
        Eigen::Vector3d point_this(laserCloudOri->points[i].x, laserCloudOri->points[i].y, laserCloudOri->points[i].z);
		point_crossmat << SKEW_SYM_MATRIX(point_this);
        J_p_xi.block<3, 3>(0, 3) = -1*point_crossmat;
        J_xi = norm_vec.transpose()*J_p_xi;
        H += J_xi.transpose() * J_xi;
        b += J_xi.transpose() * norm_p.intensity;
    }
    delta_xi = - H.inverse() * b;
    std::cout<<"delta xi:\n"<<delta_xi.transpose()<<std::endl;
    Eigen::Matrix<double, 4, 4> delta_T = se3Exp(delta_xi);
    Eigen::Matrix<double, 4, 4> T, T_new;
    T.setIdentity();
    T_new.setIdentity();
    T.block<3,3>(0,0) = state.rot_end;
    T.block<3,1>(0,3) = state.pos_end;
    T_new = delta_T * T;
    std::cout<<"delta_T:\n"<<delta_T<<std::endl; //* se3Exp
    std::cout<<"T_new:\n"<<T_new<<std::endl; //* se3Exp
    for(int i=0;i<4;i++)
        for(int j=0;j<4;j++)
        {
            T_torch_new[i][j] = T_new(i,j);
        }
    // GetUsedMask();
    return T_torch_new;
}

torch::Tensor ACMMP::GetUsedMask()
{
    torch::Tensor used_mask = torch::from_blob(point_selected_surf.data(),{int64_t(point_selected_surf.size())}).clone();
    // std::cout<<used_mask[0]<<std::endl;
    return used_mask;
}

void ACMMP::findCorrespondingSurfFeatures(int iterCount, bool rematch_en, StatesGroup &lio_state)
{
	int src_pts_size = src_pts->points.size();
	double maximum_pt_range = 0.0;
	for (int i = 0; i < src_pts_size; i += 1)
	{
		PointType &pointOri_tmpt = src_pts->points[i];
		double ori_pt_dis = sqrt(pointOri_tmpt.x * pointOri_tmpt.x + pointOri_tmpt.y * pointOri_tmpt.y + pointOri_tmpt.z * pointOri_tmpt.z);
		maximum_pt_range = std::max(ori_pt_dis, maximum_pt_range);
		PointType &pointSel_tmpt = src_pts_updated->points[i];
		//* 将当前点转换到世界坐标系
		//! 下次循环改变的在这里, 雷达位姿变了之后，雷达投影点的世界坐标变化
		pointBodyToWorld(&pointOri_tmpt, &pointSel_tmpt, lio_state); //* 利用g_lio_state
		std::vector<float> pointSearchSqDis_surf;
		PointVector points_near;
		if (iterCount == 0 || rematch_en)
		{
			point_selected_surf[i] = 1;
			ikdtree.Nearest_Search(pointSel_tmpt, NUM_MATCH_POINTS, points_near, pointSearchSqDis_surf);
			float max_distance = pointSearchSqDis_surf[NUM_MATCH_POINTS - 1];
			if (max_distance > m_maximum_pt_kdtree_dis) //* 最近的几个点中最远距离1.0超多一定阈值，不是平面点
				point_selected_surf[i] = 0;
		}
		if (point_selected_surf[i] == 0)
			continue;
		//* matX0 = argmin(X) || matA0 · X - matB0 ||
		//* 拟合平面 AX+BY+CZ+1=0
		cv::Mat matA0(NUM_MATCH_POINTS, 3, CV_32F, cv::Scalar::all(0));
		cv::Mat matB0(NUM_MATCH_POINTS, 1, CV_32F, cv::Scalar::all(-1));
		cv::Mat matX0(NUM_MATCH_POINTS, 1, CV_32F, cv::Scalar::all(0));
		for (int j = 0; j < NUM_MATCH_POINTS; j++)
		{
			matA0.at<float>(j, 0) = points_near[j].x;
			matA0.at<float>(j, 1) = points_near[j].y;
			matA0.at<float>(j, 2) = points_near[j].z;
		}
		cv::solve(matA0, matB0, matX0, cv::DECOMP_QR); // TODO
		float pa = matX0.at<float>(0, 0);
		float pb = matX0.at<float>(1, 0);
		float pc = matX0.at<float>(2, 0);
		float pd = 1;
		//* 归一化法向量 平面 AX+BY+CZ+D=0 norm(A, B, C)=1
		float ps = sqrt(pa * pa + pb * pb + pc * pc);
		pa /= ps;
		pb /= ps;
		pc /= ps;
		pd /= ps;
		bool planeValid = true;
		for (int j = 0; j < NUM_MATCH_POINTS; j++)
			//* 点到拟合平面距离
			if (fabs(pa * points_near[j].x + pb * points_near[j].y + pc * points_near[j].z + pd) > m_planar_check_dis) // Raw 0.05
				if (ori_pt_dis < maximum_pt_range * 0.90 || (ori_pt_dis < m_long_rang_pt_dis))
				{
					planeValid = false;
					point_selected_surf[i] = 0;
					break;
				}
		if (planeValid)
		{
			//* 当前点到平面距离
			float pd2 = pa * pointSel_tmpt.x + pb * pointSel_tmpt.y + pc * pointSel_tmpt.z + pd;
			double acc_distance = (ori_pt_dis < m_long_rang_pt_dis) ? m_maximum_res_dis : 1.0;
			if (pd2 < acc_distance && std::abs(pd2) <= 2.0)
			{
				point_selected_surf[i] = 1;
				PointType point;
				point.x = pa;
				point.y = pb;
				point.z = pc;
				point.intensity = pd2;
				coeffSel->push_back(point);                   //* 拟合平面参数
				laserCloudOri->push_back(pointSel_tmpt);      //* 被选中的点
			}
			else
				point_selected_surf[i] = 0;
		}
	}
}

void ACMMP::pointBodyToWorld(PointType const *const pi, PointType *const po, StatesGroup &lio_state)
{
	//* 比如int const*a;，实际上可以看成是int const (*a)，这表示指针a所指向的地址可以变，但是所指向的那个值不能变。
	//* 而int *const a;，可以看成int* (const a);，我们都知道a的值其实是一个地址，这就表示a所保存的地址是不可以变的，但是这个地址对应的值是可以变的。
	Eigen::Vector3d p_body(pi->x, pi->y, pi->z);
	Eigen::Vector3d p_global(lio_state.rot_end * p_body  + lio_state.pos_end);
	po->x = p_global(0);
	po->y = p_global(1);
	po->z = p_global(2);
	po->intensity = pi->intensity;
	// cout<<"p_body: "<<p_body.transpose()<<endl;
	// cout<<"p_global: "<<p_global.transpose()<<endl;
	// exit(1);
}

void ACMMP::acmmp_init_test(const std::string &dense_folder_, const int64_t ref_idx, const torch::Tensor &image_torch, const torch::Tensor &K_torch, const torch::Tensor &ext_torch, const torch::Tensor & depth_torch)
{
    // kdtree_test();
    dense_folder = dense_folder_;
    std::cout << "Run in ACMMP!" << std::endl;
    // std::cout << "dense_folder: "<< dense_folder << std::endl;
    if(!file_or_path_exist(dense_folder))
    {
        std::cout<<"dense_folder is not valid!!"<<std::endl;
    }
    std::string output_folder = dense_folder + std::string("/ACMMP");
    if(!file_or_path_exist(output_folder))
    {
        mkdir(output_folder.c_str(), 0777);
    }
    images.clear();
    cameras.clear();
    images_orig.clear();
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
    // std::cout << "input image_torch sizes: " << sizes << std::endl; //* [5, 1002, 1253, 3]
    // std::cout << "ndim: " << sizes.size() << std::endl;             //* 4
    // std::cout << "device: " << device << std::endl;                 //* cuda:0
    // std::cout << "dtype: " << dtype << std::endl;                   //* float
    if (ext_sizes.size() != 3 || ext_sizes[0] != sizes[0] || K_sizes[0] != 3 || K_sizes[1] != 3 || K_sizes.size() != 2)
    {
        std::cout << "Extrinsic size error or K error!" << std::endl;
        std::cout << "input ext sizes: " << ext_sizes << std::endl; //* [5, 1002, 1253, 3]
        return;
    }
    // std::cout << "K_torch:\n" << K_torch << std::endl;
    // cv::Mat K =  cv::Mat{3, 3, CV_32FC1, K_torch.clone().to(torch::kCPU).contiguous().data_ptr<float>()}; //*这中操作还没有复制到新的地址。。。。。
    //* 先保存到新的地地址，再转换
    torch::Tensor K_tmp = K_torch.clone().to(torch::kCPU).contiguous();
    cv::Mat K = cv::Mat{3, 3, CV_32FC1, K_tmp.data_ptr<float>()};
    // std::cout << "K:\n" << K << std::endl;
    int N = -1, H = -1, W = -1, C = -1;
    if (ndim != 4)
    {
        std::cout<<"Bad dim!"<<std::endl;
    }
    N = sizes[0]; C = sizes[1]; H = sizes[2]; W = sizes[3];
    torch::Tensor depth_tmp = depth_torch.to(torch::kCPU).contiguous();
    init_depth = cv::Mat{H, W, CV_32FC(1), depth_tmp.data_ptr<float>()};
    for (int i = 0; i < N; i++)
    {
        //* 将tensor转为cv::Mat。 先转换到cpu，不然会出错，并要连续化
        torch::Tensor img_i = image_torch.index_select(0, torch::tensor({i}).to(image_torch.device())).to(torch::kCPU).permute({0, 2, 3, 1}).contiguous();
        cv::Mat img = cv::Mat{H, W, CV_32FC(C), img_i.data_ptr<float>()}; //* 3通道float
        cv::Mat img_uint, img_gray, img_gray_f;
        img.convertTo(img_uint, CV_8UC3);
        cv::cvtColor(img_uint, img_gray, cv::COLOR_BGR2GRAY);
        img_gray.convertTo(img_gray_f, CV_32FC1);
        images.push_back(img_gray_f);
        images_orig.push_back(img_uint);
        torch::Tensor ext_cpu = ext_torch.index_select(0, torch::tensor({i}).to(ext_torch.device())).to(torch::kCPU).contiguous();
        cv::Mat ext = cv::Mat{4, 4, CV_32FC(1), ext_cpu.data_ptr<float>()};
        if (i == 0 && 0)
        {
            auto size_i = img_i.sizes();
            std::cout << "img_i sizes: " << size_i << std::endl;
            // std::cout<<"img_i : "<<img_i.squeeze()<<std::endl;
            std::cout << "img.rows: " << img.rows << std::endl
                      << "img.cols: " << img.cols << std::endl
                      << "img.channels(): " << img.channels() << std::endl;
            std::cout << ext << std::endl;
            // int length = 4;
            // for(int h_i=0;h_i<length;h_i++)
            //     for(int w_i=0;w_i<length;w_i++)
            //     {
            //         auto p_i = img.at<cv::Vec3f>(h_i,w_i);
            //         std::cout<<p_i<<std::endl;
            //     }
        }
        Camera camera;
        for (int i = 0; i < 3; ++i)
        {
            camera.K[3 * i + 0] = K.at<float>(i,0);
            camera.K[3 * i + 1] = K.at<float>(i,1);
            camera.K[3 * i + 2] = K.at<float>(i,2);
            camera.R[3 * i + 0] = ext.at<float>(i,0);
            camera.R[3 * i + 1] = ext.at<float>(i,1);
            camera.R[3 * i + 2] = ext.at<float>(i,2);
            camera.t[i] = ext.at<float>(i,3);
        }
        camera.depth_min = 0.5;
        camera.depth_max = 45.0;
        camera.height = H;
        camera.width = W;
        cameras.push_back(camera);
    }
    // std::cout << "images.size(): " << images.size() << std::endl;
    Problem problem;
    max_num_downscale = -1;
    int size_bound = 1000;
    int rows = images[0].rows;
    int cols = images[0].cols;
    int max_size = std::max(rows, cols);
    if (max_size > params.max_image_size)
        max_size = params.max_image_size;
    problem.max_image_size = max_size;
    int k = 0;
    while (max_size > size_bound)
    {
        max_size /= 2;
        k++;
    }
    if (k > max_num_downscale)
        max_num_downscale = k;
    problem.num_downscale = k;
    problem.ref_image_id = ref_idx;
    // std::cout<< "ref_idx: " <<ref_idx<<std::endl;
    problems.push_back(problem);
    Run();
}

void ACMMP::Run()
{
    size_t num_images = problems.size();
    std::cout << "There are " << num_images << " problems needed to be processed!" << std::endl;
    int flag = 0;
    int geom_iterations = 2;
    bool geom_consistency = false;
    bool planar_prior = false;
    bool hierarchy = false;
    bool multi_geometry = false;
    ProcessProblem(geom_consistency, planar_prior, hierarchy);
    RunFusion(geom_consistency);
}

void ACMMP::ProcessProblem(bool geom_consistency, bool planar_prior, bool hierarchy, bool multi_geometrty)
{
    int idx = 0;
    const Problem problem = problems[idx];
    cudaSetDevice(0);
    std::stringstream result_path;
    result_path << dense_folder << "/ACMMP" << "/2333_" << std::setw(8) << std::setfill('0') << problem.ref_image_id;
    std::string result_folder = result_path.str();
    if(!file_or_path_exist(result_folder))
    {
        mkdir(result_folder.c_str(), 0777);
    }
    params.init_depth_flag = false;
    // std::cout<<"InuputInitialization"<<std::endl;
    InuputInitialization_simple();
    // std::cout<<"CudaSpaceInitialization"<<std::endl;
    CudaSpaceInitialization(dense_folder, problem);
    // std::cout<<"RunPatchMatch"<<std::endl;
    RunPatchMatch();
    const int width = GetReferenceImageWidth();
    const int height = GetReferenceImageHeight();
    cv::Mat_<float> depths = cv::Mat::zeros(height, width, CV_32FC1);
    cv::Mat_<cv::Vec3f> normals = cv::Mat::zeros(height, width, CV_32FC3);
    cv::Mat_<float> costs = cv::Mat::zeros(height, width, CV_32FC1);
    for (int col = 0; col < width; ++col)
    {
        for (int row = 0; row < height; ++row)
        {
            int center = row * width + col;
            float4 plane_hypothesis = GetPlaneHypothesis(center);
            depths(row, col) = plane_hypothesis.w;
            normals(row, col) = cv::Vec3f(plane_hypothesis.x, plane_hypothesis.y, plane_hypothesis.z);
            costs(row, col) = GetCost(center);
        }
    }
    if (planar_prior && 0) {
        std::cout << "Run Planar Prior Assisted PatchMatch MVS ..." << std::endl;
        SetPlanarPriorParams();
        const cv::Rect imageRC(0, 0, width, height);
        std::vector<cv::Point> support2DPoints;
        GetSupportPoints(support2DPoints);
        const auto triangles = DelaunayTriangulation(imageRC, support2DPoints);
        cv::Mat refImage = GetReferenceImage().clone();
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
                float4 n4 = GetPriorPlaneParams(triangle, depths);
                planeParams_tri.push_back(n4);
                idx++;
            }
        }
        cv::Mat_<float> priordepths = cv::Mat::zeros(height, width, CV_32FC1);
        for (int i = 0; i < width; ++i) {
            for (int j = 0; j < height; ++j) {
                if (mask_tri(j, i) > 0) {
                    //* 计算原点到当前点的距离
                    float d = GetDepthFromPlaneParam(planeParams_tri[mask_tri(j, i) - 1], i, j);
                    if (d <= GetMaxDepth() && d >= GetMinDepth()) {
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
        CudaPlanarPriorInitialization(planeParams_tri, mask_tri);
        RunPatchMatch();
        for (int col = 0; col < width; ++col) {
            for (int row = 0; row < height; ++row) {
                int center = row * width + col;
                float4 plane_hypothesis = GetPlaneHypothesis(center);
                depths(row, col) = plane_hypothesis.w;
                normals(row, col) = cv::Vec3f(plane_hypothesis.x, plane_hypothesis.y, plane_hypothesis.z);
                costs(row, col) = GetCost(center);
            }
        }
    }
    std::string suffix = "/depths.dmb";
    std::string depth_path = result_folder + suffix;
    std::string normal_path = result_folder + "/normals.dmb";
    std::string cost_path = result_folder + "/costs.dmb";
    writeDepthDmb(depth_path, depths);
    // writeDepthDmb(depth_path, init_depth);
    writeNormalDmb(normal_path, normals);
    writeDepthDmb(cost_path, costs);
    std::cout<< "Writing depths, normals and costs to "<<result_folder<<std::endl;
    std::cout << "Processing image " << std::setw(8) << std::setfill('0') << problem.ref_image_id << " done!" << std::endl;
}

torch::Tensor ACMMP::GetDepth()
{
    std::stringstream result_path;
    result_path << dense_folder << "/ACMMP" << "/2333_" << std::setw(8) << std::setfill('0') << problems[0].ref_image_id;
    std::string result_folder = result_path.str();
    std::string suffix = "/depths.dmb";
    std::string depth_path = result_folder + suffix;
    // std::string normal_path = result_folder + "/normals.dmb";
    cv::Mat_<float> depth;
    // cv::Mat_<cv::Vec3f> normal;
    readDepthDmb(depth_path, depth);
    // readNormalDmb(normal_path, normal);
    const int cols = depth.cols;
    const int rows = depth.rows;
    torch::Tensor output = torch::from_blob(depth.ptr<float>(), /*sizes=*/{rows, cols}).clone();
    return output;
}

torch::Tensor ACMMP::GetCosts()
{
    std::stringstream result_path;
    result_path << dense_folder << "/ACMMP" << "/2333_" << std::setw(8) << std::setfill('0') << problems[0].ref_image_id;
    std::string result_folder = result_path.str();
    // std::string suffix = "/depths.dmb";
    // std::string depth_path = result_folder + suffix;
    std::string costs_path = result_folder + "/costs.dmb";
    cv::Mat_<float> costs;
    // cv::Mat_<cv::Vec3f> normal;
    readDepthDmb(costs_path, costs);
    // readNormalDmb(normal_path, normal);
    const int cols = costs.cols;
    const int rows = costs.rows;
    torch::Tensor output = torch::from_blob(costs.ptr<float>(), /*sizes=*/{rows, cols});
    return output.clone();
}

void ACMMP::InuputInitialization_simple()
{
    int idx=0;
    const Problem problem = problems[idx];
    // std::cout << "限制图像的尺寸" << std::endl;
    //* ========限制图像的尺寸==============
    // Scale cameras and images
    int max_image_size = problems[idx].cur_image_size;
    // std::cout << "images.size(): " << images.size() << std::endl;
    // for (size_t i = 0; i < images.size(); ++i)
    // {
    //     if (images[i].cols <= max_image_size && images[i].rows <= max_image_size)
    //     {
    //         continue;
    //     }
    //     const float factor_x = static_cast<float>(max_image_size) / images[i].cols;
    //     const float factor_y = static_cast<float>(max_image_size) / images[i].rows;
    //     const float factor = std::min(factor_x, factor_y);
    //     const int new_cols = std::round(images[i].cols * factor);
    //     const int new_rows = std::round(images[i].rows * factor);
    //     const float scale_x = new_cols / static_cast<float>(images[i].cols);
    //     const float scale_y = new_rows / static_cast<float>(images[i].rows);
    //     cv::Mat_<float> scaled_image_float, scaled_init_depth_float;
    //     // std::cout << "scaled_image_float: " << std::endl;
    //     cv::resize(images[i], scaled_image_float, cv::Size(new_cols, new_rows), 0, 0, cv::INTER_LINEAR);
    //     cv::resize(init_depth, scaled_init_depth_float, cv::Size(new_cols, new_rows), 0, 0, cv::INTER_LINEAR);
    //     // std::cout << "scaled_image_float end " << std::endl;
    //     init_depth = scaled_init_depth_float.clone();
    //     images[i] = scaled_image_float.clone();
    //     cameras[i].K[0] *= scale_x;
    //     cameras[i].K[2] *= scale_x;
    //     cameras[i].K[4] *= scale_y;
    //     cameras[i].K[5] *= scale_y;
    //     cameras[i].height = scaled_image_float.rows;
    //     cameras[i].width = scaled_image_float.cols;
    // }
    std::cout << "depth range" << std::endl;
    params.depth_min = cameras[0].depth_min * 0.6f;
    params.depth_max = cameras[0].depth_max * 1.2f;
    std::cout << "depthe range: " << params.depth_min << " " << params.depth_max << std::endl;
    params.num_images = (int)images.size();
    std::cout << "num images: " << params.num_images << std::endl;
    if (params.geom_consistency && 0) //* 暂时不考虑几何一致
    {
        depths.clear();
        std::stringstream result_path;
        result_path << dense_folder << "/ACMMP" << "/2333_" << std::setw(8) << std::setfill('0') << problem.ref_image_id;
        std::string result_folder = result_path.str();
        std::string suffix = "/depths.dmb";
        if (params.multi_geometry)
        {
            suffix = "/depths_geom.dmb";
        }
        std::string depth_path = result_folder + suffix;
        cv::Mat_<float> ref_depth;
        readDepthDmb(depth_path, ref_depth);
        depths.push_back(ref_depth);
        size_t num_src_images = problem.src_image_ids.size();
        for (size_t i = 0; i < num_src_images; ++i)
        {
            std::stringstream result_path;
            result_path << dense_folder << "/ACMMP" << "/2333_" << std::setw(8) << std::setfill('0') << problem.src_image_ids[i];
            std::string result_folder = result_path.str();
            std::string depth_path = result_folder + suffix;
            cv::Mat_<float> depth;
            readDepthDmb(depth_path, depth);
            depths.push_back(depth);
        }
    }
}

void ACMMP::InuputInitialization(const std::string &dense_folder, const std::vector<Problem> &problems, const int idx)
{
    images.clear();
    cameras.clear();
    const Problem problem = problems[idx];
    std::string image_folder = dense_folder + std::string("/images");
    std::string cam_folder = dense_folder + std::string("/cams");
    std::cout << "读取参考图的灰度图和相机参数" << std::endl;
    //* =======读取参考图的灰度图和相机参数=============
    //* 读取灰度图转为float存入images
    std::stringstream image_path;
    image_path << image_folder << "/" << std::setw(8) << std::setfill('0') << problem.ref_image_id << ".jpg";
    cv::Mat_<uint8_t> image_uint = cv::imread(image_path.str(), cv::IMREAD_GRAYSCALE);
    cv::Mat image_float;
    image_uint.convertTo(image_float, CV_32FC1);
    images.push_back(image_float);
    //* 读取相机参数：位姿、内参、深度范围、尺寸，存入cameras
    std::stringstream cam_path;
    cam_path << cam_folder << "/" << std::setw(8) << std::setfill('0') << problem.ref_image_id << "_cam.txt";
    Camera camera = ReadCamera(cam_path.str());
    camera.height = image_float.rows;
    camera.width = image_float.cols;
    cameras.push_back(camera);

    std::cout << "读取源图灰度图及其相机参数" << std::endl;
    //* =============读取源图灰度图及其相机参数================
    size_t num_src_images = problem.src_image_ids.size();
    for (size_t i = 0; i < num_src_images; ++i)
    {
        std::stringstream image_path;
        image_path << image_folder << "/" << std::setw(8) << std::setfill('0') << problem.src_image_ids[i] << ".jpg";
        cv::Mat_<uint8_t> image_uint = cv::imread(image_path.str(), cv::IMREAD_GRAYSCALE);
        cv::Mat image_float;
        image_uint.convertTo(image_float, CV_32FC1);
        images.push_back(image_float);
        std::stringstream cam_path;
        cam_path << cam_folder << "/" << std::setw(8) << std::setfill('0') << problem.src_image_ids[i] << "_cam.txt";
        Camera camera = ReadCamera(cam_path.str());
        camera.height = image_float.rows;
        camera.width = image_float.cols;
        cameras.push_back(camera);
    }

    std::cout << "限制图像的尺寸" << std::endl;
    //* ========限制图像的尺寸==============
    // Scale cameras and images
    int max_image_size = problems[idx].cur_image_size;
    std::cout << "images.size(): " << images.size() << std::endl;
    for (size_t i = 0; i < images.size(); ++i)
    {
        // if (i > 0) {
        //     max_image_size = problems[problem.src_image_ids[i - 1]].cur_image_size;
        // }
        if (images[i].cols <= max_image_size && images[i].rows <= max_image_size)
        {
            continue;
        }
        const float factor_x = static_cast<float>(max_image_size) / images[i].cols;
        const float factor_y = static_cast<float>(max_image_size) / images[i].rows;
        const float factor = std::min(factor_x, factor_y);
        const int new_cols = std::round(images[i].cols * factor);
        const int new_rows = std::round(images[i].rows * factor);
        const float scale_x = new_cols / static_cast<float>(images[i].cols);
        const float scale_y = new_rows / static_cast<float>(images[i].rows);
        cv::Mat_<float> scaled_image_float;
        cv::resize(images[i], scaled_image_float, cv::Size(new_cols, new_rows), 0, 0, cv::INTER_LINEAR);
        images[i] = scaled_image_float.clone();
        cameras[i].K[0] *= scale_x;
        cameras[i].K[2] *= scale_x;
        cameras[i].K[4] *= scale_y;
        cameras[i].K[5] *= scale_y;
        cameras[i].height = scaled_image_float.rows;
        cameras[i].width = scaled_image_float.cols;
    }
    std::cout << "depth range" << std::endl;
    params.depth_min = cameras[0].depth_min * 0.6f;
    params.depth_max = cameras[0].depth_max * 1.2f;
    std::cout << "depthe range: " << params.depth_min << " " << params.depth_max << std::endl;
    params.num_images = (int)images.size();
    std::cout << "num images: " << params.num_images << std::endl;
    params.disparity_min = cameras[0].K[0] * params.baseline / params.depth_max;
    params.disparity_max = cameras[0].K[0] * params.baseline / params.depth_min;
    if (params.geom_consistency)
    {
        depths.clear();
        std::stringstream result_path;
        result_path << dense_folder << "/ACMMP" << "/2333_" << std::setw(8) << std::setfill('0') << problem.ref_image_id;
        std::string result_folder = result_path.str();
        std::string suffix = "/depths.dmb";
        if (params.multi_geometry)
        {
            suffix = "/depths_geom.dmb";
        }
        std::string depth_path = result_folder + suffix;
        cv::Mat_<float> ref_depth;
        readDepthDmb(depth_path, ref_depth);
        depths.push_back(ref_depth);
        size_t num_src_images = problem.src_image_ids.size();
        for (size_t i = 0; i < num_src_images; ++i)
        {
            std::stringstream result_path;
            result_path << dense_folder << "/ACMMP" << "/2333_" << std::setw(8) << std::setfill('0') << problem.src_image_ids[i];
            std::string result_folder = result_path.str();
            std::string depth_path = result_folder + suffix;
            cv::Mat_<float> depth;
            readDepthDmb(depth_path, depth);
            depths.push_back(depth);
        }
    }
}

void ACMMP::JointBilateralUpsampling()
{
    int idx = 0;
    const Problem problem = problems[idx];
    int acmmp_size = problem.cur_image_size;
    std::stringstream result_path;
    result_path << dense_folder << "/ACMMP" << "/2333_" << std::setw(8) << std::setfill('0') << problem.ref_image_id;
    std::string result_folder = result_path.str();
    std::string suffix = "/depths.dmb";
    if (params.multi_geometry)
    {
        suffix = "/depths_geom.dmb";
    }
    std::string depth_path = result_folder + suffix;
    cv::Mat_<float> ref_depth;
    readDepthDmb(depth_path, ref_depth);
    // std::string image_folder = dense_folder + std::string("/images");
    // std::stringstream image_path;
    // image_path << image_folder << "/" << std::setw(8) << std::setfill('0') << problem.ref_image_id << ".jpg";
    // cv::Mat_<uint8_t> image_uint = cv::imread(image_path.str(), cv::IMREAD_GRAYSCALE);
    cv::Mat image_float = images_orig[0].clone();
    // image_uint.convertTo(image_float, CV_32FC1);
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

void ACMMP::RunFusion(bool geom_consistency)
{
    std::cout<<"Fusing..."<<std::endl;
    size_t num_images = problems.size();
    std::vector<cv::Mat_<float>> depths;
    std::vector<cv::Mat_<cv::Vec3f>> normals;
    std::vector<cv::Mat> masks;
    depths.clear();
    normals.clear();
    masks.clear();

    for (size_t i = 0; i < num_images; ++i) {
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
        depths.push_back(depth);
        normals.push_back(normal);
        cv::Mat mask = cv::Mat::zeros(depth.rows, depth.cols, CV_8UC1);
        masks.push_back(mask);
    }
    // std::cout<< "images.size(): "<<images.size()<<std::endl;
    bool single_proj = false;
    if(num_images<2)
    {
        std::cout<<"Only one image, no fuse"<<std::endl;
        single_proj = true;
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
                // std::cout<< "ref_depth: "<<ref_depth<<std::endl;
                if (ref_depth <= 0.0)
                    continue;
                float3 PointX = Get3DPointonWorld(c, r, ref_depth, cameras[i]);
                float3 consistent_Point = PointX;
                cv::Vec3f consistent_normal = ref_normal;
                float consistent_Color[3] = {(float)images_orig[i].at<cv::Vec3b>(r, c)[0], (float)images_orig[i].at<cv::Vec3b>(r, c)[1], (float)images_orig[i].at<cv::Vec3b>(r, c)[2]};
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
    std::cout<<"PointCloud.size(): "<<PointCloud.size()<<std::endl;
    StoreColorPlyFileBinaryPointCloud (ply_path, PointCloud);
    std::cout<<"Saving to "<<ply_path<<std::endl;
}

void ACMMP::RunPatchMatch()
{
    RunPatchMatch_cuda(cameras, texture_objects_cuda, cameras_cuda, plane_hypotheses_cuda, scaled_plane_hypotheses_cuda, costs_cuda, pre_costs_cuda, rand_states_cuda, selected_views_cuda, prior_planes_cuda, plane_masks_cuda, params, texture_depths_cuda, plane_hypotheses_host, costs_host, depths_cuda);
}

torch::Tensor ACMMP::GetDepthEdge()
{
    find_edge_cuda(cameras[0], depths_cuda, depth_edge_mask_cuda, depth_edge_mask_host);
    const int cols = cameras[0].width;
    const int rows = cameras[0].height;
    torch::Tensor output = torch::from_blob(depth_edge_mask_host, /*sizes=*/{rows, cols}).clone();
    return output;
}


void ACMMP::SetGeomConsistencyParams(bool multi_geometry = false)
{
    params.geom_consistency = true;
    params.max_iterations = 2;
    if (multi_geometry)
    {
        params.multi_geometry = true;
    }
}

void ACMMP::SetHierarchyParams()
{
    params.hierarchy = true;
}

void ACMMP::SetPlanarPriorParams()
{
    params.planar_prior = true;
}

void ACMMP::CudaPlanarPriorInitialization(const std::vector<float4> &PlaneParams, const cv::Mat_<float> &masks)
{
    prior_planes_host = new float4[cameras[0].height * cameras[0].width];
    cudaMalloc((void **)&prior_planes_cuda, sizeof(float4) * (cameras[0].height * cameras[0].width));

    plane_masks_host = new unsigned int[cameras[0].height * cameras[0].width];
    cudaMalloc((void **)&plane_masks_cuda, sizeof(unsigned int) * (cameras[0].height * cameras[0].width));

    for (int i = 0; i < cameras[0].width; ++i)
    {
        for (int j = 0; j < cameras[0].height; ++j)
        {
            int center = j * cameras[0].width + i;
            plane_masks_host[center] = (unsigned int)masks(j, i);
            if (masks(j, i) > 0)
            {
                prior_planes_host[center] = PlaneParams[masks(j, i) - 1];
            }
        }
    }

    cudaMemcpy(prior_planes_cuda, prior_planes_host, sizeof(float4) * (cameras[0].height * cameras[0].width), cudaMemcpyHostToDevice);
    cudaMemcpy(plane_masks_cuda, plane_masks_host, sizeof(unsigned int) * (cameras[0].height * cameras[0].width), cudaMemcpyHostToDevice);
}

int ACMMP::GetReferenceImageWidth()
{
    return cameras[0].width;
}

int ACMMP::GetReferenceImageHeight()
{
    return cameras[0].height;
}

cv::Mat ACMMP::GetReferenceImage()
{
    return images[0];
}

float4 ACMMP::GetPlaneHypothesis(const int index)
{
    return plane_hypotheses_host[index];
}

float ACMMP::GetCost(const int index)
{
    return costs_host[index];
}

float ACMMP::GetMinDepth()
{
    return params.depth_min;
}

float ACMMP::GetMaxDepth()
{
    return params.depth_max;
}

void ACMMP::GetSupportPoints(std::vector<cv::Point> &support2DPoints)
{
    support2DPoints.clear();
    const int step_size = 5;
    const int width = GetReferenceImageWidth();
    const int height = GetReferenceImageHeight();
    for (int col = 0; col < width; col += step_size)
    {
        for (int row = 0; row < height; row += step_size)
        {
            float min_cost = 2.0f;
            cv::Point temp_point;
            int c_bound = std::min(width, col + step_size);
            int r_bound = std::min(height, row + step_size);
            for (int c = col; c < c_bound; ++c)
            {
                for (int r = row; r < r_bound; ++r)
                {
                    int center = r * width + c;
                    if (GetCost(center) < 2.0f && min_cost > GetCost(center))
                    {
                        temp_point = cv::Point(c, r);
                        min_cost = GetCost(center);
                    }
                }
            }
            if (min_cost < 0.1f)
            {
                support2DPoints.push_back(temp_point);
            }
        }
    }
}

std::vector<Triangle> ACMMP::DelaunayTriangulation(const cv::Rect boundRC, const std::vector<cv::Point> &points)
{
    if (points.empty())
    {
        return std::vector<Triangle>();
    }

    std::vector<Triangle> results;

    std::vector<cv::Vec6f> temp_results;
    cv::Subdiv2D subdiv2d(boundRC);
    for (const auto point : points)
    {
        subdiv2d.insert(cv::Point2f((float)point.x, (float)point.y));
    }
    subdiv2d.getTriangleList(temp_results);
    for (const auto temp_vec : temp_results)
    {
        cv::Point pt1((int)temp_vec[0], (int)temp_vec[1]);
        cv::Point pt2((int)temp_vec[2], (int)temp_vec[3]);
        cv::Point pt3((int)temp_vec[4], (int)temp_vec[5]);
        results.push_back(Triangle(pt1, pt2, pt3));
    }
    return results;
}

float4 ACMMP::GetPriorPlaneParams(const Triangle triangle, const cv::Mat_<float> depths)
{
    cv::Mat A(3, 4, CV_32FC1);
    cv::Mat B(4, 1, CV_32FC1);
    float3 ptX1 = Get3DPointonRefCam(triangle.pt1.x, triangle.pt1.y, depths(triangle.pt1.y, triangle.pt1.x), cameras[0]);
    float3 ptX2 = Get3DPointonRefCam(triangle.pt2.x, triangle.pt2.y, depths(triangle.pt2.y, triangle.pt2.x), cameras[0]);
    float3 ptX3 = Get3DPointonRefCam(triangle.pt3.x, triangle.pt3.y, depths(triangle.pt3.y, triangle.pt3.x), cameras[0]);
    A.at<float>(0, 0) = ptX1.x;
    A.at<float>(0, 1) = ptX1.y;
    A.at<float>(0, 2) = ptX1.z;
    A.at<float>(0, 3) = 1.0;
    A.at<float>(1, 0) = ptX2.x;
    A.at<float>(1, 1) = ptX2.y;
    A.at<float>(1, 2) = ptX2.z;
    A.at<float>(1, 3) = 1.0;
    A.at<float>(2, 0) = ptX3.x;
    A.at<float>(2, 1) = ptX3.y;
    A.at<float>(2, 2) = ptX3.z;
    A.at<float>(2, 3) = 1.0;
    cv::SVD::solveZ(A, B); //* z = argmin_{norm(x)} {Ax}
    float4 n4 = make_float4(B.at<float>(0, 0), B.at<float>(1, 0), B.at<float>(2, 0), B.at<float>(3, 0));
    float norm2 = sqrt(pow(n4.x, 2) + pow(n4.y, 2) + pow(n4.z, 2));
    if (n4.w < 0) //* 法线指向原点一侧
    {
        norm2 *= -1;
    }
    n4.x /= norm2;
    n4.y /= norm2;
    n4.z /= norm2;
    n4.w /= norm2;

    return n4;
}

float ACMMP::GetDepthFromPlaneParam(const float4 plane_hypothesis, const int x, const int y)
{
    return -plane_hypothesis.w * cameras[0].K[0] / ((x - cameras[0].K[2]) * plane_hypothesis.x + (cameras[0].K[0] / cameras[0].K[4]) * (y - cameras[0].K[5]) * plane_hypothesis.y + cameras[0].K[0] * plane_hypothesis.z);
}

void JBUAddImageToTextureFloatGray(std::vector<cv::Mat_<float>> &imgs, cudaTextureObject_t texs[], cudaArray *cuArray[], const int &numSelViews)
{
    for (int i = 0; i < numSelViews; i++)
    {
        int index = i;
        int rows = imgs[index].rows;
        int cols = imgs[index].cols;
        // Create channel with floating point type
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
        // Allocate array with correct size and number of channels
        cudaMallocArray(&cuArray[i], &channelDesc, cols, rows);
        cudaMemcpy2DToArray(cuArray[i], 0, 0, imgs[index].ptr<float>(), imgs[index].step[0], cols * sizeof(float), rows, cudaMemcpyHostToDevice);

        // Specify texture
        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(cudaResourceDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cuArray[i];

        // Specify texture object parameters
        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(cudaTextureDesc));
        texDesc.addressMode[0] = cudaAddressModeWrap;
        texDesc.addressMode[1] = cudaAddressModeWrap;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        // Create texture object
        cudaCreateTextureObject(&(texs[i]), &resDesc, &texDesc, NULL);
    }
    return;
}

JBU::JBU() {}

JBU::~JBU()
{
    free(depth_h);

    cudaFree(depth_d);
    cudaFree(jp_d);
    cudaFree(jt_d);
}

void JBU::CudaRun()
{
    CudaRun_cuda(jp_h, jp_d, jt_d, depth_d, depth_h);
}

void JBU::InitializeParameters(int n)
{
    depth_h = (float *)malloc(sizeof(float) * n);

    cudaMalloc((void **)&depth_d, sizeof(float) * n);

    cudaMalloc((void **)&jp_d, sizeof(JBUParameters) * 1);
    cudaMemcpy(jp_d, &jp_h, sizeof(JBUParameters) * 1, cudaMemcpyHostToDevice);

    cudaMalloc((void **)&jt_d, sizeof(JBUTexObj) * 1);
    cudaMemcpy(jt_d, &jt_h, sizeof(JBUTexObj) * 1, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}

void RunJBU(const cv::Mat_<float> &scaled_image_float, const cv::Mat_<float> &src_depthmap, const std::string &dense_folder, const Problem &problem)
{
    uint32_t rows = scaled_image_float.rows;
    uint32_t cols = scaled_image_float.cols;
    int Imagescale = std::max(scaled_image_float.rows / src_depthmap.rows, scaled_image_float.cols / src_depthmap.cols);

    if (Imagescale == 1)
    {
        std::cout << "Image.rows = Depthmap.rows" << std::endl;
        return;
    }

    std::vector<cv::Mat_<float>> imgs(JBU_NUM);
    imgs[0] = scaled_image_float.clone();
    imgs[1] = src_depthmap.clone();

    JBU jbu;
    jbu.jp_h.height = rows;
    jbu.jp_h.width = cols;
    jbu.jp_h.s_height = src_depthmap.rows;
    jbu.jp_h.s_width = src_depthmap.cols;
    jbu.jp_h.Imagescale = Imagescale;
    JBUAddImageToTextureFloatGray(imgs, jbu.jt_h.imgs, jbu.cuArray, JBU_NUM);

    jbu.InitializeParameters(rows * cols);
    jbu.CudaRun();

    cv::Mat_<float> depthmap = cv::Mat::zeros(rows, cols, CV_32FC1);

    for (uint32_t i = 0; i < cols; ++i)
    {
        for (uint32_t j = 0; j < rows; ++j)
        {
            int center = i + cols * j;
            if (jbu.depth_h[center] != jbu.depth_h[center])
            { //* nan
                // std::cout << "wrong!" << std::endl;
                // std::cout << "jbu.depth_h[center]: "<<jbu.depth_h[center] << std::endl;
                continue;
            }
            depthmap(j, i) = jbu.depth_h[center];
        }
    }

    cv::Mat_<float> disp0 = depthmap.clone();
    std::stringstream result_path;
    result_path << dense_folder << "/ACMMP" << "/2333_" << std::setw(8) << std::setfill('0') << problem.ref_image_id;
    std::string result_folder = result_path.str();
    mkdir(result_folder.c_str(), 0777);
    std::string depth_path = result_folder + "/depths.dmb";
    writeDepthDmb(depth_path, disp0);

    for (int i = 0; i < JBU_NUM; i++)
    {
        CUDA_SAFE_CALL(cudaDestroyTextureObject(jbu.jt_h.imgs[i]));
        CUDA_SAFE_CALL(cudaFreeArray(jbu.cuArray[i]));
    }
    cudaDeviceSynchronize();
}
