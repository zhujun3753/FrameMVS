#include "pointcloud_rgbd.hpp"

void CheckCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}

__device__ float generate(curandState *globalState, int ind)
{
	curandState localState = globalState[ind];
	float RANDOM = curand_uniform(&localState);// uniform distribution
	globalState[ind] = localState;
	return RANDOM;
}

__global__ void setup_kernel(int width, int height, curandState *state, unsigned long seed)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx >= width*height)
        return;
	curand_init(seed, idx, 0, &state[idx]);// initialize the state
}

// float atomicMin
__device__ __forceinline__ float fatomicMin(float *address, float val)
{
    int ret = __float_as_int(*address);
    while(val < __int_as_float(ret))
    {
        int old = ret;
        if((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
            break;
    }
    return __int_as_float(ret);
}

__global__ void proj_pts2depth_kernel(float3 * pts_cuda, int num_pts, float * K_cuda, float * T_cuda, float * depth_cuda, int width, int height)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx >= num_pts)
        return;
    const float3 & pt_w = pts_cuda[idx];
    float3 pt_c;
    pt_c.x = T_cuda[0]*pt_w.x + T_cuda[1]*pt_w.y + T_cuda[2] *pt_w.z + T_cuda[3];
    pt_c.y = T_cuda[4]*pt_w.x + T_cuda[5]*pt_w.y + T_cuda[6] *pt_w.z + T_cuda[7];
    pt_c.z = T_cuda[8]*pt_w.x + T_cuda[9]*pt_w.y + T_cuda[10]*pt_w.z + T_cuda[11];
    if(pt_c.z<0.1)
        return;
    float u,v;
    u = (K_cuda[0]*pt_c.x + K_cuda[1]*pt_c.z)/pt_c.z;
    v = (K_cuda[2]*pt_c.y + K_cuda[3]*pt_c.z)/pt_c.z;
    if(u<0 || v<0 || u>width-1 || v>height-1)
        return;
    int center = (int)((int)v*width + (int)u);
    fatomicMin(&depth_cuda[center], pt_c.z);
    // depth_cuda[center] = pt_c.z;
}

void proj_pts2depth(float3 * pts, int num_pts, float * K, float * T, float * depth, int width, int height)
{
	int threads = 512;
    int blocks = num_pts/threads + ((num_pts % threads)?1:0);
    float3 * pts_cuda;
    float * depth_cuda;
	float * K_cuda, *T_cuda;
    cudaMalloc((void**)&pts_cuda, sizeof(float3)*num_pts);
    cudaMalloc((void**)&depth_cuda, sizeof(float)*width*height);
    cudaMalloc((void**)&K_cuda, sizeof(float)*4);
    cudaMalloc((void**)&T_cuda, sizeof(float)*12);
    CheckCUDAError("cudaMalloc");
	// printf("num_pts:%d, width:%d, height:%d\n", num_pts, width, height);
    
	cudaMemcpy(pts_cuda, pts, sizeof(float)*num_pts, cudaMemcpyHostToDevice);
    CheckCUDAError("pts_cuda");
    cudaMemcpy(depth_cuda, depth, sizeof(float)*width*height, cudaMemcpyHostToDevice);
    // CheckCUDAError("depth_cuda");
    cudaMemcpy(K_cuda, K, sizeof(float)*4, cudaMemcpyHostToDevice);
    // CheckCUDAError("K_cuda");
    cudaMemcpy(T_cuda, T, sizeof(float)*12, cudaMemcpyHostToDevice);
    CheckCUDAError("cudaMemcpyHostToDevice");
    
    proj_pts2depth_kernel<<<blocks, threads>>>(pts_cuda, num_pts, K_cuda, T_cuda, depth_cuda, width, height);
    cudaThreadSynchronize();
    CheckCUDAError("cal_uv_cuda");

    cudaMemcpy(depth, depth_cuda, sizeof(float)*width*height, cudaMemcpyDeviceToHost);
    CheckCUDAError("cudaMemcpyDeviceToHost");

    cudaFree(pts_cuda);
    cudaFree(depth_cuda);
    cudaFree(K_cuda);
    cudaFree(T_cuda);
    CheckCUDAError("cudaFree");
    // printf("CUDA cudaFree end\n");
}

__global__ void proj_pts2depth_with_attr_kernel(float3 * pts_cuda, int * valid_local_ids_cuda, int num_pts, float * K_cuda, float * T_cuda, float * depth_cuda, int width, int height, int opertator_flag)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx >= num_pts)
        return;
    if(opertator_flag==0)
    {
        valid_local_ids_cuda[idx] = 0;
    }
    const float3 & pt_w = pts_cuda[idx];
    float3 pt_c;
    pt_c.x = T_cuda[0]*pt_w.x + T_cuda[1]*pt_w.y + T_cuda[2] *pt_w.z + T_cuda[3];
    pt_c.y = T_cuda[4]*pt_w.x + T_cuda[5]*pt_w.y + T_cuda[6] *pt_w.z + T_cuda[7];
    pt_c.z = T_cuda[8]*pt_w.x + T_cuda[9]*pt_w.y + T_cuda[10]*pt_w.z + T_cuda[11];
    if(pt_c.z<0.1)
        return;
    float u,v;
    u = (K_cuda[0]*pt_c.x + K_cuda[1]*pt_c.z)/pt_c.z;
    v = (K_cuda[2]*pt_c.y + K_cuda[3]*pt_c.z)/pt_c.z;
    if(u<0 || v<0 || u>width-1 || v>height-1)
        return;
    int center = (int)((int)v*width + (int)u);
    if(opertator_flag==0)
    {
        valid_local_ids_cuda[idx] = 1;
    }

    fatomicMin(&depth_cuda[center], pt_c.z);
    // depth_cuda[center] = pt_c.z;
}

void proj_pts2depth_with_attr(float3 * pts, int * valid_local_ids, int num_pts, float * K, float * T, float * depth, int width, int height, int opertator_flag)
{
    //* opertator_flag 决定操作方式
    //* 0 表示确定那些点能投影上去
	int threads = 512;
    int blocks = num_pts/threads + ((num_pts % threads)?1:0);
    float3 * pts_cuda;
    float * depth_cuda;
	float * K_cuda, *T_cuda;
    int * valid_local_ids_cuda;
    cudaMalloc((void**)&pts_cuda, sizeof(float3)*num_pts);
    cudaMalloc((void**)&depth_cuda, sizeof(float)*width*height);
    cudaMalloc((void**)&K_cuda, sizeof(float)*4);
    cudaMalloc((void**)&T_cuda, sizeof(float)*12);
    cudaMalloc((void**)&valid_local_ids_cuda, sizeof(int)*num_pts);
    CheckCUDAError("cudaMalloc");
	// printf("num_pts:%d, width:%d, height:%d\n", num_pts, width, height);
    
	cudaMemcpy(pts_cuda, pts, sizeof(float)*num_pts, cudaMemcpyHostToDevice);
    CheckCUDAError("pts_cuda");
    cudaMemcpy(depth_cuda, depth, sizeof(float)*width*height, cudaMemcpyHostToDevice);
    // CheckCUDAError("depth_cuda");
    cudaMemcpy(K_cuda, K, sizeof(float)*4, cudaMemcpyHostToDevice);
    // CheckCUDAError("K_cuda");
    cudaMemcpy(T_cuda, T, sizeof(float)*12, cudaMemcpyHostToDevice);
	cudaMemcpy(valid_local_ids_cuda, valid_local_ids, sizeof(int)*num_pts, cudaMemcpyHostToDevice);
    CheckCUDAError("cudaMemcpyHostToDevice");
    
    proj_pts2depth_with_attr_kernel<<<blocks, threads>>>(pts_cuda, valid_local_ids_cuda, num_pts, K_cuda, T_cuda, depth_cuda, width, height, opertator_flag);
    cudaThreadSynchronize();
    CheckCUDAError("cal_uv_cuda");

    cudaMemcpy(depth, depth_cuda, sizeof(float)*width*height, cudaMemcpyDeviceToHost);
    cudaMemcpy(valid_local_ids, valid_local_ids_cuda, sizeof(int)*num_pts, cudaMemcpyDeviceToHost);
    CheckCUDAError("cudaMemcpyDeviceToHost");

    cudaFree(pts_cuda);
    cudaFree(depth_cuda);
    cudaFree(K_cuda);
    cudaFree(T_cuda);
    cudaFree(valid_local_ids_cuda);
    CheckCUDAError("cudaFree");
    // printf("CUDA cudaFree end\n");
}

__device__ void solve_Ax(int N, float a[3][4])
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

__device__ void fit_plane(const float3 * selected_pts, int N, float M[3])
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
    solve_Ax(3, c);
    for (int i = 0; i < 3; i++)
    {
        M[i] = c[i][3];
    }
}

__global__ void bilinear_interplote_depth_comp_kernel(float * depth_cuda, curandState* devStates, float * K_cuda, int width, int height)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx >= width*height)
        return;
    int u = idx%width;
    int v = idx/width;
    float cur_depth = depth_cuda[idx];
    const int num_pt = 1000;
    const int ransac_n = 6;
    int num_iterations = 100;
    int radius = 50;
    float ths = 0.01;
    float M[3];
    float3 selected_pts[num_pt];
    float dist[num_pt];
    int selected_ids[ransac_n];
    float3 fit_pts[ransac_n];
    int num_in=0;
    int max_inlies_num = 0;
    int new_u, new_v;
    float fx=K_cuda[0], cx=K_cuda[1], fy=K_cuda[2], cy=K_cuda[3];
    float best_param[3];
    bool debug = (u==331 && v==230);
    int step=2;
    float min_d=100, max_d=-1.0;
    for(int i=0; i<radius; i+=step)
    {
        for(int j=0; j<radius; j+=step)
        {
            if(num_in>=num_pt)
                break;
            new_u = u-i;
            new_v = v-j;
            if(new_u>=0 && new_v>=0 && new_u<width && new_v<height)
            {
                float new_depth = depth_cuda[new_v*width + new_u];
                if(new_depth>0.1)
                {
                    float3 pt;
                    pt.x = (new_u - cx)/fx * new_depth;
                    pt.y = (new_v - cy)/fy * new_depth;
                    pt.z = new_depth;
                    selected_pts[num_in] = pt;
                    num_in++;
                    min_d = min_d>new_depth?new_depth:min_d;
                    max_d = max_d<new_depth?new_depth:max_d;
                }
            }
            if(num_in>=num_pt)
                break;
            new_u = u-i;
            new_v = v+j;
            if(new_u>=0 && new_v>=0 && new_u<width && new_v<height)
            {
                float new_depth = depth_cuda[new_v*width + new_u];
                if(new_depth>0.1)
                {
                    float3 pt;
                    pt.x = (new_u - cx)/fx * new_depth;
                    pt.y = (new_v - cy)/fy * new_depth;
                    pt.z = new_depth;
                    selected_pts[num_in] = pt;
                    num_in++;
                    min_d = min_d>new_depth?new_depth:min_d;
                    max_d = max_d<new_depth?new_depth:max_d;
                }
            }
            if(num_in>=num_pt)
                break;
            new_u = u+i;
            new_v = v-j;
            if(new_u>=0 && new_v>=0 && new_u<width && new_v<height)
            {
                float new_depth = depth_cuda[new_v*width + new_u];
                if(new_depth>0.1)
                {
                    float3 pt;
                    pt.x = (new_u - cx)/fx * new_depth;
                    pt.y = (new_v - cy)/fy * new_depth;
                    pt.z = new_depth;
                    selected_pts[num_in] = pt;
                    num_in++;
                    min_d = min_d>new_depth?new_depth:min_d;
                    max_d = max_d<new_depth?new_depth:max_d;
                }
            }
            if(num_in>=num_pt)
                break;
            new_u = u+i;
            new_v = v+j;
            if(new_u>=0 && new_v>=0 && new_u<width && new_v<height)
            {
                float new_depth = depth_cuda[new_v*width + new_u];
                if(new_depth>0.1)
                {
                    float3 pt;
                    pt.x = (new_u - cx)/fx * new_depth;
                    pt.y = (new_v - cy)/fy * new_depth;
                    pt.z = new_depth;
                    selected_pts[num_in] = pt;
                    num_in++;
                    min_d = min_d>new_depth?new_depth:min_d;
                    max_d = max_d<new_depth?new_depth:max_d;
                }
            }
        }
    }
    if(debug)
        printf("min_d: %f, max_d: %f\n", min_d, max_d);

    if(num_in<3)
        return;
    for(int k=0; k<num_iterations; k++)
    {
        //* 生成随机索引
        //* 点选择
        int selected_pts_id=0;
        for(int i=0; i<ransac_n; i++)
        {
            int sel_id = generate(devStates, idx) * num_in;
            fit_pts[i] = selected_pts[sel_id];
        }
        //* 平面拟合
        float M[3];
        fit_plane(fit_pts, ransac_n, M);
        int step = 1;
        int inliers_num=0;
        float A = M[1];
        float B = M[2];
        float C = -1;
        float D = M[0];
        float norm_n = sqrt(A*A + B*B + C*C);
        for(int i=0; i<num_in; i++)
        {
            float d = fabs(A*selected_pts[i].x + B*selected_pts[i].y + C*selected_pts[i].z + D)/norm_n;
            if(d<ths)
            {
                inliers_num++;
            }
        }
        if(inliers_num>max_inlies_num)
        {
            max_inlies_num = inliers_num;
            best_param[0] = M[0];
            best_param[1] = M[1];
            best_param[2] = M[2];
            if(debug)
            {
                printf("k: %d, max_inlies_num: %d\n", k, max_inlies_num);
            }
        }
    }
    float A = best_param[1];
    float B = best_param[2];
    float C = -1;
    float D = best_param[0];
    float norm_n = sqrt(A*A + B*B + C*C);
    float4 plane_param;
    plane_param.x = A/D; plane_param.y = B/D; plane_param.z = C/D; plane_param.w = 1;
    for(int i=0; i<num_in; i++)
    {
        dist[i] = fabs(A*selected_pts[i].x + B*selected_pts[i].y + C*selected_pts[i].z + D)/norm_n;
    }
    float Hessien[3][4];
    for(int iter_num=0; iter_num<10; iter_num++)
    {
        for(int i=0;i<3;i++)
        {
            Hessien[i][0] = Hessien[i][1] = Hessien[i][2] = Hessien[i][3] = 0.0;
        }
        int inliers_num=0;
        for (int i = 0; i < num_in; i+=1)
        {
            if(dist[i]<2*ths)
            {
                const float3 &  pt = selected_pts[i];
                float r = plane_param.x*pt.x + plane_param.y*pt.y + plane_param.z*pt.z + 1;
                Hessien[0][0] += pt.x*pt.x; Hessien[0][1] += pt.x*pt.y; Hessien[0][2] += pt.x*pt.z; Hessien[0][3] -= pt.x * r;
                Hessien[1][0] += pt.y*pt.x; Hessien[1][1] += pt.y*pt.y; Hessien[1][2] += pt.y*pt.z; Hessien[1][3] -= pt.y * r;
                Hessien[2][0] += pt.z*pt.x; Hessien[2][1] += pt.z*pt.y; Hessien[2][2] += pt.z*pt.z; Hessien[2][3] -= pt.z * r;
                inliers_num++;
            }
        }
        if(debug)
        {
            printf("iter_num: %d, inliers_num: %d\n", iter_num, inliers_num);
        }
        if(inliers_num<3)
            break;
        solve_Ax(3, Hessien);
        plane_param.x += Hessien[0][3];
        plane_param.y += Hessien[1][3];
        plane_param.z += Hessien[2][3];
        plane_param.w = 1;
        // for(int i=0; i<num_in; i++)
        // {
        //     dist[i] = fabs(A*selected_pts[i] + B*selected_pts[i] + C*selected_pts[i] + D)/norm_n;
        //     if(dist[i]<ths)
        //     {
        //         inliers_num++;
        //     }
        // }
    }
    float fit_depth = -1/(plane_param.z + plane_param.x*(u-cx)/fx + plane_param.y*(v-cy)/fy);
    if(cur_depth<0.1 && fit_depth>0.1 && fit_depth<100)
    {
        depth_cuda[idx] = fit_depth;
    }
    if(cur_depth>0.1 && fabs(fit_depth-cur_depth)<0.2)
    {
        depth_cuda[idx] = fit_depth;
    }
}

void bilinear_interplote_depth_comp(float * depth, float * K, int width, int height)
{
    int threads = 512;
    int num_pts = width*height;
    int blocks = num_pts/threads + ((num_pts % threads)?1:0);
    float * depth_cuda, * K_cuda;
    cudaMalloc((void**)&depth_cuda, sizeof(float)*width*height);
    cudaMalloc((void**)&K_cuda, sizeof(float)*4);
    CheckCUDAError("cudaMalloc");
	// printf("num_pts:%d, width:%d, height:%d\n", num_pts, width, height);
    
    cudaMemcpy(depth_cuda, depth, sizeof(float)*width*height, cudaMemcpyHostToDevice);
    CheckCUDAError("depth_cuda");
    cudaMemcpy(K_cuda, K, sizeof(float)*4, cudaMemcpyHostToDevice);
    CheckCUDAError("K_cuda");
    
    //* 设置随机数
    curandState* devStates;
	cudaMalloc(&devStates, width*height * sizeof(curandState));
    long clock_for_rand = clock();
    setup_kernel<<<blocks, threads>>>(width, height, devStates, clock_for_rand);
    cudaThreadSynchronize();
    bilinear_interplote_depth_comp_kernel<<<blocks, threads>>>(depth_cuda, devStates, K_cuda, width, height);
    cudaThreadSynchronize();
    CheckCUDAError("proj_pts2depth_kernel");

    cudaMemcpy(depth, depth_cuda, sizeof(float)*width*height, cudaMemcpyDeviceToHost);
    CheckCUDAError("cudaMemcpyDeviceToHost");

    cudaFree(depth_cuda);
    cudaFree(K_cuda);
    CheckCUDAError("cudaFree");
}

__global__ void point_to_plane_dist_kernel(int n_pts, const float3 *ref_pts_cuda, float *dist_cuda, float4 plane_param)
{
    const int center = blockIdx.x * blockDim.x + threadIdx.x;
    if (center >= n_pts || center < 0)
    {
        return;
    }
    float norm_n = sqrt(plane_param.x * plane_param.x + plane_param.y * plane_param.y + plane_param.z * plane_param.z);
    float dist = fabs(plane_param.x * ref_pts_cuda[center].x + plane_param.y * ref_pts_cuda[center].y + plane_param.z * ref_pts_cuda[center].z + plane_param.w) / norm_n;
    dist_cuda[center] = dist;
}

void point_to_plane_dist_cuda(int n_pts, const float3 *ref_pts_cuda, float *dist_cuda, float4 plane_param)
{
    // Matrix addition kernel launch from host code
    dim3 threadsPerBlock(512);
    dim3 numBlocks((n_pts + threadsPerBlock.x - 1) / threadsPerBlock.x);
    point_to_plane_dist_kernel<<<numBlocks, threadsPerBlock>>>(n_pts, ref_pts_cuda, dist_cuda, plane_param);
    cudaThreadSynchronize();
    CheckCUDAError("point_to_plane_dist_kernel");
}

__global__ void point_to_plane_sign_dist_kernel(int n_pts, const float3 *ref_pts_cuda, float *dist_cuda, float4 plane_param)
{
    const int center = blockIdx.x * blockDim.x + threadIdx.x;
    if (center >= n_pts || center < 0)
    {
        return;
    }
    float norm_n = sqrt(plane_param.x * plane_param.x + plane_param.y * plane_param.y + plane_param.z * plane_param.z);
    float dist = (plane_param.x * ref_pts_cuda[center].x + plane_param.y * ref_pts_cuda[center].y + plane_param.z * ref_pts_cuda[center].z + plane_param.w) / norm_n;
    dist_cuda[center] = dist;
}

void point_to_plane_sign_dist_cuda(int n_pts, const float3 *ref_pts_cuda, float *dist_cuda, float4 plane_param)
{
    // Matrix addition kernel launch from host code
    dim3 threadsPerBlock(512);
    dim3 numBlocks((n_pts + threadsPerBlock.x - 1) / threadsPerBlock.x);
    point_to_plane_sign_dist_kernel<<<numBlocks, threadsPerBlock>>>(n_pts, ref_pts_cuda, dist_cuda, plane_param);
    cudaThreadSynchronize();
    CheckCUDAError("point_to_plane_dist_kernel");
}

__global__ void depth_remove_outlier_kernel(int width, int height, const float *depth_cuda, float *depth_new_cuda)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx >= width*height)
        return;
    int u = idx%width;
    int v = idx/width;
    const int2 p = make_int2(u, v);
    if (p.x >= width || p.y >= height || p.x < 0 || p.y < 0)
    {
        return;
    }
    const int center = p.y * width + p.x;
    float cur_depth = depth_cuda[center];
    bool debug = false; //* (p.x == 1122 && p.y == 902)
    //* 采用放射搜索策略
    int num_dirs = 16;      //* 搜索方向数量
    float angles[50];       //* 方向角度
    float2 dirs[50];        //* 方向向量
    bool active_flag[50];   //* 是否结束
    float search_dist[50];  //* 沿射线搜索距离
    int2 stop_offset[50];   //* 终止坐标偏置
    float stop_v[50];       //* 终止值, 0表示可以继续搜索，0-1 表示终止值，2表示越界
    float stop_ths = 0.1;     //* 终止阈值
    float max_dist = 100.0; //* 最远搜索100个像素
    angles[0] = 0.0;
    dirs[0].x = 1.0;
    dirs[0].y = 0.0;
    active_flag[0] = true;
    float delta_angle = 360.0 / (float)num_dirs;
    for (int i = 1; i < num_dirs; i++)
    {
        angles[i] = angles[i - 1] + delta_angle;
        dirs[i].x = cos(angles[i] / 180 * M_PI);
        dirs[i].y = sin(angles[i] / 180 * M_PI);
        active_flag[i] = true;
        // printf("angle[%d]: %f (%f, %f)\t \n", i, angles[i], dirs[i].x, dirs[i].y);
    }
    if (debug)
    {
        printf("cur_depth: %f \n", cur_depth);
    }
    for (int i = 0; i < num_dirs; i++)
    {
        float radius = 1;
        while (radius < max_dist)
        {
            int off_x = (int)(dirs[i].x * radius);
            int off_y = (int)(dirs[i].y * radius);
            int2 new_p = make_int2(p.x + off_x, p.y + off_y);
            if (i == 3 && debug && 0)
            {
                printf("radius: %f, new_p: (%d, %d)\n", radius, new_p.x, new_p.y);
            }
            if (new_p.x < width && new_p.y < height && new_p.x >= 0 && new_p.y >= 0)
            {
                float new_depth = depth_cuda[new_p.y * width + new_p.x];
                float depth_diff = cur_depth - new_depth;
                if (depth_diff > stop_ths && new_depth > 0) //* 达到阈值而终止
                {
                    search_dist[i] = radius;
                    stop_offset[i].x = off_x;
                    stop_offset[i].y = off_y;
                    active_flag[i] = false;
                    stop_v[i] = new_depth;
                    break;
                }
            }
            else //* 越界而终止
            {
                int off_x = (int)(dirs[i].x * radius);
                int off_y = (int)(dirs[i].y * radius);
                search_dist[i] = radius;
                stop_offset[i].x = off_x;
                stop_offset[i].y = off_y;
                active_flag[i] = false;
                stop_v[i] = 2.0;
                break;
            }
            if (radius + 1 >= max_dist && active_flag[i])
            {
                int off_x = (int)(dirs[i].x * radius);
                int off_y = (int)(dirs[i].y * radius);
                search_dist[i] = radius;
                stop_offset[i].x = off_x;
                stop_offset[i].y = off_y;
                active_flag[i] = true; //* 还可以继续延伸
            }
            radius++;
        }
        if (debug)
        {
            // printf("angle[%d]: %f (%f, %f)\t \n", i, angles[i], dirs[i].x, dirs[i].y);
            printf("idx: %d, active_flag: %d, search_dist: %f, stop_pos: (%d, %d), stop_v: %f\n", i, active_flag[i], search_dist[i], p.x + stop_offset[i].x, p.y + stop_offset[i].y, stop_v[i]);
        }
    }
    int num_inactive = 0;
    for (int i = 0; i < num_dirs; i++)
    {
        if (!active_flag[i])
        {
            num_inactive++;
        }
    }
    if (num_inactive >= num_dirs - 3)
    {
        depth_new_cuda[center] = 0.0;
    }
}

void depth_remove_outlier_cuda(float * depth_old, float * depth_new, int width, int height)
{
    //* opertator_flag 决定操作方式
    //* 0 表示确定那些点能投影上去
	int threads = 512;
    int num_pts = width*height;
    int blocks = num_pts/threads + ((num_pts % threads)?1:0);
    float *depth_old_cuda, *depth_new_cuda;
    cudaMalloc((void**)&depth_old_cuda, sizeof(float)*num_pts);
    cudaMalloc((void**)&depth_new_cuda, sizeof(float)*num_pts);
    CheckCUDAError("cudaMalloc");
	// printf("num_pts:%d, width:%d, height:%d\n", num_pts, width, height);
    
	cudaMemcpy(depth_old_cuda, depth_old, sizeof(float)*num_pts, cudaMemcpyHostToDevice);
    cudaMemcpy(depth_new_cuda, depth_old, sizeof(float)*num_pts, cudaMemcpyHostToDevice);
    CheckCUDAError("cudaMemcpyHostToDevice");
    
    depth_remove_outlier_kernel<<<blocks, threads>>>(width, height, depth_old_cuda,depth_new_cuda);
    cudaThreadSynchronize();
    CheckCUDAError("depth_remove_outlier_kernel");

    cudaMemcpy(depth_new, depth_new_cuda, sizeof(float)*num_pts, cudaMemcpyDeviceToHost);
    CheckCUDAError("cudaMemcpyDeviceToHost");

    cudaFree(depth_old_cuda);
    cudaFree(depth_new_cuda);
    CheckCUDAError("cudaFree");
    // printf("CUDA cudaFree end\n");
}

__global__ void depth_comp_with_attr_kernel(float * depth_old_cuda, float * depth_new_cuda, float3 * plane_params_cuda, float * K_cuda, int width, int height)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx >= width*height)
        return;
    const float3 & plane_param = plane_params_cuda[idx];
    float cur_depth = depth_old_cuda[idx];
    float fx=K_cuda[0], cx=K_cuda[1], fy=K_cuda[2], cy=K_cuda[3];
    int u = idx%width;
    int v = idx/width;
    const int2 p = make_int2(u, v);
    int radius = 5;
    int step = 1;
    int u_new, new_v;
    bool debug = false;
    //* 采用放射搜索策略
    const int num_dirs = 16;      //* 搜索方向数量
    float angles[num_dirs];       //* 方向角度
    float2 dirs[num_dirs];        //* 方向向量
    bool valid_flag[num_dirs];   //* 是否结束
    float search_dist[num_dirs];  //* 沿射线搜索距离
    int2 stop_offset[num_dirs];   //* 终止坐标偏置
    float3 stop_v[num_dirs];       //* 终止值, 0表示可以继续搜索，0-1 表示终止值，2表示越界
    float stop_ths = 1;     //* 终止阈值
    float max_dist = 10.0; //* 最远搜索100个像素
    angles[0] = 0.0;
    dirs[0].x = 1.0;
    dirs[0].y = 0.0;
    valid_flag[0] = false;
    float delta_angle = 360.0 / (float)num_dirs;
    for (int i = 1; i < num_dirs; i++)
    {
        angles[i] = angles[i - 1] + delta_angle;
        dirs[i].x = cos(angles[i] / 180 * M_PI);
        dirs[i].y = sin(angles[i] / 180 * M_PI);
        valid_flag[i] = false;
        // printf("angle[%d]: %f (%f, %f)\t \n", i, angles[i], dirs[i].x, dirs[i].y);
    }
    if (debug)
    {
        printf("cur_depth: %f \n", cur_depth);
    }
    for (int i = 0; i < num_dirs; i++)
    {
        float radius = 1;
        while (radius < max_dist)
        {
            int off_x = (int)(dirs[i].x * radius);
            int off_y = (int)(dirs[i].y * radius);
            int2 new_p = make_int2(p.x + off_x, p.y + off_y);
            if (new_p.x < width && new_p.y < height && new_p.x >= 0 && new_p.y >= 0)
            {
                float new_depth = depth_old_cuda[new_p.y * width + new_p.x];
                float3 plane_param_new = plane_params_cuda[new_p.y * width + new_p.x];
                if (plane_param_new.x>0 || plane_param_new.y>0 || plane_param_new.z>0) //* 有
                {
                    search_dist[i] = radius;
                    stop_offset[i].x = off_x;
                    stop_offset[i].y = off_y;
                    valid_flag[i] = true;
                    stop_v[i] = plane_param_new;
                    break;
                }
            }
            else //* 越界而终止
            {
                int off_x = (int)(dirs[i].x * radius);
                int off_y = (int)(dirs[i].y * radius);
                search_dist[i] = radius;
                stop_offset[i].x = off_x;
                stop_offset[i].y = off_y;
                valid_flag[i] = false;
                break;
            }
            radius++;
        }
        if (debug)
        {
            // printf("angle[%d]: %f (%f, %f)\t \n", i, angles[i], dirs[i].x, dirs[i].y);
            printf("idx: %d, valid_flag: %d, search_dist: %f, stop_pos: (%d, %d)%f\n", i, valid_flag[i], search_dist[i], p.x + stop_offset[i].x, p.y + stop_offset[i].y);
        }
    }
    int num_invalid = 0;
    for (int i = 0; i < num_dirs; i++)
    {
        if (valid_flag[i])
        {
            num_invalid++;
        }
    }
    if(num_invalid==0)
    {
        return;
    }
    float3 candidate_plane_params[num_dirs];
    int candidate_plane_vote[num_dirs];
    //* 初始化
    for(int i=0; i<num_dirs; i++)
    {
        candidate_plane_params[i].x = 0;
        candidate_plane_params[i].y = 0;
        candidate_plane_params[i].z = 0;
        candidate_plane_vote[i] = 0;
    }
    //*
    for(int i=0; i<num_dirs; i++)
    {
        if (valid_flag[i])
        {
            float3 param = stop_v[i];
            for(int j=0; j<num_dirs; j++)
            {
                float3 cand = candidate_plane_params[j];
                if(cand.x==0 && cand.y==0 && cand.z==0)
                {
                    candidate_plane_params[j] = param;
                    candidate_plane_vote[j]++;
                }
                else
                {
                    if(param.x==cand.x && param.y==cand.y && param.z==cand.z)
                    {
                        candidate_plane_vote[j]++;
                    }
                }
            }
        }
    }
    float3 best_param;
    int max_vote=0;
    for(int i=0; i<num_dirs; i++)
    {
        if(candidate_plane_vote[i]>max_vote)
        {
            best_param = candidate_plane_params[i];
            max_vote = candidate_plane_vote[i];
        }
    }
    float z = -1.0 / (best_param.x * (u-cx)/fx + best_param.y * (v-cy)/fy + best_param.z);
    depth_new_cuda[idx] = z;

}

void depth_comp_with_attr(float * depth_old, float * depth_new, float3 * plane_params, float * K, int width, int height)
{
    //* opertator_flag 决定操作方式
    //* 0 表示确定那些点能投影上去
	int threads = 512;
    int num_pts = width*height;
    int blocks = num_pts/threads + ((num_pts % threads)?1:0);
    float *depth_old_cuda, *depth_new_cuda, *K_cuda;
    float3 *plane_params_cuda;
    cudaMalloc((void**)&depth_old_cuda, sizeof(float)*num_pts);
    cudaMalloc((void**)&depth_new_cuda, sizeof(float)*num_pts);
    cudaMalloc((void**)&K_cuda, sizeof(float)*4);
    cudaMalloc((void**)&plane_params_cuda, sizeof(float3)*num_pts);
    CheckCUDAError("cudaMalloc");
	// printf("num_pts:%d, width:%d, height:%d\n", num_pts, width, height);
    
	cudaMemcpy(depth_old_cuda, depth_old, sizeof(float)*num_pts, cudaMemcpyHostToDevice);
    cudaMemcpy(depth_new_cuda, depth_old, sizeof(float)*num_pts, cudaMemcpyHostToDevice);
    cudaMemcpy(K_cuda, K, sizeof(float)*4, cudaMemcpyHostToDevice);
	cudaMemcpy(plane_params_cuda, plane_params, sizeof(float3)*num_pts, cudaMemcpyHostToDevice);
    CheckCUDAError("cudaMemcpyHostToDevice");
    
    depth_remove_outlier_kernel<<<blocks, threads>>>(width, height, depth_old_cuda,depth_new_cuda);
    cudaThreadSynchronize();
    depth_comp_with_attr_kernel<<<blocks, threads>>>(depth_old_cuda, depth_new_cuda, plane_params_cuda, K_cuda, width, height);
    cudaThreadSynchronize();
    CheckCUDAError("depth_comp_with_attr_kernel");

    cudaMemcpy(depth_new, depth_new_cuda, sizeof(float)*num_pts, cudaMemcpyDeviceToHost);
    CheckCUDAError("cudaMemcpyDeviceToHost");

    cudaFree(depth_old_cuda);
    cudaFree(depth_new_cuda);
    cudaFree(K_cuda);
    cudaFree(plane_params_cuda);
    CheckCUDAError("cudaFree");
    // printf("CUDA cudaFree end\n");
}
