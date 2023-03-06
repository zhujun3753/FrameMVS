
#include "simple_tools.h"

// const int smoothing_grid[32][2] =
// {
//     {-1,-1}, {,}, {,}, {,}, {,}, {,}, {,}, {,},
//     {,}, {,}, {,}, {,}, {,}, {,}, {,}, {,},
//     {,}, {,}, {,}, {,}, {,}, {,}, {,}, {,},
//     {,}, {,}, {,}, {,}, {,}, {,}, {,}, {,},
// };
// //返回整数
// __device__ unsigned int    curand (curandState_t *state)
// //返回包括[0,1]之间的伪随机数单精度浮点序列，包括0和1
// __device__ float     curand_uniform (curandState_t *state)
// //返回均值为0，标准差为1的伪随机数单精度浮点序列
// __device__ float     curand_normal (curandState_t *state)
// //双精度
// __device__ double     curand_uniform_double (curandState_t *state)
// //双精度
// __device__ double     curand_normal_double (curandState_t *state)
__global__ void kernel_set_random(int width, int height, curandState *rand_states_cuda, long clock_for_rand)
{
    const int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (p.x >= width || p.y >= height || p.x < 0 || p.y < 0)
    {
        return;
    }
    const int center = p.y * width + p.x;
    curand_init(clock_for_rand, p.y, p.x, &rand_states_cuda[center]);
}

//* 从小到大排序
__device__ void sort_small(float *d, const int n)
{
    int j;
    for (int i = 1; i < n; i++)
    {
        float tmp = d[i];
        for (j = i; j >= 1 && tmp < d[j - 1]; j--)
            d[j] = d[j - 1];
        d[j] = tmp;
    }
}

//* 从大到小排序
__device__ void sort_large(float *d, const int n)
{
    int j;
    for (int i = 1; i < n; i++)
    {
        float tmp = d[i];
        for (j = i; j >= 1 && tmp > d[j - 1]; j--)
            d[j] = d[j - 1];
        d[j] = tmp;
    }
}

//* 这个算法除了可以计算Ax=b ，还可以求逆，只需要令b为单位向量即可，求出几个x再合并为矩阵
__device__ void column_principle_gauss(int N, float a[3][4])
{
    int k = 0, i = 0, r = 0, j = 0;
    float t;
    //* 行变换求对角矩阵，NxN的矩阵只需要对N-1列处理即可，最后一列不处理
    for (k = 0; k < N - 1; k++)
    {
        //* 寻找当前列对角元及其以下元素中的绝对值最大值，避免用0
        r = k;
        for (i = k + 1; i < N; i++)
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

__device__ void Least_squares(float3 *v_Point, int N, float M[3])
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
        c[0][1] = c[0][1] + v_Point[i].x;
        c[0][2] = c[0][2] + v_Point[i].y;
        c[0][3] = c[0][3] + v_Point[i].z;
        c[1][1] = c[1][1] + v_Point[i].x * v_Point[i].x;
        c[1][2] = c[1][2] + v_Point[i].x * v_Point[i].y;
        c[1][3] = c[1][3] + v_Point[i].x * v_Point[i].z;
        c[2][2] = c[2][2] + v_Point[i].y * v_Point[i].y;
        c[2][3] = c[2][3] + v_Point[i].y * v_Point[i].z;
    }
    c[1][0] = c[0][1];
    c[2][0] = c[0][2];
    c[2][1] = c[1][2];

    column_principle_gauss(3, c);

    for (int i = 0; i < 3; i++)
    {
        M[i] = c[i][3];
    }
}

__device__ void test_fit()
{
    float test[24][3] =
        {
            {-735, -312, 6},
            {-252, -298, 6},
            {290, -264, 7},
            {657, -252, 7},
            {-753, 82, 15},
            {-92, 27, 15},
            {656, 31, 14},
            {-726, 389, 24},
            {-27, 355, 25},
            {652, 413, 26},
            {-922, -306, 4},
            {-915, -114, 10},
            {-791, 87, 14},
            {-729, 390, 22},
            {24, 422, 25},
            {1, 273, 21},
            {10, 168, 18},
            {-2, 3, 13},
            {-16, -130, 11},
            {-22, -292, 6},
            {615, -342, 5},
            {724, -137, 10},
            {712, 64, 15},
            {728, 359, 21},
        };
    float3 points[24];
    int N = 24;
    for (int i = 0; i < N; i++)
    {

        points[i].x = test[i][0];
        points[i].y = test[i][1];
        // points[i].z = test[i][2];
        points[i].z = 1.04 + 1.02 * points[i].x + 1.03 * points[i].y;
    }
    float M[3];
    Least_squares(points, N, M);
    for (int i = 0; i < 3; i++)
    {
        printf("M%d = %lf\n", i, M[i]);
    }
    float mean_err, max_err;
    max_err = 0.0;
    mean_err = 0.0;
    for (int i = 0; i < N; i++)
    {

        float new_z = M[0] + M[1] * points[i].x + M[2] * points[i].y;
        float err_tmp = fabs(new_z - points[i].z);
        // printf("new_z: %f, err_tmp: %f\n\n", new_z, err_tmp);
        mean_err += err_tmp;
        if (err_tmp > max_err)
        {
            max_err = err_tmp;
        }
        // printf("mean error: %f, max error: %f\n\n", mean_err, max_err);
    }
    mean_err /= N;
    printf("mean error: %f, max error: %f\n\n", mean_err, max_err);
}

__global__ void depth_interp_comp_kernel(int width, int height, const float *depth_cuda, float *color_smooth_cuda, float3 *normal_cuda, float *depth_new_cuda, curandState *rand_states_cuda)
{
    const int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (p.x >= width || p.y >= height || p.x < 0 || p.y < 0)
    {
        return;
    }
    const int center = p.y * width + p.x;
    float cur_depth = depth_cuda[center];
    float cur_color_smooth = color_smooth_cuda[center];
    float3 cur_normal = normal_cuda[center];
    bool debug = true;

    if (debug && p.x == 559 && p.y == 699)
    {
        //* 采用放射搜索策略
        int num_dirs = 16;      //* 搜索方向数量
        float angles[50];       //* 方向角度
        float2 dirs[50];        //* 方向向量
        bool active_flag[50];   //* 是否结束
        float search_dist[50];  //* 沿射线搜索距离
        int2 stop_offset[50];   //* 终止坐标偏置
        float stop_v[50];       //* 终止值, 0表示可以继续搜索，0-1 表示终止值，2表示越界
        float stop_ths = 0.5;   //* 终止阈值
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
                    float new_color_smooth = color_smooth_cuda[new_p.y * width + new_p.x];
                    float color_smooth_diff = fabs(new_color_smooth - cur_color_smooth);
                    if (color_smooth_diff > stop_ths) //* 达到阈值而终止
                    {
                        search_dist[i] = radius;
                        stop_offset[i].x = off_x;
                        stop_offset[i].y = off_y;
                        active_flag[i] = false;
                        stop_v[i] = new_color_smooth;
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
                printf("angle[%d]: %f (%f, %f)\t \n", i, angles[i], dirs[i].x, dirs[i].y);
                printf("idx: %d, active_flag: %d, search_dist: %f, stop_pos: (%d, %d), stop_v: %f\n", i, active_flag[i], search_dist[i], p.x + stop_offset[i].x, p.y + stop_offset[i].y, stop_v[i]);
            }
        }

        //* 采用菱形搜索策略
        int last_radius = 3;     //* 初始半径
        int max_search_num = 10; //* 最大搜索次数
        int cur_radius = 3;
        int2 pts_offset[4][100] = {10000}; //* 选择的偏置
        float max_color_smooth_diff = 0.0;
        float min_color_smooth_diff = 2.0;
        int2 min_color_smooth_diff_offset, max_color_smooth_diff_offset;
        cur_radius = last_radius;
        for (int idx = 0; idx < max_search_num; idx++)
        {
            continue;
            //* 根据搜索的结果调整搜索半径
            for (int off_x = -cur_radius; off_x <= cur_radius; off_x++)
            {
                // float rx = curand_normal(&rand_states_cuda[center]);
                // float ry = curand_normal(&rand_states_cuda[center]);
                // // printf("idx: %d, sam_i: %d, rx: %f, ry: %f\n", idx, sam_i, rx, ry);
                // pts_offset[idx][sam_i].x = (int)(rx*radius);
                // pts_offset[idx][sam_i].y = (int)(ry*radius);
                // printf("idx: %d, sam_i: %d, offset_x: %d, \toffset_y: %d, cs: %f\n", idx, sam_i, pts_offset[idx][sam_i].x, pts_offset[idx][sam_i].y);
                // int2 new_p = make_int2(p.x * blockDim.x + , p.y * blockDim.y + threadIdx.y);
                int off_y = cur_radius - (int)fabs(off_x);
                int2 new_p = make_int2(p.x + off_x, p.y + off_y);
                if (new_p.x < width && new_p.y < height && new_p.x >= 0 && new_p.y >= 0)
                {
                    float new_color_smooth = color_smooth_cuda[new_p.y * width + new_p.x];
                    float color_smooth_diff = fabs(new_color_smooth - cur_color_smooth);
                    if (min_color_smooth_diff > color_smooth_diff)
                    {
                        min_color_smooth_diff = color_smooth_diff;
                        min_color_smooth_diff_offset.x = off_x;
                        min_color_smooth_diff_offset.y = off_y;
                    }
                    if (max_color_smooth_diff < color_smooth_diff)
                    {
                        max_color_smooth_diff = color_smooth_diff;
                        max_color_smooth_diff_offset.x = off_x;
                        max_color_smooth_diff_offset.y = off_y;
                        // printf("new_p: (%d, %d)\n", new_p.x, new_p.y);
                        // printf("p: (%d, %d)\n", p.x, p.y);
                        // printf("offset: (%d, %d)\n", off_x, off_y);
                        // printf("new_color_smooth: %f\n", new_color_smooth);
                    }
                }
                off_y = 0 - off_y;
                new_p = make_int2(p.x + off_x, p.y + off_y);
                if (new_p.x < width && new_p.y < height && new_p.x >= 0 && new_p.y >= 0)
                {
                    float new_color_smooth = color_smooth_cuda[new_p.y * width + new_p.x];
                    float color_smooth_diff = fabs(new_color_smooth - cur_color_smooth);
                    if (min_color_smooth_diff > color_smooth_diff)
                    {
                        min_color_smooth_diff = color_smooth_diff;
                        min_color_smooth_diff_offset.x = off_x;
                        min_color_smooth_diff_offset.y = off_y;
                    }
                    if (max_color_smooth_diff < color_smooth_diff)
                    {
                        max_color_smooth_diff = color_smooth_diff;
                        max_color_smooth_diff_offset.x = off_x;
                        max_color_smooth_diff_offset.y = off_y;
                    }
                }
            }
            if (min_color_smooth_diff < 1e-2)
            {
                cur_radius *= 2;
            }
            printf("idx: %d, cur_radius: %d\n", idx, cur_radius);
            printf("min_color_smooth_diff: %f, max_color_smooth_diff: %f\n", min_color_smooth_diff, max_color_smooth_diff);
            printf("min_color_smooth_diff_offset: (%d, %d)\n", min_color_smooth_diff_offset.x, min_color_smooth_diff_offset.y);
            printf("max_color_smooth_diff_offset: (%d, %d)\n", max_color_smooth_diff_offset.x, max_color_smooth_diff_offset.y);
        }
    }
}

__global__ void depth_filter_kernel(int width, int height, const float *depth_cuda, float *depth_new_cuda, curandState *rand_states_cuda)
{
    const int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (p.x >= width || p.y >= height || p.x < 0 || p.y < 0)
    {
        return;
    }
    const int center = p.y * width + p.x;
    float cur_depth = depth_cuda[center];
    // float cur_color_smooth = color_smooth_cuda[center];
    // float3 cur_normal = normal_cuda[center];
    bool debug = false;

    if ((debug && p.x == 1122 && p.y == 902) || !debug)
    {
        //* 采用放射搜索策略
        int num_dirs = 16;      //* 搜索方向数量
        float angles[50];       //* 方向角度
        float2 dirs[50];        //* 方向向量
        bool active_flag[50];   //* 是否结束
        float search_dist[50];  //* 沿射线搜索距离
        int2 stop_offset[50];   //* 终止坐标偏置
        float stop_v[50];       //* 终止值, 0表示可以继续搜索，0-1 表示终止值，2表示越界
        float stop_ths = 1;     //* 终止阈值
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
        return;

        //* 采用菱形搜索策略
        int last_radius = 3;     //* 初始半径
        int max_search_num = 10; //* 最大搜索次数
        int cur_radius = 3;
        int2 pts_offset[4][100] = {10000}; //* 选择的偏置
        float max_depth_diff = 0.0;
        float min_depth_diff = 2.0;
        int2 min_depth_diff_offset, max_depth_diff_offset;
        cur_radius = last_radius;
        for (int idx = 0; idx < max_search_num; idx++)
        {
            continue;
            //* 根据搜索的结果调整搜索半径
            for (int off_x = -cur_radius; off_x <= cur_radius; off_x++)
            {
                // float rx = curand_normal(&rand_states_cuda[center]);
                // float ry = curand_normal(&rand_states_cuda[center]);
                // // printf("idx: %d, sam_i: %d, rx: %f, ry: %f\n", idx, sam_i, rx, ry);
                // pts_offset[idx][sam_i].x = (int)(rx*radius);
                // pts_offset[idx][sam_i].y = (int)(ry*radius);
                // printf("idx: %d, sam_i: %d, offset_x: %d, \toffset_y: %d, cs: %f\n", idx, sam_i, pts_offset[idx][sam_i].x, pts_offset[idx][sam_i].y);
                // int2 new_p = make_int2(p.x * blockDim.x + , p.y * blockDim.y + threadIdx.y);
                int off_y = cur_radius - (int)fabs(off_x);
                int2 new_p = make_int2(p.x + off_x, p.y + off_y);
                if (new_p.x < width && new_p.y < height && new_p.x >= 0 && new_p.y >= 0)
                {
                    float new_depth = depth_cuda[new_p.y * width + new_p.x];
                    float depth_diff = fabs(new_depth - cur_depth);
                    if (min_depth_diff > depth_diff)
                    {
                        min_depth_diff = depth_diff;
                        min_depth_diff_offset.x = off_x;
                        min_depth_diff_offset.y = off_y;
                    }
                    if (max_depth_diff < depth_diff)
                    {
                        max_depth_diff = depth_diff;
                        max_depth_diff_offset.x = off_x;
                        max_depth_diff_offset.y = off_y;
                        // printf("new_p: (%d, %d)\n", new_p.x, new_p.y);
                        // printf("p: (%d, %d)\n", p.x, p.y);
                        // printf("offset: (%d, %d)\n", off_x, off_y);
                        // printf("new_depth: %f\n", new_depth);
                    }
                }
                off_y = 0 - off_y;
                new_p = make_int2(p.x + off_x, p.y + off_y);
                if (new_p.x < width && new_p.y < height && new_p.x >= 0 && new_p.y >= 0)
                {
                    float new_depth = depth_cuda[new_p.y * width + new_p.x];
                    float depth_diff = fabs(new_depth - cur_depth);
                    if (min_depth_diff > depth_diff)
                    {
                        min_depth_diff = depth_diff;
                        min_depth_diff_offset.x = off_x;
                        min_depth_diff_offset.y = off_y;
                    }
                    if (max_depth_diff < depth_diff)
                    {
                        max_depth_diff = depth_diff;
                        max_depth_diff_offset.x = off_x;
                        max_depth_diff_offset.y = off_y;
                    }
                }
            }
            if (min_depth_diff < 1e-2)
            {
                cur_radius *= 2;
            }
            printf("idx: %d, cur_radius: %d\n", idx, cur_radius);
            printf("min_depth_diff: %f, max_depth_diff: %f\n", min_depth_diff, max_depth_diff);
            printf("min_depth_diff_offset: (%d, %d)\n", min_depth_diff_offset.x, min_depth_diff_offset.y);
            printf("max_depth_diff_offset: (%d, %d)\n", max_depth_diff_offset.x, max_depth_diff_offset.y);
        }
    }
}

//* 从某一初始半径开始搜索，每个方向先逐个搜索
__global__ void normal_esti_kernel0(int width, int height, const float *depth_new_cuda, float3 *cam_pts_cuda, float3 *esti_normal_cuda, curandState *rand_states_cuda)
{
    const int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (p.x >= width || p.y >= height || p.x < 0 || p.y < 0)
    {
        return;
    }
    const int center = p.y * width + p.x;
    float cur_depth = depth_new_cuda[center];
    if (cur_depth < 0.1)
    {
        return;
    }
    float3 cur_pt = cam_pts_cuda[center];
    bool debug = true;

    if ((debug && p.x == 927 && p.y == 141) || !debug)
    {
        //* 采用放射搜索策略
        int num_dirs = 16;      //* 搜索方向数量
        float angles[50];       //* 方向角度
        float2 dirs[50];        //* 方向向量
        bool valid_flag[50];    //* 是否有效
        float search_dist[50];  //* 沿射线搜索距离
        int2 stop_offset[50];   //* 终止坐标偏置
        float3 stop_pt[50];     //* 终止值, 0表示可以继续搜索，0-1 表示终止值，2表示越界
        float stop_ths = 10;    //* 终止阈值
        float max_dist = 100.0; //* 最远搜索100个像素

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
            float radius = 10.0;
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
                    float new_depth = depth_new_cuda[new_p.y * width + new_p.x];
                    float depth_diff = fabs(cur_depth - new_depth);
                    if (depth_diff < stop_ths && new_depth > 0) //* 达到阈值而终止
                    {
                        search_dist[i] = radius;
                        stop_offset[i].x = off_x;
                        stop_offset[i].y = off_y;
                        valid_flag[i] = true;
                        stop_pt[i] = cam_pts_cuda[new_p.y * width + new_p.x];
                        break;
                    }
                }
                else //* 越界而终止
                {
                    break;
                }
                radius++;
            }
            if (debug)
            {
                // printf("angle[%d]: %f (%f, %f)\t \n", i, angles[i], dirs[i].x, dirs[i].y);
                // printf("idx: %d, valid_flag: %d, angle: %f, search_dist: %f, stop_pos: (%d, %d), \n", i, valid_flag[i], angles[i], search_dist[i], p.x + stop_offset[i].x, p.y + stop_offset[i].y);
            }
        }
        int num_valid = 0;
        for (int i = 0; i < num_dirs; i++)
        {
            if (valid_flag[i])
            {
                num_valid++;
            }
        }
        if (num_valid >= num_dirs - 3)
        {
            float3 esti_normal;
            esti_normal.x = -cur_pt.x;
            esti_normal.y = -cur_pt.y;
            esti_normal.z = -cur_pt.z;
            esti_normal_cuda[center] = esti_normal;
            //* 相对方向的最小夹角决定曲率
            float max_cos_theta = -2.0;
            for (int i = 0; i < 8; i++)
            {
                if (valid_flag[i] && valid_flag[i + 8])
                {
                    //* 计算夹角
                    float3 dir1, dir2;
                    dir1.x = stop_pt[i].x - cur_pt.x;
                    dir1.y = stop_pt[i].y - cur_pt.y;
                    dir1.z = stop_pt[i].z - cur_pt.z;
                    dir2.x = stop_pt[i + 8].x - cur_pt.x;
                    dir2.y = stop_pt[i + 8].y - cur_pt.y;
                    dir2.z = stop_pt[i + 8].z - cur_pt.z;
                    float dir1_dot_dir2 = dir1.x * dir2.x + dir1.y * dir2.y + dir1.z * dir2.z;
                    float dir1_norm = sqrt(dir1.x * dir1.x + dir1.y * dir1.y + dir1.z * dir1.z);
                    float dir2_norm = sqrt(dir2.x * dir2.x + dir2.y * dir2.y + dir2.z * dir2.z);
                    float cos_theta = dir1_dot_dir2 / (dir1_norm * dir2_norm);
                    if (max_cos_theta < cos_theta)
                    {
                        max_cos_theta = cos_theta;
                    }
                    if (debug)
                    {
                        // printf("angle[%d]: %f (%f, %f)\t \n", i, angles[i], dirs[i].x, dirs[i].y);
                        printf("idx: %d, valid_flag: %d, angle: %f, search_dist: %f, stop_pos: (%d, %d), \n", i, valid_flag[i], angles[i], search_dist[i], p.x + stop_offset[i].x, p.y + stop_offset[i].y);
                        printf("idx: %d, valid_flag: %d, angle: %f, search_dist: %f, stop_pos: (%d, %d), \n", i + 8, valid_flag[i + 8], angles[i + 8], search_dist[i + 8], p.x + stop_offset[i + 8].x, p.y + stop_offset[i + 8].y);
                        printf("idx: %d, angle: %f, cos_theta: %f, theta: %f\n", i, angles[i], cos_theta, acosf(cos_theta) / M_PI * 180);
                        // printf("dir1:(%f, %f, %f), dir2: (%f, %f, %f)\n", dir1.x, dir1.y, dir1.z, dir2.x, dir2.y, dir2.z);
                        // printf("stop_pt[i]:(%f, %f, %f), stop_pt[i+8]: (%f, %f, %f)\n\n", stop_pt[i].x, stop_pt[i].y, stop_pt[i].z, stop_pt[i+8].x, stop_pt[i+8].y, stop_pt[i+8].z);
                    }
                }
            }
            // float H[3][3];
            if (debug)
            {
                // printf("angle[%d]: %f (%f, %f)\t \n", i, angles[i], dirs[i].x, dirs[i].y);
                printf("max_cos_theta: %f \n", max_cos_theta);
            }
        }
        return;
    }
}

//* 先对每两个反方向搜索，寻找这两个方向在一定范围内的夹角最大值，得到多个方向的夹角最大值，再来判断这个点的属性：平面点还是突起轮廓点
//* 先不考虑噪声的影响，单纯计算点属性
__global__ void curve_esti_kernel(int width, int height, const float *depth_new_cuda, float3 *cam_pts_cuda, float *curve_cuda, curandState *rand_states_cuda)
{
    const int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (p.x >= width || p.y >= height || p.x < 0 || p.y < 0)
    {
        return;
    }
    const int center = p.y * width + p.x;
    float cur_depth = depth_new_cuda[center];
    if (cur_depth < 0.1)
    {
        return;
    }
    float3 cur_pt = cam_pts_cuda[center];
    bool debug = false;

    if ((debug && p.x == 880 && p.y == 676) || !debug)
    {
        if (debug)
        {
            printf("cur_depth: %f \n", cur_depth);
            printf("cur pt:(%d, %d)\n", p.x, p.y);
        }
        //* 采用放射搜索策略
        int num_dirs = 16;     //* 搜索方向数量
        float angles[50];      //* 方向角度
        float2 dirs[50];       //* 方向向量
        bool valid_flag[50];   //* 是否有效
        float search_dist[50]; //* 沿射线搜索距离
        int2 stop_offset[50];  //* 终止坐标偏置
        float3 stop_pt[50];    //* 终止值, 0表示可以继续搜索，0-1 表示终止值，2表示越界
        float max_cos[50];     //* 记录每两个相反方向的夹角最大值
        float stop_ths = 10;   //* 终止阈值
        float max_dist = 50.0; //* 最远搜索100个像素

        angles[0] = 0.0;
        dirs[0].x = 1.0;
        dirs[0].y = 0.0;
        valid_flag[0] = false;
        //* 初始化搜索方向
        float delta_angle = 360.0 / (float)num_dirs;
        for (int i = 1; i < num_dirs; i++)
        {
            angles[i] = angles[i - 1] + delta_angle;
            dirs[i].x = cos(angles[i] / 180 * M_PI);
            dirs[i].y = sin(angles[i] / 180 * M_PI);
            valid_flag[i] = false;
            // printf("angle[%d]: %f (%f, %f)\t \n", i, angles[i], dirs[i].x, dirs[i].y);
        }
        //* 开始搜索，同时搜索两个相反方向
        int half_num_dirs = num_dirs / 2;
        int num_valid = 0;
        for (int i = 0; i < half_num_dirs; i++)
        {
            max_cos[i] = -1.0;
            float radius_step = 10.0;
            float search_dist1, search_dist2;
            float3 stop_pt1, stop_pt2;
            int2 stop_offset1, stop_offset2;
            for (float init_radius = 10.0; init_radius < max_dist; init_radius += radius_step)
            {
                valid_flag[i] = false;
                valid_flag[i + half_num_dirs] = false;
                float radius = init_radius;
                while (radius < max_dist)
                {
                    int off_x = (int)(dirs[i].x * radius);
                    int off_y = (int)(dirs[i].y * radius);
                    int2 new_p = make_int2(p.x + off_x, p.y + off_y);
                    if (new_p.x < width && new_p.y < height && new_p.x >= 0 && new_p.y >= 0)
                    {
                        float new_depth = depth_new_cuda[new_p.y * width + new_p.x];
                        float depth_diff = fabs(cur_depth - new_depth);
                        if (depth_diff < stop_ths && new_depth > 0) //* 达到阈值而终止
                        {
                            search_dist1 = radius;
                            stop_offset1.x = off_x;
                            stop_offset1.y = off_y;
                            valid_flag[i] = true;
                            stop_pt1 = cam_pts_cuda[new_p.y * width + new_p.x];
                            break;
                        }
                    }
                    else //* 越界而终止
                    {
                        break;
                    }
                    radius++;
                }
                if (!valid_flag[i])
                {
                    continue;
                }
                radius = init_radius;
                while (radius < max_dist)
                {
                    int off_x = (int)(dirs[i + half_num_dirs].x * radius);
                    int off_y = (int)(dirs[i + half_num_dirs].y * radius);
                    int2 new_p = make_int2(p.x + off_x, p.y + off_y);
                    if (new_p.x < width && new_p.y < height && new_p.x >= 0 && new_p.y >= 0)
                    {
                        float new_depth = depth_new_cuda[new_p.y * width + new_p.x];
                        float depth_diff = fabs(cur_depth - new_depth);
                        if (depth_diff < stop_ths && new_depth > 0) //* 达到阈值而终止
                        {
                            search_dist2 = radius;
                            stop_offset2.x = off_x;
                            stop_offset2.y = off_y;
                            valid_flag[i + half_num_dirs] = true;
                            stop_pt2 = cam_pts_cuda[new_p.y * width + new_p.x];
                            break;
                        }
                    }
                    else //* 越界而终止
                    {
                        break;
                    }
                    radius++;
                }
                if (!valid_flag[i + half_num_dirs])
                {
                    continue;
                }
                //* 计算夹角
                float3 dir1, dir2;
                dir1.x = stop_pt1.x - cur_pt.x;
                dir1.y = stop_pt1.y - cur_pt.y;
                dir1.z = stop_pt1.z - cur_pt.z;
                dir2.x = stop_pt2.x - cur_pt.x;
                dir2.y = stop_pt2.y - cur_pt.y;
                dir2.z = stop_pt2.z - cur_pt.z;
                float dir1_dot_dir2 = dir1.x * dir2.x + dir1.y * dir2.y + dir1.z * dir2.z;
                float dir1_norm = sqrt(dir1.x * dir1.x + dir1.y * dir1.y + dir1.z * dir1.z);
                float dir2_norm = sqrt(dir2.x * dir2.x + dir2.y * dir2.y + dir2.z * dir2.z);
                float cos_theta = dir1_dot_dir2 / (dir1_norm * dir2_norm);
                if (max_cos[i] < cos_theta)
                {
                    max_cos[i] = cos_theta;
                    search_dist[i] = search_dist1;
                    stop_offset[i] = stop_offset1;
                    stop_pt[i] = stop_pt1;
                    search_dist[i + half_num_dirs] = search_dist2;
                    stop_offset[i + half_num_dirs] = stop_offset2;
                    stop_pt[i + half_num_dirs] = stop_pt2;
                }
            }
            if (valid_flag[i + half_num_dirs] && valid_flag[i])
            {
                num_valid++;
            }

            if (debug)
            {
                // printf("angle[%d]: %f (%f, %f)\t \n", i, angles[i], dirs[i].x, dirs[i].y);
                printf("idx: %d, search_dist: %f, stop_pos: (%d, %d), \n", i, search_dist[i], p.x + stop_offset[i].x, p.y + stop_offset[i].y);
                printf("idx: %d, search_dist: %f, stop_pos: (%d, %d), \n", i + half_num_dirs, search_dist[i + half_num_dirs], p.x + stop_offset[i + half_num_dirs].x, p.y + stop_offset[i + half_num_dirs].y);
                printf("stop_pt[i]:(%f, %f, %f), stop_pt[i+8]: (%f, %f, %f)\n", stop_pt[i].x, stop_pt[i].y, stop_pt[i].z, stop_pt[i + half_num_dirs].x, stop_pt[i + half_num_dirs].y, stop_pt[i + half_num_dirs].z);
                printf("dir1:(%f, %f, %f), dir2: (%f, %f, %f)\n", stop_pt[i].x - cur_pt.x, stop_pt[i].y - cur_pt.y, stop_pt[i].z - cur_pt.z, stop_pt[i + half_num_dirs].x - cur_pt.x, stop_pt[i + half_num_dirs].y - cur_pt.y, stop_pt[i + half_num_dirs].z - cur_pt.z);
                printf("idx: %d, angle: %f, cos_theta: %f, theta: %f\n\n", i, angles[i], max_cos[i], acosf(max_cos[i]) / M_PI * 180);
            }
        }
        sort_large(max_cos, half_num_dirs);
        if (debug)
        {
            for (int i = 0; i < half_num_dirs; i++)
            {
                printf("%f, %f\n", max_cos[i], acosf(max_cos[i]) / M_PI * 180);
            }
        }
        curve_cuda[center] = 0.5 * (max_cos[0] + 1);

        return;
    }
}

void depth_filter_cuda(int width, int height, const float *depth_cuda, float *depth_new_cuda, curandState *rand_states_cuda)
{
    int BLOCK_W = 32;
    int BLOCK_H = (BLOCK_W / 2);

    dim3 grid_size_randinit;
    grid_size_randinit.x = (width + 16 - 1) / 16;
    grid_size_randinit.y = (height + 16 - 1) / 16;
    grid_size_randinit.z = 1;
    dim3 block_size_randinit;
    block_size_randinit.x = 16;
    block_size_randinit.y = 16;
    block_size_randinit.z = 1;

    //* 设置随机数
    long clock_for_rand = clock();
    kernel_set_random<<<grid_size_randinit, block_size_randinit>>>(width, height, rand_states_cuda, clock_for_rand);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    // depth_interp_comp_kernel<<<grid_size_randinit, block_size_randinit>>>(width, height, depth_cuda, color_smooth_cuda, normal_cuda, depth_new_cuda, rand_states_cuda);
    //* 深度滤波，主要去掉反射造成的深度点！
    depth_filter_kernel<<<grid_size_randinit, block_size_randinit>>>(width, height, depth_cuda, depth_new_cuda, rand_states_cuda);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
}

__global__ void depth_smooth_kernel(int width, int height, const float *depth_cuda, float *depth_new_cuda, float3 *cam_pts_cuda, float *curve_cuda, float *color_smooth_cuda, curandState *rand_states_cuda, float *cam_param_cuda)
{
    const int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (p.x >= width || p.y >= height || p.x < 0 || p.y < 0)
    {
        return;
    }
    const int center = p.y * width + p.x;
    float cur_depth = depth_cuda[center];
    float3 cur_pt = cam_pts_cuda[center];
    float cur_color_smooth = color_smooth_cuda[center];
    bool debug = false;

    if ((debug && p.x == 597 && p.y == 449) || !debug)
    // if ((debug && p.x == 880 && p.y == 418) || !debug )
    {
        if (debug)
        {
            printf("cur_depth: %f \n", cur_depth);
            printf("cur pt:(%d, %d)\n", p.x, p.y);
            printf("cam_param_cuda\n");
            for (int i = 0; i < 4; i++)
            {
                printf("%f ,", cam_param_cuda[i]);
            }
            printf("\n");
            // test_fit(); //* 拟合测试
        }
        //* 采用放射搜索策略
        int num_dirs = 16;     //* 搜索方向数量
        float angles[50];      //* 方向角度
        float2 dirs[50];       //* 方向向量
        bool valid_flag[50];   //* 是否有效
        float search_dist[50]; //* 沿射线搜索距离
        int2 stop_offset[50];  //* 终止坐标偏置
        float3 stop_pt[50];    //* 终止值, 0表示可以继续搜索，0-1 表示终止值，2表示越界
        float max_cos[50];     //* 记录每两个相反方向的夹角最大值
        float stop_ths = 10;   //* 终止阈值
        float max_dist = 50.0; //* 最远搜索100个像素
        float color_smooth_ths = 0.1, curve_ths = 0.1, ideal_color_smooth_ths = 0.01;

        angles[0] = 0.0;
        dirs[0].x = 1.0;
        dirs[0].y = 0.0;
        valid_flag[0] = false;
        //* 初始化搜索方向
        float delta_angle = 360.0 / (float)num_dirs;
        for (int i = 1; i < num_dirs; i++)
        {
            angles[i] = angles[i - 1] + delta_angle;
            dirs[i].x = cos(angles[i] / 180 * M_PI);
            dirs[i].y = sin(angles[i] / 180 * M_PI);
            valid_flag[i] = false;
            // printf("angle[%d]: %f (%f, %f)\t \n", i, angles[i], dirs[i].x, dirs[i].y);
        }
        //* 首先考虑颜色平滑值，如果周围的的颜色平滑值都很小，则可以直接判断这点平滑
        bool smooth_flag = true;
        for (int i = 0; i < num_dirs && smooth_flag; i++)
        {
            float radius = 1;
            while (radius < 10 && smooth_flag)
            {
                int off_x = (int)(dirs[i].x * radius);
                int off_y = (int)(dirs[i].y * radius);
                int2 new_p = make_int2(p.x + off_x, p.y + off_y);
                if (new_p.x < width && new_p.y < height && new_p.x >= 0 && new_p.y >= 0)
                {
                    float new_color_smooth = color_smooth_cuda[new_p.y * width + new_p.x];
                    float color_smooth_diff = fabs(new_color_smooth - cur_color_smooth);
                    if (color_smooth_diff > color_smooth_ths) //* 达到阈值而终止
                    {
                        smooth_flag = false;
                        break;
                    }
                }
                else //* 越界而终止
                {
                    smooth_flag = false;
                    break;
                }
                radius++;
            }
        }
        if (smooth_flag)
        {
            //* 寻找合适点以平滑当前点
            for (int i = 0; i < num_dirs; i++)
            {
                valid_flag[i] = false;
                float radius = 10.0;
                while (radius < max_dist)
                {
                    int off_x = (int)(dirs[i].x * radius);
                    int off_y = (int)(dirs[i].y * radius);
                    int2 new_p = make_int2(p.x + off_x, p.y + off_y);
                    if (new_p.x < width && new_p.y < height && new_p.x >= 0 && new_p.y >= 0)
                    {
                        float new_color_smooth = color_smooth_cuda[new_p.y * width + new_p.x];
                        float color_smooth_diff = fabs(new_color_smooth - cur_color_smooth);
                        float new_depth = depth_cuda[new_p.y * width + new_p.x];
                        float curve = curve_cuda[new_p.y * width + new_p.x];
                        if (new_depth > 0) //* 先做好记录，下面的判断不能保证采样点的深度有效
                        {
                            search_dist[i] = radius;
                            stop_offset[i].x = off_x;
                            stop_offset[i].y = off_y;
                            stop_pt[i] = cam_pts_cuda[new_p.y * width + new_p.x];
                        }
                        // if(color_smooth_diff>color_smooth_ths || (curve>curve_ths && color_smooth_diff>ideal_color_smooth_ths) || radius+1>=max_dist) //* 达到阈值而终止
                        if (color_smooth_diff < color_smooth_ths && new_depth > 0) //* 达到阈值而终止
                        {
                            valid_flag[i] = true; //* 主要说明当前方向有效，而不是某个点
                            break;
                        }
                    }
                    else //* 越界而终止
                    {
                        break;
                    }
                    radius++;
                }
                if (debug && valid_flag[i])
                {
                    // printf("angle[%d]: %f (%f, %f)\t \n", i, angles[i], dirs[i].x, dirs[i].y);
                    printf("idx: %d, search_dist: %f, stop_pos: (%d, %d), \n", i, search_dist[i], p.x + stop_offset[i].x, p.y + stop_offset[i].y);
                    printf("stop_pt[i]:(%f, %f, %f)\n", stop_pt[i].x, stop_pt[i].y, stop_pt[i].z);
                    printf("dir:(%f, %f, %f)\n\n", stop_pt[i].x - cur_pt.x, stop_pt[i].y - cur_pt.y, stop_pt[i].z - cur_pt.z);
                }
            }
            float3 v_points[50];
            int num_v_points = 0;
            float max_depth = 0, min_depth = 1000;
            for (int i = 0; i < num_dirs; i++)
            {
                if (valid_flag[i])
                {
                    v_points[num_v_points] = stop_pt[i];
                    num_v_points++;
                    if (stop_pt[i].z > max_depth)
                    {
                        max_depth = stop_pt[i].z;
                    }
                    if (stop_pt[i].z < min_depth)
                    {
                        min_depth = stop_pt[i].z;
                    }
                }
            }
            if (num_v_points < 14)
            {
                return;
            }
            float M[3];
            Least_squares(v_points, num_v_points, M);
            float mean_err, max_err;
            max_err = 0.0;
            mean_err = 0.0;
            for (int i = 0; i < num_v_points; i++)
            {

                float new_z = M[0] + M[1] * v_points[i].x + M[2] * v_points[i].y;
                float err_tmp = fabs(new_z - v_points[i].z);
                mean_err += err_tmp;
                if (err_tmp > max_err)
                {
                    max_err = err_tmp;
                }
            }
            mean_err /= num_v_points;
            if (debug)
            {
                for (int i = 0; i < 3; i++)
                {
                    printf("M%d = %lf\n", i, M[i]);
                }
                mean_err /= num_v_points;
                printf("mean error: %f, max error: %f\n\n", mean_err, max_err);
            }
            float esti_depth = M[0] / (1 - M[1] * (p.x - cam_param_cuda[1]) / cam_param_cuda[0] - M[2] * (p.x - cam_param_cuda[3]) / cam_param_cuda[2]);
            if (esti_depth > 0.5 && esti_depth < 10.0 && mean_err < 0.1)
            {
                depth_new_cuda[center] = esti_depth;
            }
            if (debug)
            {
                printf("esti_depth: %f\n", esti_depth);
            }
        }
        if (debug)
        {
            printf("smooth_flag: %d \n", (int)smooth_flag);
        }
        return;
    }
}

void depth_interp_comp_cuda(int width, int height, const float *depth_cuda, float *color_smooth_cuda, float3 *normal_cuda, float *depth_new_cuda, curandState *rand_states_cuda, float3 *cam_pts_cuda, float3 *esti_normal_cuda, float *curve_cuda, float *cam_param_cuda)
{
    int BLOCK_W = 32;
    int BLOCK_H = (BLOCK_W / 2);

    dim3 grid_size_randinit;
    grid_size_randinit.x = (width + 16 - 1) / 16;
    grid_size_randinit.y = (height + 16 - 1) / 16;
    grid_size_randinit.z = 1;
    dim3 block_size_randinit;
    block_size_randinit.x = 16;
    block_size_randinit.y = 16;
    block_size_randinit.z = 1;

    //* 设置随机数
    long clock_for_rand = clock();
    kernel_set_random<<<grid_size_randinit, block_size_randinit>>>(width, height, rand_states_cuda, clock_for_rand);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    //* 曲率估计
    curve_esti_kernel<<<grid_size_randinit, block_size_randinit>>>(width, height, depth_cuda, cam_pts_cuda, curve_cuda, rand_states_cuda);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    //* 深度平滑
    depth_smooth_kernel<<<grid_size_randinit, block_size_randinit>>>(width, height, depth_cuda, depth_new_cuda, cam_pts_cuda, curve_cuda, color_smooth_cuda, rand_states_cuda, cam_param_cuda);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    // cudaMemcpy(depth_host, depth_cuda, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
    // CUDA_SAFE_CALL(cudaDeviceSynchronize());
}

__global__ void ransac_plane_cal_dist_kernel(int n_pts, const float3 *ref_pts_cuda, float *dist_cuda, float4 plane_param)
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

void ransac_plane_cal_dist_cuda(int n_pts, const float3 *ref_pts_cuda, float *dist_cuda, float4 plane_param)
{
    // Matrix addition kernel launch from host code
    dim3 threadsPerBlock(512);
    dim3 numBlocks((n_pts + threadsPerBlock.x - 1) / threadsPerBlock.x);
    ransac_plane_cal_dist_kernel<<<numBlocks, threadsPerBlock>>>(n_pts, ref_pts_cuda, dist_cuda, plane_param);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
}
