#include "CUDA_KDtree.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
#include <cstdio>

#define CUDA_STACK 100 // fixed size stack elements for each thread, increase as required. Used in SearchAtNodeRange.

__device__ float Distance(const Point &a, const Point &b)
{
    float dist = 0;

    for(int i=0; i < KDTREE_DIM; i++) {
        float d = a.coords[i] - b.coords[i];
        dist += d*d;
    }

    return dist;
}


__device__ void SearchAtNodeRange(const CUDA_KDNode *nodes, const int *indexes, const Point *pts, const Point &query,int cur, float range, int *ret_index, float *ret_dist)
{
    // Goes through all the nodes that are within "range"
    int best_idx = 0;
    float best_dist = FLT_MAX;
    // Ok, we don't have nice STL vectors to use, and we can't dynamically allocate memory with CUDA??
    // We'll use a fixed length stack, increase this as required
    int to_visit[CUDA_STACK];
    int to_visit_pos = 0;
    to_visit[to_visit_pos++] = cur;
    while(to_visit_pos)
    {
        int next_search[CUDA_STACK];
        int next_search_pos = 0;
        while(to_visit_pos)
        {
            cur = to_visit[to_visit_pos-1];
            to_visit_pos--;
            int split_axis = nodes[cur].level % KDTREE_DIM;
            if(nodes[cur].left == -1)
            {
                for(int i=0; i < nodes[cur].num_indexes; i++)
                {
                    int idx = indexes[nodes[cur].indexes + i];
                    float d = Distance(query, pts[idx]);
                    if(d < best_dist)
                    {
                        best_dist = d;
                        best_idx = idx;
                    }
                }
            }
            else
            {
                if(next_search_pos+1>=CUDA_STACK)
                {
                    continue;
                }
                next_search[next_search_pos++] = nodes[cur].left;
                next_search[next_search_pos++] = nodes[cur].right;
                
            }
            // printf("next_search_pos:%d\n", next_search_pos);
        }
        // No memcpy available??
        for(int i=0; i  < next_search_pos; i++)
            to_visit[i] = next_search[i];
        to_visit_pos = next_search_pos;
    }
    *ret_index = best_idx;
    *ret_dist = best_dist;
}

__device__ void SearchAtNode(const CUDA_KDNode *nodes, const int *indexes, const Point *pts, int cur, const Point &query, int *ret_index, float *ret_dist, int *ret_node)
{
    // Finds the first potential candidate
    int best_idx = 0;
    float best_dist = FLT_MAX;
    while(true)
    {
        int split_axis = nodes[cur].level % KDTREE_DIM;
        // printf("nodes[%d].num_indexes: %d\n",cur, nodes[cur].num_indexes);
        //* nodes[13652].num_indexes: 312 其余均为0
        if(nodes[cur].left == -1)
        {
            *ret_node = cur;
            for(int i=0; i < nodes[cur].num_indexes; i++)
            {
                int idx = indexes[nodes[cur].indexes + i];
                float dist = Distance(query, pts[idx]);
                if(dist < best_dist)
                {
                    best_dist = dist;
                    best_idx = idx;
                }
            }
            break;
        }
        else if(query.coords[split_axis] < nodes[cur].split_value)
        {
            cur = nodes[cur].left;
        }
        else {
            cur = nodes[cur].right;
        }
    }
    *ret_index = best_idx;
    *ret_dist = best_dist;
}

__device__ void Search(const CUDA_KDNode *nodes, const int *indexes, const Point *pts, const Point &query, int *ret_index, float *ret_dist)
{
    // Find the first closest node, this will be the upper bound for the next searches
    int best_node = 0;
    int best_idx = 0;
    float best_dist = FLT_MAX;
    float radius = 0;
    SearchAtNode(nodes, indexes, pts, 0 /* root */, query, &best_idx, &best_dist, &best_node);
    radius = sqrt(best_dist);
    // Now find other possible candidates
    int cur = best_node;
    // printf("cur: %d\n", cur);
    // while(nodes[cur].parent != -1)
    // {
    //     // Go up
    //     int parent = nodes[cur].parent;
    //     int split_axis = nodes[parent].level % KDTREE_DIM;
    //     // Search the other node
    //     float tmp_dist = FLT_MAX;
    //     int tmp_idx;
    //     // printf("parent: %d\n", parent);
    //     if(fabs(nodes[parent].split_value - query.coords[split_axis]) <= radius)
    //     {
    //         // Search opposite node
    //         if(nodes[parent].left != cur)
    //             SearchAtNodeRange(nodes, indexes, pts, query, nodes[parent].left, radius, &tmp_idx, &tmp_dist);
    //         else
    //             SearchAtNodeRange(nodes, indexes, pts, query, nodes[parent].right, radius, &tmp_idx, &tmp_dist);
    //     }
    //     if(tmp_dist < best_dist)
    //     {
    //         best_dist = tmp_dist;
    //         best_idx = tmp_idx;
    //     }
    //     cur = parent;
    //     // printf("cur: %d\n", cur);
    // }
    *ret_index = best_idx;
    *ret_dist = best_dist;
    // printf("best_idx: %d, best_dist: %f\n", best_idx, best_dist);
}

__global__ void SearchBatch(const CUDA_KDNode *nodes, const int *indexes, const Point *pts, int num_pts, Point *queries, int num_queries, int *ret_index, float *ret_dist)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    // if(idx >= num_queries)
    if(idx >= 1)
        return;
    // printf("queries[%d]:(%f, %f, %f)\n",idx, queries[idx].coords[0], queries[idx].coords[1], queries[idx].coords[2] );
    Search(nodes, indexes, pts, queries[idx], &ret_index[idx], &ret_dist[idx]);
}

// void cuda_test()
// {
//     printf("just a test\n");
// }

void SearchBatch_cpp(CUDA_KDNode *m_gpu_nodes, int *m_gpu_indexes, Point *m_gpu_points,
    int m_num_points, const vector <Point> &queries, vector <int> &indexes, vector <float> &dists)
{
    int threads = 512;
    int blocks = queries.size()/threads + ((queries.size() % threads)?1:0);

    Point *gpu_queries;
    int *gpu_ret_indexes;
    float *gpu_ret_dist;

    indexes.resize(queries.size());
    dists.resize(queries.size());

    cudaMalloc((void**)&gpu_queries, sizeof(Point)*queries.size());
    cudaMalloc((void**)&gpu_ret_indexes, sizeof(int)*queries.size());
    cudaMalloc((void**)&gpu_ret_dist, sizeof(float)*queries.size());

    CheckCUDAError("Search");

    cudaMemcpy(gpu_queries, &queries[0], sizeof(Point)*queries.size(), cudaMemcpyHostToDevice);

    CheckCUDAError("Search");

    printf("CUDA blocks/threads: %d %d\n", blocks, threads);

    SearchBatch<<<blocks, threads>>>(m_gpu_nodes, m_gpu_indexes, m_gpu_points, m_num_points, gpu_queries, queries.size(), gpu_ret_indexes, gpu_ret_dist);
    cudaThreadSynchronize();
    printf("CUDA SearchBatch end\n");

    CheckCUDAError("Search");

    cudaMemcpy(&indexes[0], gpu_ret_indexes, sizeof(int)*queries.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(&dists[0], gpu_ret_dist, sizeof(float)*queries.size(), cudaMemcpyDeviceToHost);
    printf("CUDA cudaMemcpy end\n");

    cudaFree(gpu_queries);
    cudaFree(gpu_ret_indexes);
    cudaFree(gpu_ret_dist);
    printf("CUDA cudaFree end\n");
}