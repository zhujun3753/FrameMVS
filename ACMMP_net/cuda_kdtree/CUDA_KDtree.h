#ifndef __CUDA_KDTREE_H__
#define __CUDA_KDTREE_H__

#include "KDtree.h"
#include <vector>
#include <torch/custom_class.h>
#include <torch/script.h>

struct CUDA_KDNode
{
    int level;
    int parent, left, right;
    float split_value;
    int num_indexes;
    int indexes;
};

using namespace std;

void CheckCUDAError(const char *msg);
void SearchBatch_cpp(CUDA_KDNode *m_gpu_nodes, int *m_gpu_indexes, Point *m_gpu_points,
    int m_num_points, const vector <Point> &queries, vector <int> &indexes, vector <float> &dists);
void cuda_test();
void cuda_test1();

//！ 不加 : torch::CustomClassHolder 的话所有与类相关的函数在.cu下的实现都找不到。。。。。
struct CUDA_KDTree : torch::CustomClassHolder
{
    ~CUDA_KDTree();
    void CreateKDTree(KDNode *root, int num_nodes, const vector <Point> &data);
    void Search(const vector <Point> &queries, vector <int> &indexes, vector <float> &dists);

    CUDA_KDNode *m_gpu_nodes;
    int *m_gpu_indexes;
    Point *m_gpu_points;
    int m_num_points;
};



#endif
