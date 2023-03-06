#include <cstdio>
#include <vector>
#include <cstdlib>
#include <float.h>
#include <sys/time.h>
#include <iostream>

#include "KDtree.h"
#include "CUDA_KDtree.h"

double TimeDiff(timeval t1, timeval t2)
{
    double t;
    t = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
    t += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms

    return t;
}

int main()
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

    return 0;
}



