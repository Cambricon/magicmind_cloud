#include "post_process.hpp"
#include "utils.hpp"

std::vector<int> ArgTopK(const float *data, int classes, int k) 
{
    std::vector<int> result;
    result.reserve(k);
    for (int i = 0; i < k; ++i) result.push_back(i);
    auto comp = [data] (int a, int b) { return data[a] > data[b]; };
    std::make_heap(result.begin(), result.end(), comp);
    for (int i = k; i < classes; ++i) {
        if (comp(i, result[0])) {
            std::pop_heap(result.begin(), result.end(), comp);
            result.back() = i ;
            std::push_heap(result.begin(), result.end(), comp);
        }
    }
    std::sort(result.begin(), result.end(), comp);
    return result;
}

