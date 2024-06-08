#include <iostream>
#include <sycl/sycl.hpp>

#include "bitonic.hpp"

#define DEF_ARR_SIZE 0x10000000

/**
 * Pass in the number of values you want to use for the sort. Otherwise use DEF_ARR_SIZE
*/
int main(int argc, char** argv) 
{
    long unsigned int N;
    N = DEF_ARR_SIZE;
    if(argc > 1) N = 1 << atoi(argv[1]);

    std::cout << "Number Elems: " << N << std::endl;
    
    // select device for offload
    sycl::queue q(sycl::gpu_selector_v);

    // choose the type for the reduction
    typedef float T;

    // initialize some data array
    auto data = sycl::malloc_shared<T>(N,q);

    std::cout << "Naive Solution" << std::endl;
    randomizeArr(data, N);

    // Print input array
    std::cout << "Input Array:" << std::endl;
    for(int i = 0; i < 5; i++)
        std::cout << data[i] << ", ";
    std::cout << "..., ";
    for(int i = N-5; i < N; i++)
        std::cout << data[i] << ", ";
    std::cout << std::endl;

    // computation on GPU
    auto t1 = std::chrono::steady_clock::now();
    bitonic_sort_naive(data, N, q);
    auto t2 = std::chrono::steady_clock::now();

    // Print output array
    std::cout << "Output Array:" << std::endl;
    for(int i = 0; i < 5; i++)
        std::cout << data[i] << ", ";
    std::cout << "..., ";
    for(int i = N-5; i < N; i++)
        std::cout << data[i] << ", ";
    std::cout << std::endl;

    std::cout << "Time (ms): " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1000000.0f << std::endl;
    
    if (N <= 32768){
        std::cout << "Kernel Solution" << std::endl;
        randomizeArr(data, N);

        // Print input array
        std::cout << "Input Array:" << std::endl;
        for(int i = 0; i < 5; i++)
            std::cout << data[i] << ", ";
        std::cout << "..., ";
        for(int i = N-5; i < N; i++)
            std::cout << data[i] << ", ";
        std::cout << std::endl;

        // computation on GPU
        t1 = std::chrono::steady_clock::now();
        bitonic_sort(data, N, q);
        t2 = std::chrono::steady_clock::now();

        // Print output array
        std::cout << "Output Array:" << std::endl;
        for(int i = 0; i < 5; i++)
            std::cout << data[i] << ", ";
        std::cout << "..., ";
        for(int i = N-5; i < N; i++)
            std::cout << data[i] << ", ";
        std::cout << std::endl;

        std::cout << "Time (ms): " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1000000.0f << std::endl;
    }
    
    
    free(data, q);

    return 0;
}