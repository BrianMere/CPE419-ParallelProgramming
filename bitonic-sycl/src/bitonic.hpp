#ifndef BITONIC_SORT_H
#define BITONIC_SORT_H

#include <cstdlib>
#include <sycl/sycl.hpp>
#include <map>

/**
 * Generate an array of numbers in the range -1 to 1
*/
template <typename T>
void randomizeArr(T * arr, int n)
{
    for(int i = 0; i < n; i++)
        arr[i] =  2.0f * (static_cast<T> (rand()) / static_cast<T> (RAND_MAX)) - 1.0f;
}

/**
 * Do the per thread sort. The idea is that each thread will just do a swap check based on the wrapper arguments.
 * If no chunking is required, then pass chunks = 1 for just one loop of work. 
*/
template <typename T> void bitonic_sort_per_thread(T *arr, int n, int j, int k, int threadId, int chunks)
{
    int start = threadId * chunks;
    for (int i = start; (i < start + chunks) && (i < n); i++)
    {
        int l = (i^j); 
        if ( i < n && l > i){
                if (  ((i & k) == 0) && (arr[i] > arr[l])
                || ((i & k) != 0) && (arr[i] < arr[l]) )
                {
                        T temp = arr[i];
                        arr[i] = arr[l];
                        arr[l] = temp;
                }
        }
    }
    
}

template <typename T> void bitonic_sort(T *arr, int n, sycl::queue q)
{
    /**
     * for(k = 2; j >= lowest power of 2 greater than n; k *= 2): // jk is the max size of the block of our current run
     *  for(j = k/2; j > 0; j /= 2): // j is the size of the translation when comparing elements
     *     "do the swap work per thread for many threads"
    */

    // int num_buffers = n - 1; // We'll make 1 + 2 + 4 + ... + n//2 buffers which is equivalent to n - 1 buffers
    // std::map<int, sycl::buffer<T>> buffs; // contained within this vector... 
    // for(int i = 0; i < num_buffers; i++)
    //     // buffs[i] = 

    // for(int k = 2; k < n; k << 1){
    //     // Create memory kernel buffers for each k pass that are of size k. We need n / k of them consequently.
    //     int num_buffers = k / n;
    //     int buff_size = k; 
    //     auto R = sycl::range<1>{ k };
    //     std::vector<sycl::buffer<T>> buffs;
        
    //     // Push a new buffer into the vector for our pass. 
    //     for(int i = 0; i < num_buffers; i++)
    //         buffs.push_back(new sycl::buffer<T>{R});
        
    //     // Initialize our buffers to have the previous pass's data. Each kernel is a request to copy data.
    //     for(int i = 0; i < num_buffers; i++)
    //     {
    //         q.submit([&](sycl::handler& h){
    //             sycl::accessor out(buffs[i], h, sycl::write_only);
    //             h.parallel_for(sycl::range<1>{ buff_size }, [=](sycl::id<1> i){
    //                 out[j] = arr[i * buff_size + j];
    //             });
    //         });
    //     }


    //     for(int j = k / 2; j > 0; j >> 1){
    //         q.submit([&](sycl::handler& h){
    //             h.parallel_for(sycl::nd_range<1>(sycl::range<1>(n), sycl::range<1>()), [=](sycl::nd_item<1> item){
    //                 auto idx = item.get_global_id();
    //                 //bitonic_sort_per_thread(arr, n, j, k, idx, 1);
    //             });
    //         });
    //     }
    // }

    // q.wait(); // we've scheduled everything, as well as each dependency. Just sit back and relax and wait!
}

template<typename T> void bitonic_sort_naive(T * arr, int n, sycl::queue q)
{
    for(int k = 2; k <= n; k <<= 1)
    {
        for(int j = k >> 1; j > 0; j >>= 1)
        {
            q.parallel_for(n, [=](auto i){
                bitonic_sort_per_thread(arr, n, j, k, i, 1);
            }).wait();
        }
    }
}


#endif // BITONIC_SORT_H