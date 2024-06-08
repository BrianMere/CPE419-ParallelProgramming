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
                if (  (((i & k) == 0) && (arr[i] > arr[l]))
                || (((i & k) != 0) && (arr[i] < arr[l])) )
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
    std::map<int, sycl::event> events; // contained within this mapping... 
    for(int k = 2; k <= n; k <<= 1)
    {
        for(int j = k >> 1; j > 0; j >>= 1)
        {
            // here we make an event which is only the j amount of iterations/threads to partition stuff more out.
            for(int offset = 0; offset < n / 2; offset += j)
            {
                events[offset] = q.submit([&] (sycl::handler& h) {
                    if(j == k >> 1) {// on the first j-sweep of every k sweep, it just depends on a combination of previous events
                        std::vector<sycl::event> dependency;
                        for(int j_off = 0; j_off < j; j_off++)
                        {
                            dependency.push_back(events[j_off]);
                        }
                        h.depends_on(dependency);
                    }
                    else // otherwise, it depends on a single, larger event
                        h.depends_on(events[offset / 2]);

                    h.parallel_for(j, [=](auto i){
                        bitonic_sort_per_thread(arr, n, j, k, i + offset, 1);
                    });
                }); 
            }
        }
    }

    q.wait(); // we've scheduled everything, as well as each dependency. Just sit back and relax and wait!
}

template<typename T> void bitonic_sort_naive(T * arr, int n, sycl::queue q)
{
    /**
     * for(k = 2; j >= lowest power of 2 greater than n; k *= 2): // jk is the max size of the block of our current run
     *  for(j = k/2; j > 0; j /= 2): // j is the size of the translation when comparing elements
     *     "do the swap work per thread for many threads"
    */
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