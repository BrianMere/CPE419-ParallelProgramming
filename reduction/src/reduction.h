#ifndef REDUCTION_H
#define REDUCTION_H

#include "device_launch_parameters.h"
#include "cuda.h"

#define MAXNUM_THDSPERBLK 1024

template<typename T>
__device__ T op(T x, T y) { return  x + y; };

/**
 * Apply a reduction to get the "sum" of a generic type T. 
 * 
 * You MUST make sure that the T* data is cudaMalloced ALONG WITH the *op lambda to make sure it's available on the 
 * GPU
 * 
 * Make sure that d_in has size n.
*/
template<typename T>
__global__ void reduceOperation(T* d_in, T* d_res, unsigned int n)
{
    int idx = getFlattenedIdx();
    __shared__ T cache[MAXNUM_THDSPERBLK]; // we may have extra memory but that's okay
    int max_thdsperblk = blockDim.x * blockDim.y * blockDim.z;
    int c_idx = idx % max_thdsperblk;

    // Copy data entry to d_out
    cache[c_idx] = d_in[idx];
    __syncthreads();

    // Do the reduction within d_out. Here `s` is the offset between entries to add
    for(unsigned int s = 1; s < max_thdsperblk; s <<= 1)
    {
        // only do the following if you're in the cache AND you're the 'left' thread
        if (c_idx + s < max_thdsperblk && c_idx % (s << 1) == 0)
        {
            cache[c_idx] = op(cache[c_idx], cache[c_idx + s]);
        }
        __syncthreads();
    }

    // Upload cache data to the d_in
    if(c_idx == 0)
        d_in[idx] = cache[0];
    __syncthreads();

    // Then combine cache answers
    for(unsigned int s = max_thdsperblk; s < n; s <<= 1)
    {
        // again only do the following if you're in the size AND are the 'left' thread
        if (idx + s < n && idx % (s << 1) == 0)
        {
            d_in[idx] = op(d_in[idx], d_in[idx + s]);
        }
        __syncthreads();
    }

    // At the end, d_in[0] has the data we want.
    if(idx == 0)
        d_res[0] = d_in[0];
}

#endif