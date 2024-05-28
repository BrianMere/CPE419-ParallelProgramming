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
 * 
 * This reduction reduces entries closer to each idx, then spreads out over time. 
*/

template<typename T>
__global__ void reduceAdjacent1(T* d_in, T* d_res, unsigned int n)
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
        if (c_idx + s < max_thdsperblk && idx + s < n && c_idx % (s << 1) == 0)
        {
            cache[c_idx] = op(cache[c_idx], cache[c_idx + s]);
        }
        __syncthreads();
    }

    // Upload cache data to the d_in
    if(c_idx == 0 && idx<n)
        d_in[idx] = cache[0];
    if(idx==0)
    {
        *d_res = d_in[0]; 
    }
}

template<typename T>
__global__ void reduceAdjacent2(T* d_in, T* d_res, unsigned int n, int s)
{
    int idx = getFlattenedIdx();

    // again only do the following if you're in the size AND are the 'left' thread
    if (idx + s < n && idx % (s << 1) == 0)
    {
        d_in[idx] = op(d_in[idx], d_in[idx + s]);
    }
    __syncthreads();
    

    // At the end, d_in[0] has the data we want.
    if(idx == 0)
        *d_res = d_in[0];
}

template<typename T>
void reduceAdjacent(T* d_in, T* d_res, unsigned int n, int blocks, int threads_per)
{
    reduceAdjacent1<<<blocks, threads_per>>>(d_in, d_res, n);
    for(unsigned int s = threads_per; s < n; s <<= 1)
    {

        reduceAdjacent2<<<blocks, threads_per>>>(d_in, d_res, n, s);
    }
}


/**
 * Apply a reduction to get the "sum" of a generic type T. 
 * 
 * You MUST make sure that the T* data is cudaMalloced ALONG WITH the *op lambda to make sure it's available on the 
 * GPU
 * 
 * Make sure that d_in has size n.
 * 
 * This is the version that reduces with entries farther from it until it gets closer together.
*/
template<typename T>
__global__ void reduceSpread1(T* d_in, T* d_res, unsigned int n, int s)
{
    int idx = getFlattenedIdx();
    int max_thdsperblk = blockDim.x * blockDim.y * blockDim.z;


    // again only do the following if you're in the size AND are the 'left' group of threads
    if (idx + s < n && idx < s)
    {
        d_in[idx] = op(d_in[idx], d_in[idx + s]);
    }

    
}

template<typename T>
__global__ void reduceSpread2(T* d_in, T* d_res, unsigned int n)
{
    int idx = getFlattenedIdx();
    int max_thdsperblk = blockDim.x * blockDim.y * blockDim.z;

    __shared__ T cache[MAXNUM_THDSPERBLK]; // we may have extra memory but that's okay
    int c_idx = idx % max_thdsperblk;

    // Copy data entry to d_out
    cache[c_idx] = d_in[idx];
    __syncthreads();

    // Do the reduction within d_out. Here `s` is the offset between entries to add
    for(unsigned int s = max_thdsperblk >> 1; s > 0; s >>= 1)
    {
        // only do the following if you're in the cache AND you're the 'left' thread
        if (c_idx + s < max_thdsperblk&& idx+s<n && c_idx < s)
        {
            cache[c_idx] = op(cache[c_idx], cache[c_idx + s]);
        }
        __syncthreads();
    }

    // Upload cache data to the d_in
    if(c_idx == 0)
        d_res[0] = cache[0];
    __syncthreads();
}

template<typename T>
void reduceSpread(T* d_in, T* d_res, unsigned int n, int blocks, int threads_per_block)
{
    // First reduce down to a size of a power of 2
    int n_clog = 1;
    while(n_clog < n)
        n_clog <<= 1;

    // Then combine cache answers (s here is half the considered array size)
    for(unsigned int s = n_clog >> 1; s >= threads_per_block; s >>= 1)
    {
        reduceSpread1<<<blocks,threads_per_block>>>(d_in, d_res, n, s);
    }
    reduceSpread2<<<1, threads_per_block>>>(d_in, d_res, n);
}


template<typename T>
void reduce_omp(T* d_in, T* d_res, unsigned int n)
{
    T sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for (uint64_t i = 0; i < n; i++)
    {
        sum += d_in[i];
    }
    *d_res = sum; 
}

#endif