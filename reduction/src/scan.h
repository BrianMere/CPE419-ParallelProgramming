#ifndef SCAN_H_
#define SCAN_H_

#include "cuda.h"

template<typename T>
__global__ void reduceDown(T* d_in, T* d_res, unsigned int n, int s)
{
    int idx = getFlattenedIdx();

    // again only do the following if you're in the size AND are the 'left' thread
    if (idx + s < n && idx % (s << 1) == 0)
    {
        int r_idx = n - 1 - idx;
        d_in[r_idx] = op(d_in[r_idx], d_in[r_idx - s]);
    }

}

template<typename T>
__global__ void reduceUp(T* d_in, T* d_res, unsigned int n, int s)
{
    int idx = getFlattenedIdx();

    // again only do the following if you're in the size AND are the 'left' thread
    if (idx + s < n && idx % (s << 1) == 0)
    {
        int r_idx = n - 1 - idx;
        T prev_val = d_in[r_idx];
        d_in[r_idx] = op(d_in[r_idx], d_in[r_idx - s]);
        d_in[r_idx - s] = prev_val;

    }

}

template<typename T>
void segmentScan(T* d_in, T* d_res, unsigned int n, int blocks, int threads_per)
{
    unsigned int s;
    for(s = 1; s < n; s <<= 1)
    {
        reduceDown<<<blocks, threads_per>>>(d_in, d_res, n, s);
    }
    cudaMemcpy(d_res, &d_in[n-1], sizeof(T), cudaMemcpyDeviceToDevice);
    cudaMemset(&d_in[n-1], 0, sizeof(T));
    for(s; s >= 1; s >>= 1)
    {
        reduceUp<<<blocks, threads_per>>>(d_in, d_res, n, s);
    }

}

#endif