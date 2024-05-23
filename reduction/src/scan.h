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
        d_in[idx + s] = op(d_in[idx], d_in[idx + s]);
    }

}

template<typename T>
void segmentScan(T* d_in, T* d_res, unsigned int n, int blocks, int threads_per)
{
    for(unsigned int s = 1; s < n; s <<= 1)
    {
        reduceDown<<<blocks, threads_per>>>(d_in, d_res, n, s);
    }
    cudaMemcpy(d_res, &d_in[n-1], sizeof(T), cudaMemcpyDeviceToDevice);
}

#endif