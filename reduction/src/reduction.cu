#include "reduction.h"
#include "cuda.h"


template<typename T>
/**
 * Apply a reduction to get the "sum" of a generic type T. 
 * 
 * You MUST make sure that the T* data is cudaMalloced ALONG WITH the *op lambda to make sure it's available on the 
 * GPU
 * 
 * Make sure that d_in has size n.
*/
__global__ void reduceOperation(T* d_in, T* d_res, unsigned int n)
{
    int idx = getFlattenedIdx();
    __shared__ T cache[MAXNUM_THDSPERBLK];
    int c_idx = idx % MAXNUM_THDSPERBLK;

    // Copy data entry to d_out
    cache[c_idx] = d_in[idx];
    __syncthreads();

    // Do the reduction within d_out. Here `s` is the offset between entries to add
    for(unsigned int s = 1; s < MAXNUM_THDSPERBLK; s <<= 1)
    {
        if (c_idx + s < MAXNUM_THDSPERBLK)
        {
            cache[c_idx] = op(cache[c_idx], cache[c_idx + s]);
        }
        __syncthreads();
    }

    // Upload cache data to the d_in
    d_in[idx] = cache[0];
    __syncthreads();

    // Then combine cache answers
    for(unsigned int s = MAXNUM_THDSPERBLK; s < n; s <<= 1)
    {
        if (idx + s < n)
        {
            d_in[idx] = op(d_in[idx], d_in[idx + s]);
        }
        __syncthreads();
    }

    // At the end, d_in[0] has the data we want.
    if(idx == 0)
        d_res[0] = d_in[0];
}
