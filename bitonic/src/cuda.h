/**
 * This file has some generalizations of common cuda practices that I just want to put in a header file
*/
#ifndef CUDA_HELPING_H
#define CUDA_HELPING_H

#include "device_launch_parameters.h"
#include <iostream>
#include <functional>

/**
 * Convert a 3D thread index to a 1D flattened idx.
*/
__device__ inline int getFlattenedIdx()
{
    // int maxx = blockIdx.x * blockDim.x;
    // int maxy = blockIdx.y * blockDim.y;
    // int maxz = blockIdx.z * blockDim.z;

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // int j = threadIdx.y + blockIdx.y * blockDim.y;
    // int k = threadIdx.z + blockIdx.z * blockDim.z;

    return idx;
}

#define CEIL_DIV(A, B) ((A+B-1) / B) 

inline void gpuTime(std::function<void(void)> block, std::string msg)
{                                            
    cudaEvent_t start, stop;                              
    float elapsed=0;                                       
    cudaEventCreate(&start);                              
    cudaEventCreate(&stop);                              
    cudaEventRecord(start, 0);                         
    block();                                  
    cudaEventRecord(stop, 0);                      
    cudaEventSynchronize(stop);                    
    cudaEventElapsedTime(&elapsed, start, stop);   
    std::cout << msg << ": " << std::to_string(elapsed) << " ms" << std::endl;  
}           

#define CUDA_ERR_CHK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#endif 