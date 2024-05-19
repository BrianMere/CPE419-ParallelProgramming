#ifndef REDUCTION_H
#define REDUCTION_H

#include "device_launch_parameters.h"

#define MAXNUM_THDSPERBLK 1024

template<typename T>
__device__ T op(T x, T y) { return  x + y; };

template<typename T>
__global__ void reduceOperation(T* d_in, T* d_res, unsigned int n);

#endif