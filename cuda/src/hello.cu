#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include "matrix.h"

#define THD_PER_BLK 256
#define L1_$LINE_BYTES 128
#define TILESIZE (L1_$LINE_BYTES / sizeof(mat_t) )
#define CEIL_DIV(A, B) ((A+B-1) / B) 

cudaEvent_t start, stop;
float elapsed=0;

#define TIME(f, msg)                            \
cudaEventCreate(&start);                        \
cudaEventCreate(&stop);                         \
cudaEventRecord(start, 0);                      \
f;                                              \
cudaEventRecord(stop, 0);                       \
cudaEventSynchronize (stop);                    \
cudaEventElapsedTime(&elapsed, start, stop);    \
printf(msg": %f ms\n", elapsed);    

__global__ void helloCUDA()
{
    printf("Hello CUDA %d\n", threadIdx.x + blockIdx.x * blockDim.x);
}

__global__ void init(mat_t * A, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if ( i < N )
    {
        // curandState state;
        // curand_init(clock64(), i, 0, &state);
        // A[i] = curand_uniform(&state) * 100;
        A[i] = 1.0f / (i+1)  * 100;
    }
}

__global__ void mmult_stride(mat_t * A, mat_t * B, mat_t * C, int m, int n, int p)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = blockDim.x * gridDim.x;

    int num_elems = m * p;


    for (; i < num_elems; i += total_threads)
    {
        int rowA = i / m;
        int colB = i % p;
        int sum = 0;
        for (int j = 0; j < n; j++)
        {
            sum += A[j + rowA * n]  * B[j*p + colB];
        }
        C[i] = sum;
    }

}

__global__ void mmult_nostride(mat_t * A, mat_t * B, mat_t * C, int m, int n, int p)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    // int total_threads = blockDim.x * gridDim.x;

    int num_elems = m * p;
    int rowA = i / m;
    int colB = i % p;
    int sum = 0;
    if (i < num_elems)
    {
        for (int j = 0; j < n; j++)
        {
            sum += A[j + rowA * n]  * B[j*p + colB];
        }
        C[i] = sum;
    }

}

// // kernel call to tiling implementation. TILESIZE is likely 32 on a side. 
// __global__ void mmult_tiling(mat_t * A, mat_t * B, mat_t * C, int m, int n, int p)
// {
//     __shared__ mat_t s_A[TILESIZE][TILESIZE];
//     __shared__ mat_t s_B[TILESIZE][TILESIZE];
    
    
//     int idx = threadIdx.x + blockIdx.x * blockDim.x;
//     int idy = threadIdx.y + blockIdx.y * blockDim.y;

//     int tilex = blockIdx.x;
//     int tiley = blockIdx.y;
    
//     int offx = threadIdx.x + tilex * blockDim.x; // row of Mat A
//     int offy = threadIdx.y + tiley * blockDim.y; // row of Mat B
    
//     for(int i = 0; i < TILESIZE; i++)
//     {
//         s_A[tiley][tilex] = A[row*width + (i*2 + tx)];
//         s_B[tiley][tilex] = B[(i*2 + ty)*width + col];
//         __syncthreads();

//     }    
// }

__global__ void sum(mat_t * A, mat_t * B, mat_t * C, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if ( i < N )
    {
        C[i] = A[i] + B[i];
    }
    
}

int main()
{
    cudaDeviceProp prop;   
    cudaGetDeviceProperties( &prop, 0);
    printf("Device: %s\n%d threads per block\n%d per MP\n", prop.name, prop.maxThreadsPerBlock, prop.maxThreadsPerMultiProcessor);
    printf("%d total multiprocessors\n", prop.multiProcessorCount);
    int block_count = prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor / prop.maxThreadsPerBlock;
    printf("Using %d blocks\n", block_count);
    // printf("l1 Cache size: %ld\n", prop.sharedMemPerBlock);
    printf("\n");
    
    unsigned int m = 2000;
    unsigned int n = 2000;
    unsigned int p = 2000;
    
    mat_t * A, * B, * C, * D;
    mat_t * h_A, * h_B, * h_C, * h_D;

    h_A = (mat_t *) malloc(m * n * sizeof(mat_t));
    h_B = (mat_t *) malloc(n * p * sizeof(mat_t));
    h_C = (mat_t *) malloc(m * p * sizeof(mat_t));
    h_D = (mat_t *) malloc(m * p * sizeof(mat_t));

    cudaMalloc ((void**)&A, m * n * sizeof(mat_t));
    cudaMalloc ((void**)&B, n * p * sizeof(mat_t));
    cudaMalloc ((void**)&C, m * p * sizeof(mat_t));
    cudaMalloc ((void**)&D, m * p * sizeof(mat_t));

    init    <<<CEIL_DIV(m * n, THD_PER_BLK), THD_PER_BLK>>> (A,m * n);
    init    <<<CEIL_DIV(n * p, THD_PER_BLK), THD_PER_BLK>>> (B,n * p);
    // sum     <<<CEIL_DIV(n, THD_PER_BLK), THD_PER_BLK>>> (A,B,C,n);

    TIME((mmult_nostride   <<<CEIL_DIV(m * p, 256), 256>>> (A,B,C, m, n, p)), "GPU NOSTRIDE 256");
    TIME((mmult_nostride   <<<CEIL_DIV(m * p, 512), 512>>> (A,B,C, m, n, p)), "GPU NOSTRIDE 512");
    TIME((mmult_nostride   <<<CEIL_DIV(m * p, 1024), 1024>>> (A,B,C, m, n, p)), "GPU NOSTRIDE 1024");

    TIME((mmult_stride   <<<block_count,256>>> (A,B,D, m, n, p)), "GPU STRIDE 32x256");
    TIME((mmult_stride   <<<block_count,512>>> (A,B,D, m, n, p)), "GPU STRIDE 32x512");
    TIME((mmult_stride   <<<block_count,1024>>> (A,B,D, m, n, p)), "GPU STRIDE 32x1024");

    // TIME((mmult_tiling   <<<(m / TILESIZE, p / TILESIZE),(TILESIZE, TILESIZE)>>> (A,B,C, m, n, p)), "GPU Tiled");

    cudaMemcpy(h_A, A, m * n * sizeof(mat_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B, B, n * p * sizeof(mat_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C, C, m * p * sizeof(mat_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_D, D, m * p * sizeof(mat_t), cudaMemcpyDeviceToHost);
    printf("\tError: %.3f\n", test(h_D, h_C, m, p));
    cudaError_t err = cudaDeviceSynchronize();

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    // printf("%f | %f \n%f | %f\n\n", h_A[0], h_A[1], h_A[2], h_A[3]);
    // printf("%f | %f \n%f | %f\n\n", h_B[0], h_B[1], h_B[2], h_B[3]);
    // printf("%f | %f \n%f | %f\n\n", h_C[0], h_C[1], h_C[2], h_C[3]);

    TIME(ompMatMult(h_A,h_B,h_D, m, n, p, 2), "OMP 2");
    printf("\tError: %.3f\n", test(h_D, h_C, m, p));
    TIME(ompMatMult(h_A,h_B,h_D, m, n, p, 4), "OMP 4");
    printf("\tError: %.3f\n", test(h_D, h_C, m, p));
    TIME(ompMatMult(h_A,h_B,h_D, m, n, p, 8), "OMP 8");
    printf("\tError: %.3f\n", test(h_D, h_C, m, p));
    
    TIME(pthread_matmult(h_A,h_B,h_D, m, n, p, 2), "PTHREAD 2");
    printf("\tError: %.3f\n", test(h_D, h_C, m, p));
    TIME(pthread_matmult(h_A,h_B,h_D, m, n, p, 4), "PTHREAD 4");
    printf("\tError: %.3f\n", test(h_D, h_C, m, p));
    TIME(pthread_matmult(h_A,h_B,h_D, m, n, p, 8), "PTHREAD 8");
    printf("\tError: %.3f\n", test(h_D, h_C, m, p));

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_D);

    return 0;
}