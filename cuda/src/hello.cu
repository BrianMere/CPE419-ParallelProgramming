#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include "matrix.h"

#define THD_PER_BLK 256
#define CEIL_DIV(A, B) ((A+B-1) / B) 



static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

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
    printf("%d total multiprocessors\n\n", prop.multiProcessorCount);
    int block_count = prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor / prop.maxThreadsPerBlock;
    printf("Using %d blocks\n", block_count);
    
    unsigned int n = 1000;
    unsigned int m = 1000;
    unsigned int p = 1000;
    
    mat_t * A, * B, * C;
    mat_t * h_A, * h_B, * h_C;

    h_A = (mat_t *) malloc(m * n * sizeof(mat_t));
    h_B = (mat_t *) malloc(n * p * sizeof(mat_t));
    h_C = (mat_t *) malloc(m * p * sizeof(mat_t));

    cudaMalloc ((void**)&A, m * n * sizeof(mat_t));
    cudaMalloc ((void**)&B, n * p * sizeof(mat_t));
    cudaMalloc ((void**)&C, m * p * sizeof(mat_t));

    float elapsed=0;
    cudaEvent_t start, stop;

    init    <<<CEIL_DIV(m * n, THD_PER_BLK), THD_PER_BLK>>> (A,m * n);
    init    <<<CEIL_DIV(n * p, THD_PER_BLK), THD_PER_BLK>>> (B,n * p);
    // sum     <<<CEIL_DIV(n, THD_PER_BLK), THD_PER_BLK>>> (A,B,C,n);

    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR( cudaEventRecord(start, 0));
    mmult_nostride   <<<CEIL_DIV(m * p, THD_PER_BLK), THD_PER_BLK>>> (A,B,C, n, m, p);
    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize (stop) );
    HANDLE_ERROR(cudaEventElapsedTime(&elapsed, start, stop) );
    printf("GPU NoStride: %f ms\n", elapsed);
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));

    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR( cudaEventRecord(start, 0));
    mmult_stride   <<<block_count,prop.maxThreadsPerBlock>>> (A,B,C, n, m, p);
    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize (stop) );
    HANDLE_ERROR(cudaEventElapsedTime(&elapsed, start, stop) );
    printf("GPU Stride: %f ms\n", elapsed);

    cudaMemcpy(h_A, A, n * sizeof(A), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B, B, n * sizeof(B), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C, C, n * sizeof(C), cudaMemcpyDeviceToHost);

    cudaError_t err = cudaDeviceSynchronize();

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    // printf("%f | %f \n%f | %f\n\n", h_A[0], h_A[1], h_A[2], h_A[3]);
    // printf("%f | %f \n%f | %f\n\n", h_B[0], h_B[1], h_B[2], h_B[3]);
    // printf("%f | %f \n%f | %f\n\n", h_C[0], h_C[1], h_C[2], h_C[3]);

    free(h_A);
    free(h_B);
    free(h_C);

    

    return 0;
}