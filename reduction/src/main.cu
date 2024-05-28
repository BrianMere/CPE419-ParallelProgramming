#include "reduction.h"
#include "scan.h"
#include "cuda.h"

#include <string>

#define DEF_ARR_SIZE 0x200000
#define THDS_PER_BLK 1024

template <typename T>
/**
 * Randomize n entries in our array.
*/
void randomizeArr(T* arr, int n)
{
    for(int i = 0; i < n; i++)
        // arr[i] = i;
        arr[i] = 1;
}

/**
 * Pass in the the following args:
 * num_sum: number of values to use for the reduction
*/
int main(int argc, char **argv)
{
    typedef int T; // define our type we want to use here.
    T* arr, *gpu_arr, *res, *gpu_res, *test_res;
    int n = DEF_ARR_SIZE;

    cudaDeviceProp prop;   
    cudaGetDeviceProperties( &prop, 0);
    printf("Device: %s\n%d threads per block\n%d per MP\n", prop.name, prop.maxThreadsPerBlock, prop.maxThreadsPerMultiProcessor);
    printf("%d total multiprocessors\n", prop.multiProcessorCount);
    int block_count = prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor / prop.maxThreadsPerBlock;
    printf("Using %d blocks\n", block_count);
    // printf("l1 Cache size: %ld\n", prop.sharedMemPerBlock);
    printf("\n");

    if (argc > 1) n = atoi(argv[1]);
    arr = (T*) malloc(sizeof(T) * n);
    res = (T*) malloc(sizeof(T));
    test_res = (T*) malloc(sizeof(T));
    cudaMalloc(&gpu_arr, sizeof(T) * n);
    cudaMalloc(&gpu_res, sizeof(T));

    randomizeArr(arr, n);

    // Run expected result
    cudaEvent_t start, stop;                              
    float elapsed=0;                                       
    cudaEventCreate(&start);                              
    cudaEventCreate(&stop);                              
    cudaEventRecord(start, 0);   
    reduce_omp(arr, test_res, n);
    cudaEventRecord(stop, 0);                      
    cudaEventSynchronize(stop);                    
    cudaEventElapsedTime(&elapsed, start, stop);   
    std::cout << "OMP Implementation: " << std::to_string(elapsed) << " ms" << std::endl; 

    std::cout << "Expected value of: " << std::to_string(*test_res) << std::endl;

    // Actually run the main timable block
    std::string msg = "Reduction (close) pass time";

    // Warm up the Cache and libraries
        // Pass data to the GPU 
        CUDA_ERR_CHK(cudaMemcpy(gpu_arr, arr, sizeof(T) * n, cudaMemcpyHostToDevice));

        // Do the reduction 
        reduceAdjacent(gpu_arr, gpu_res, n, CEIL_DIV(n, THDS_PER_BLK), THDS_PER_BLK);

        // Get data from the GPU (only want to get the res data)
        CUDA_ERR_CHK(cudaMemcpy(res, gpu_res, sizeof(T), cudaMemcpyDeviceToHost)); 

    randomizeArr(arr, n);                            
    elapsed=0;                                       
    cudaEventCreate(&start);                              
    cudaEventCreate(&stop);                              
    cudaEventRecord(start, 0);                         

    // Pass data to the GPU 
    CUDA_ERR_CHK(cudaMemcpy(gpu_arr, arr, sizeof(T) * n, cudaMemcpyHostToDevice));

    // Do the reduction 
    reduceAdjacent(gpu_arr, gpu_res, n, CEIL_DIV(n, THDS_PER_BLK), THDS_PER_BLK);

    // Get data from the GPU (only want to get the res data)
    CUDA_ERR_CHK(cudaMemcpy(res, gpu_res, sizeof(T), cudaMemcpyDeviceToHost)); 

    cudaEventRecord(stop, 0);                      
    cudaEventSynchronize(stop);                    
    cudaEventElapsedTime(&elapsed, start, stop);   
    std::cout << msg << ": " << std::to_string(elapsed) << " ms" << std::endl; 

    // Output results and free data.
    std::cout << "Result of reduction: " << std::to_string(*res) << std::endl;

    randomizeArr(arr, n);

    // Actually run the main timable block
    msg = "Reduction (far) pass time";
    // Warm up libraries again
        // Pass data to the GPU 
        CUDA_ERR_CHK(cudaMemcpy(gpu_arr, arr, sizeof(T) * n, cudaMemcpyHostToDevice));

        // Do the reduction 
        reduceSpread(gpu_arr, gpu_res, n, CEIL_DIV(n, THDS_PER_BLK), THDS_PER_BLK);

        // Get data from the GPU (only want to get the res data)
        CUDA_ERR_CHK(cudaMemcpy(res, gpu_res, sizeof(T), cudaMemcpyDeviceToHost)); 

    randomizeArr(arr, n);
    elapsed=0;                                       
    cudaEventCreate(&start);                              
    cudaEventCreate(&stop);                              
    cudaEventRecord(start, 0);                         

    // Pass data to the GPU 
    CUDA_ERR_CHK(cudaMemcpy(gpu_arr, arr, sizeof(T) * n, cudaMemcpyHostToDevice));

    // Do the reduction 
    reduceSpread(gpu_arr, gpu_res, n, CEIL_DIV(n, THDS_PER_BLK), THDS_PER_BLK);

    // Get data from the GPU (only want to get the res data)
    CUDA_ERR_CHK(cudaMemcpy(res, gpu_res, sizeof(T), cudaMemcpyDeviceToHost)); 

    cudaEventRecord(stop, 0);                      
    cudaEventSynchronize(stop);                    
    cudaEventElapsedTime(&elapsed, start, stop);   
    std::cout << msg << ": " << std::to_string(elapsed) << " ms" << std::endl; 

    randomizeArr(arr, n);                            
    elapsed=0;                                       
    cudaEventCreate(&start);                              
    cudaEventCreate(&stop);                              
    cudaEventRecord(start, 0);  
    // Output results and free data.
    std::cout << "Result of reduction: " << std::to_string(*res) << std::endl;
    cudaMemset(gpu_res, 0, sizeof(T));
    CUDA_ERR_CHK(cudaMemcpy(gpu_arr, arr, sizeof(T) * n, cudaMemcpyHostToDevice));
    // Do the reduction 
    segmentScan(gpu_arr, gpu_res, n, CEIL_DIV(n, THDS_PER_BLK), THDS_PER_BLK);

    // Get data from the GPU (only want to get the res data)
    CUDA_ERR_CHK(cudaMemcpy(res, gpu_res, sizeof(T), cudaMemcpyDeviceToHost)); 
    CUDA_ERR_CHK(cudaMemcpy(arr, gpu_arr, sizeof(T) * n, cudaMemcpyDeviceToHost));
    cudaEventRecord(stop, 0);                      
    cudaEventSynchronize(stop);                    
    cudaEventElapsedTime(&elapsed, start, stop);  

    std::cout << "Result of Scan: " << std::to_string(*res) << std::endl;
    for (int i = 0; i < 5; i++)
    {
        printf("%d, ", arr[i]);
    }
    printf("... ");
    for (int i = n-5; i < n; i++)
    {
        printf("%d, ", arr[i]);
    }
    printf("\n");

    std::cout << "Scan Time: " << std::to_string(elapsed) << " ms" << std::endl;


    cudaFree(gpu_arr);
    cudaFree(gpu_res);
    free(arr);
    free(res);

    return 0;

}