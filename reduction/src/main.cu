#include "reduction.h"
#include "cuda.h"

#include <string>

#define DEF_ARR_SIZE 0x1000

template <typename T>
/**
 * Randomize n entries in our array.
*/
void randomizeArr(T* arr, int n)
{
    for(int i = 0; i < n; i++)
        arr[i] = i;
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

    if (argc > 1) n = atoi(argv[1]);
    arr = (T*) malloc(sizeof(T) * n);
    res = (T*) malloc(sizeof(T));
    test_res = (T*) malloc(sizeof(T));
    cudaMalloc(&gpu_arr, sizeof(T) * n);
    cudaMalloc(&gpu_res, sizeof(T));

    randomizeArr(arr, n);

    // Run expected result
    reduce_omp(arr, test_res, n);
    std::cout << "Expected value of: " << std::to_string(*test_res) << std::endl;

    // Actually run the main timable block
    std::string msg = "Reduction (close) pass time";

    // Warm up the Cache and libraries
        // Pass data to the GPU 
        CUDA_ERR_CHK(cudaMemcpy(gpu_arr, arr, sizeof(T) * n, cudaMemcpyHostToDevice));

        // Do the reduction 
        reduceAdjacent<<<CEIL_DIV(n, 256), 256>>>(gpu_arr, gpu_res, n);

        // Get data from the GPU (only want to get the res data)
        CUDA_ERR_CHK(cudaMemcpy(res, gpu_res, sizeof(T), cudaMemcpyDeviceToHost)); 

    cudaEvent_t start, stop;                              
    float elapsed=0;                                       
    cudaEventCreate(&start);                              
    cudaEventCreate(&stop);                              
    cudaEventRecord(start, 0);                         

    // Pass data to the GPU 
    CUDA_ERR_CHK(cudaMemcpy(gpu_arr, arr, sizeof(T) * n, cudaMemcpyHostToDevice));

    // Do the reduction 
    reduceAdjacent<<<CEIL_DIV(n, 256), 256>>>(gpu_arr, gpu_res, n);

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
        reduceSpread<<<CEIL_DIV(n, 256), 256>>>(gpu_arr, gpu_res, n);

        // Get data from the GPU (only want to get the res data)
        CUDA_ERR_CHK(cudaMemcpy(res, gpu_res, sizeof(T), cudaMemcpyDeviceToHost)); 

    elapsed=0;                                       
    cudaEventCreate(&start);                              
    cudaEventCreate(&stop);                              
    cudaEventRecord(start, 0);                         

    // Pass data to the GPU 
    CUDA_ERR_CHK(cudaMemcpy(gpu_arr, arr, sizeof(T) * n, cudaMemcpyHostToDevice));

    // Do the reduction 
    reduceSpread<<<CEIL_DIV(n, 256), 256>>>(gpu_arr, gpu_res, n);

    // Get data from the GPU (only want to get the res data)
    CUDA_ERR_CHK(cudaMemcpy(res, gpu_res, sizeof(T), cudaMemcpyDeviceToHost)); 

    cudaEventRecord(stop, 0);                      
    cudaEventSynchronize(stop);                    
    cudaEventElapsedTime(&elapsed, start, stop);   
    std::cout << msg << ": " << std::to_string(elapsed) << " ms" << std::endl; 

    // Output results and free data.
    std::cout << "Result of reduction: " << std::to_string(*res) << std::endl;


    cudaFree(gpu_arr);
    cudaFree(gpu_res);
    free(arr);
    free(res);

    return 0;

}