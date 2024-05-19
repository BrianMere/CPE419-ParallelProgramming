#include "reduction.h"
#include "cuda.h"

#include <string>

#define DEF_ARR_SIZE 2048

template <typename T>
/**
 * Randomize n entries in our array.
*/
void randomizeArr(T* arr, int n)
{
    for(int i = 0; i < n; i++)
        arr[i] = i + 1;
}

/**
 * Pass in the the following args:
 * num_sum: number of values to use for the reduction
*/
int main(int argc, char **argv)
{
    typedef float T; // define our type we want to use here.
    T* arr, *gpu_arr, *res, *gpu_res;
    int n = DEF_ARR_SIZE;

    if (argc > 1) n = atoi(argv[1]);
    arr = (T*) malloc(sizeof(T) * n);
    res = (T*) malloc(sizeof(T));
    cudaMalloc(&gpu_arr, sizeof(T) * n);
    cudaMalloc(&gpu_res, sizeof(T));
    randomizeArr(arr, n);


    // Actually run the main timable block
    std::string msg = "Reduction pass time";

    cudaEvent_t start, stop;                              
    float elapsed=0;                                       
    cudaEventCreate(&start);                              
    cudaEventCreate(&stop);                              
    cudaEventRecord(start, 0);                         

    // Pass data to the GPU 
    CUDA_ERR_CHK(cudaMemcpy(gpu_arr, arr, sizeof(T) * n, cudaMemcpyHostToDevice));

    // Do the reduction 
    reduceOperation<<<CEIL_DIV(n, 256), 256>>>(gpu_arr, gpu_res, n);

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