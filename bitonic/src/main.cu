// #include "reduction.h"
// #include "scan.h"
#include "cuda.h"

#include <string>

#define DEF_ARR_SIZE 0x10000000
#define THDS_PER_BLK 1024

typedef float T; // define our type we want to use here.

template <typename T>
/**
 * Randomize n entries in our array.
*/
void randomizeArr(T * arr, int n)
{
    for(int i = 0; i < n; i++)
        arr[i] =  2.0f * (static_cast<T> (rand()) / static_cast<T> (RAND_MAX)) - 1.0f;
}

template <typename T>
__global__ void bitonic_sort(T* arr, int n, int j, int k)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int l = (i^j); 
    if ( i < n && l > i){
            if (  ((i & k) == 0) && (arr[i] > arr[l])
               || ((i & k) != 0) && (arr[i] < arr[l]) )
               {
                    T temp = arr[i];
                    arr[i] = arr[l];
                    arr[l] = temp;
               }
    }
}

template <typename T>
void bitonic_sort_wrap(T * arr, int n)
{
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    for (int k = 2; k <= n; k *= 2) {
        for (int j = k/2; j > 0; j /= 2){
            bitonic_sort<<<CEIL_DIV(n, 1024), 1024, 0, stream>>>(arr, n, j, k);
        }
    }
    cudaStreamSynchronize(stream);
}

template <typename T>
__global__ void bitonic_sort_chunk(T* arr, int n, int j, int k, int chunks)
{
    int start = (threadIdx.x + blockIdx.x * blockDim.x )* chunks;
    for (int i = start; (i < (start+chunks)) && (i<n); i++)
    {
        int l = (i^j); 
        if (l > i){
                if (  ((i & k) == 0) && (arr[i] > arr[l])
                || ((i & k) != 0) && (arr[i] < arr[l]) )
                {
                        T temp = arr[i];
                        arr[i] = arr[l];
                        arr[l] = temp;
                }
        }
    }
}

template <typename T>
void bitonic_sort_wrap_chunk(T * arr, int n)
{
    int chunks = 2;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    for (int k = 2; k <= n; k *= 2) {
        for (int j = k/2; j > 0; j /= 2){
            bitonic_sort_chunk<<<CEIL_DIV(n, (1024*chunks)), 1024, 0, stream>>>(arr, n, j, k, chunks);
        }
    }
    cudaStreamSynchronize(stream);
}

int compare( const void* a, const void* b)
{
     int int_a = * ( (int*) a );
     int int_b = * ( (int*) b );

     if ( int_a == int_b ) return 0;
     else if ( int_a < int_b ) return -1;
     else return 1;
}


/**
 * Pass in the the following args:
 * num_sum: number of values to use for the reduction
*/
int main(int argc, char **argv)
{
   
    T* arr, *gpu_arr, *res, *gpu_res, *test_res;
    int n = DEF_ARR_SIZE;

    if (argc > 1) n = 1 << atoi(argv[1]);
    arr = (T*) malloc(sizeof(T) * n);
    // res = (T*) malloc(sizeof(T));
    // test_res = (T*) malloc(sizeof(T));
    cudaMalloc(&gpu_arr, sizeof(T) * n);
    // cudaMalloc(&gpu_res, sizeof(T));

    randomizeArr(arr, n);

    // Run expected result
    // reduce_omp(arr, test_res, n);
    // std::cout << "Expected value of: " << std::to_string(*test_res) << std::endl;

    // Actually run the main timable block
    // std::string msg = "Reduction (close) pass time";

    // // Warm up the Cache and libraries
    // Pass data to the GPU 
    cudaMemcpy(gpu_arr, arr, sizeof(T) * n, cudaMemcpyHostToDevice);
    // Do the sort 
    bitonic_sort_wrap(gpu_arr, n);
    // Get data from the GPU 
    cudaMemcpy(arr, gpu_arr, sizeof(T)*n, cudaMemcpyDeviceToHost); 

    randomizeArr(arr, n);
    printf("Input Array: \n");
    for (int i = 0; i < 5; i++)
    {
        printf("%f, ", arr[i]);
    }
    printf("...");
    for (int i = n-5; i < n; i++)
    {
        printf("%f, ", arr[i]);
    }
    printf("\n");

    cudaEvent_t start, stop;                              
    float elapsed=0; 

    cudaEventCreate(&start);                              
    cudaEventCreate(&stop);                              
    cudaEventRecord(start, 0);                         

    // Pass data to the GPU 
    cudaMemcpy(gpu_arr, arr, sizeof(T) * n, cudaMemcpyHostToDevice);
    // Do the sort 
    // bitonic_sort_wrap_chunk(gpu_arr, n);
    bitonic_sort_wrap(gpu_arr, n);

    // Get data from the GPU 
    cudaMemcpy(arr, gpu_arr, sizeof(T)*n, cudaMemcpyDeviceToHost); 

    cudaEventRecord(stop, 0);                      
    cudaEventSynchronize(stop);                    
    cudaEventElapsedTime(&elapsed, start, stop);   
    printf("Sort took %fms for %d elems\n", elapsed, n);
    
    printf("Output Array: \n");
    for (int i = 0; i < 5; i++)
    {
        printf("%f, ", arr[i]);
    }
    printf("...");
    for (int i = n-5; i < n; i++)
    {
        printf("%f, ", arr[i]);
    }
    printf("\n");

    randomizeArr(arr, n);

    cudaEventCreate(&start);                              
    cudaEventCreate(&stop);                              
    cudaEventRecord(start, 0);   

    qsort( arr, n, sizeof(T), compare );

    cudaEventRecord(stop, 0);                      
    cudaEventSynchronize(stop);                    
    cudaEventElapsedTime(&elapsed, start, stop);   
    printf("CPU Sort took %fms for %d elems\n", elapsed, n);
   

    cudaFree(gpu_arr);
    // cudaFree(gpu_res);
    free(arr);
    // free(res);

    return 0;
}

