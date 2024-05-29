#include <stdlib.h>
#include <omp.h>
#include <iostream>
#include <cstdint>
#include <math.h>


#define DEF_ARR_SIZE 0x200000

template <typename T>
/**
 * Randomize n entries in our array.
*/
void randomizeArr(T* arr, int n)
{
    for(int i = 0; i < n; i++)
        // arr[i] = i;
        arr[i] = sqrt((float) (i + 0.5f));
}


template<typename T>
void reduce_omp(T* d_in, T* d_res, unsigned int n)
{
    T sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for (uint64_t i = 0; i < n; i++)
    {
        sum += d_in[i];
    }
    *d_res = sum; 
}

int main(int argc, char **argv) 
{
    typedef float T; // define our type we want to use here.
    T* arr, *res;
    int n = DEF_ARR_SIZE;
    if (argc > 1) n = atoi(argv[1]);

    arr = (T*) malloc(sizeof(T) * n);
    res = (T*) malloc(sizeof(T));

    double begin = omp_get_wtime();
    reduce_omp(arr, res, n);
    double end = omp_get_wtime();
    std::cout << "OMP Implementation: " << std::to_string((end - begin) * 1000) << " ms" << std::endl; 
}