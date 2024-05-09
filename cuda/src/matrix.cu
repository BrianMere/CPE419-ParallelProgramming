#include "matrix.h"
#include <pthread.h>
#include <math.h>
#include "omp.h"

#define CORE 16

void ompMatMult(mat_t *A, mat_t *B, mat_t *C, int m, int n, int p, int n_threads)
{
    int num_elems = m * p;
    #pragma omp parallel for num_threads(n_threads)
    for(int i = 0; i < num_elems; i++)
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

void pthread_matmult(mat_t *A, mat_t *B, mat_t *C, int m, int n, int p, int n_threads)
{
    matrix_in_t in[CORE];
    pthread_t thread[CORE];
    for (int i = 0; i < n_threads; i++)
    {
        in[i].A = A;
        in[i].B = B;
        in[i].C = C;
        in[i].m = m;
        in[i].n = n;
        in[i].p = p;
        in[i].max_cores = n_threads;
        in[i].core_num = i;
    }

    for (int i = 0; i < n_threads; i++) {
        pthread_create(&thread[i], NULL, &pthreadMultMatrices, (void*)&in[i]);
      }
      for (int i = 0; i < n_threads; i++) {
        pthread_join(thread[i], NULL);
      }
}

void* pthreadMultMatrices(void* arg) {
    matrix_in_t * in = (matrix_in_t *) arg;
    int i = in->core_num;
    int total_threads = in->max_cores;
    int m = in->m;
    int n = in->n;
    int p = in->p;
    mat_t * A = in->A;
    mat_t * B = in->B;
    mat_t * C = in->C;
    int num_elems = in->m * in->p;


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
    return NULL;
}

/**
 * Checks if the two passed matrices are equal. 
*/
mat_t test(mat_t * pos, mat_t * comp, int M, int N)
{
    mat_t sum = 0;
    #pragma omp parallel for reduction(+: sum)
    for(int i = 0; i < N * M; ++i)
    {
        sum += abs(pos[i] - comp[i]);
    }
   return sum;
}