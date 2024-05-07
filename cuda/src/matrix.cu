#include "matrix.h"
#include <pthread.h>
#include "omp.h"

#define CORE 8

void ompMatMult(mat_t *A, mat_t *B, mat_t *C, int m, int n, int p)
{
    int num_elems = m * p;
    #pragma omp parallel for
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

void pthread_matmult(mat_t *A, mat_t *B, mat_t *C, int m, int n, int p)
{
    matrix_in_t in[CORE];
    pthread_t thread[CORE];
    for (int i = 0; i < CORE; i++)
    {
        in[i].A = A;
        in[i].B = B;
        in[i].C = C;
        in[i].m = m;
        in[i].n = n;
        in[i].p = p;
        in[i].max_cores = CORE;
        in[i].core_num = i;
    }

    for (int i = 0; i < CORE; i++) {
        pthread_create(&thread[i], NULL, &pthreadMultMatrices, (void*)&in[i]);
      }
      for (int i = 0; i < CORE; i++) {
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