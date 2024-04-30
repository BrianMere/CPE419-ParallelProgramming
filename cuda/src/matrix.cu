#include "matrix.h"

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






void pthread_matmult()
{
    for (int i = 0; i < CORE; i++) {
        pthread_create(&thread[i], NULL, &addMatrices, (void*)i);
      }
      for (int i = 0; i < CORE; i++) {
         pthread_join(thread[i], NULL);
      }
}

void* addMatrices(void* arg) {
   int core = (int) arg;
   // Each thread computes 1/3rd of matrix addition
   for (int i = core * MAX / num_threads; i < (core + 1) * MAX / num_threads && i < MAX; i++) {
      for (int j = 0; j < MAX; j++) {
         add[i][j] = AMat[i][j] + BMat[i][j];
      }
   }
}