#ifndef MAT_H
#define MAT_H
typedef float mat_t;

void ompMatMult(mat_t *A, mat_t *B, mat_t *C, int m, int n, int p, int n_threads);
void pthread_matmult(mat_t *A, mat_t *B, mat_t *C, int m, int n, int p, int n_threads);
void* pthreadMultMatrices(void* arg);
typedef struct matrix
{
    int core_num;
    int max_cores;
    mat_t * A;
    mat_t * B;
    mat_t * C;
    int n;
    int m;
    int p;
} matrix_in_t;

mat_t test(mat_t * pos, mat_t * comp, int M, int N);

#endif