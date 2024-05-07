typedef float mat_t;

void ompMatMult(mat_t *A, mat_t *B, mat_t *C, int m, int n, int p);
void pthread_matmult(mat_t *A, mat_t *B, mat_t *C, int m, int n, int p);
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
