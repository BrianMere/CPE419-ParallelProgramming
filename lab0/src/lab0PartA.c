#include <stdio.h>
#include <sys/sysinfo.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <pthread.h>


#define CORE 3
#define MAX 1000
#define NUM_THREADS 3

float AMat[MAX][MAX];
float BMat[MAX][MAX];

pthread_t thread[CORE];

float add[MAX][MAX];

void* addMatrices(void* arg) {
   int core = (int) arg;
   // Each thread computes 1/3rd of matrix addition
   for (int i = core * MAX / NUM_THREADS; i < (core + 1) * MAX / NUM_THREADS; i++) {
      for (int j = 0; j < MAX; j++) {
         add[i][j] = AMat[i][j] + BMat[i][j];
      }
   }
}

/**
 * Using the address of the given matrix, create a pseudo random matrix. 
*/
void initMatrices()
{
   for(int i = 0; i < MAX; ++i)
   {
      for(int j = 0; j < MAX; ++j)
      {
         AMat[i][j] = i * 0.41f + j * 1.41f;
         BMat[i][j] = i * 0.21f + j * 0.9f;
      }
   }
}

/**
 * Adds matrices A and B together sequentially and puts it into ret.
*/
void sequentialAddMat(float A[MAX][MAX], float B[MAX][MAX], float ret[MAX][MAX])
{
   for(int i = 0; i < MAX; ++i)
   {
      for(int j = 0; j < MAX; ++j)
      {
         ret[i][j] = A[i][j] + B[i][j];
      }
   }
}

/**
 * Checks if the two passed matrices are equal. 
*/
int test(float pos[MAX][MAX], float comp[MAX][MAX])
{
   for(int i = 0; i < MAX; ++i)
   {
      for(int j = 0; j < MAX; ++j)
      {
         if(pos[i][j] != comp[i][j])
         {
            return 0;
         }
      }
   }
   return 1;
}

void printSystemInfo() 
{
   printf("This system has %d processors configured and "
      "%d processors available.\n",
      get_nprocs_conf(), get_nprocs());
}

int main() {

   printSystemInfo();

   initMatrices();

   struct timespec begin, end;
   double elapsed;

   clock_gettime(CLOCK_MONOTONIC, &begin);

   // spawn threads to do work here

   for (int i = 0; i < CORE; i++) {
      pthread_create(&thread[i], NULL, &addMatrices, (void*)i);
   }
   for (int i = 0; i < CORE; i++) {
      pthread_join(thread[i], NULL);
   }

   clock_gettime(CLOCK_MONOTONIC, &end);

   elapsed = end.tv_sec - begin.tv_sec;
   elapsed += (end.tv_nsec - begin.tv_nsec) / 1000000000.0;

   printf("Time for parallel: %lf\n", elapsed);


   // Do the same for sequential
   float res[MAX][MAX];

   clock_gettime(CLOCK_MONOTONIC, &begin);

   sequentialAddMat(AMat, BMat, res);

   clock_gettime(CLOCK_MONOTONIC, &end);

   elapsed = end.tv_sec - begin.tv_sec;
   elapsed += (end.tv_nsec - begin.tv_nsec) / 1000000000.0;

   printf("Time for sequential: %lf\n", elapsed);

   if(!test(res, add))
   {
      printf("However, the matrices aren't the same\n");
   }

   printf("End Program!!!\n");
}