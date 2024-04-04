#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <pthread.h>


#define CORE 4
#define MAX 1000
#define fScale 40
float AMat[MAX][MAX];
float BMat[MAX][MAX];

void populate_matrices()
{
   for (int i = 0; i < MAX; i++)
      for (int j = 0; j < MAX; j++)
      {
         AMat[i][j] = (j * 0.43); //generate random float values at each spot in matrix
         BMat[i][j] = (1 + j * 0.59);
      }

   
}
pthread_t thread[CORE];

int add[MAX][MAX];

void* addMatrices(void* arg) {
   int core = (int) arg;
   // Each thread computes 1/3rd of matrix addition
   for (int i = core * MAX / CORE; i < (core + 1) * MAX / CORE; i++) {
      for (int j = 0; j < MAX; j++) {
         add[i][j] = AMat[i][j] + BMat[i][j];
      }
   }
   return NULL;
}

int main() {
   populate_matrices();
   struct timespec begin, end;
   double elapsed;
   clock_gettime(CLOCK_MONOTONIC, &begin);
   for (int i = 0; i < CORE; i++) {
      pthread_create(&thread[i], NULL, &addMatrices, (void*)i);
   }
   for (int i = 0; i < CORE; i++) {
      pthread_join(thread[i], NULL);
   }
   clock_gettime(CLOCK_MONOTONIC, &end);

   elapsed = end.tv_sec - begin.tv_sec;
   elapsed += (end.tv_nsec - begin.tv_nsec) / 1000000000.0;
   printf("Sucess!!!\nTime Taken %lf\n", elapsed);
   return 0;
}



