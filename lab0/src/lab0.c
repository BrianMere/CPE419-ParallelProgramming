#include <stdio.h>
#include <sys/sysinfo.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <pthread.h>


#define CORE 3
#define MAX 3

int AMat[MAX][MAX] = {{10, 20, 30},
   {40, 50, 60},
   {70, 80, 50}
};
int BMat[MAX][MAX] = {{80, 60, 20},
   {30, 20, 15},
   {10, 14, 35}
};

pthread_t thread[CORE];

int add[MAX][MAX];

void* addMatrices(void* arg) {
   int core = (int) arg;
   // Each thread computes 1/3rd of matrix addition
   for (int i = core * MAX / 3; i < (core + 1) * MAX / 3; i++) {
      for (int j = 0; j < MAX; j++) {
         add[i][j] = AMat[i][j] + BMat[i][j];
      }
   }
}

void printSystemInfo() 
{
   printf("This system has %d processors configured and "
      "%d processors available.\n",
      get_nprocs_conf(), get_nprocs());
}

main() {

   printSystemInfo();

   for (int i = 0; i < CORE; i++) {
      pthread_create(&thread[i], NULL, &addMatrices, (void*)i);
   }
   for (int i = 0; i < CORE; i++) {
      pthread_join(thread[i], NULL);
   }
   printf("Success!!!");
}