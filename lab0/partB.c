#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <pthread.h>

#define CORE 7
#define AColBRow 1000 //num columns of A and num rows of B
#define ARow 1000 //num Rows of A
#define BCol 1000

float AMat[ARow*AColBRow];
float BMat[AColBRow*BCol];
float CMat[ARow*BCol]; //product matrix dimensions

pthread_t thread[CORE];



void populate_matrices()
{
    for (int i = 0; i < ARow; i++)
        for (int j = 0; j < AColBRow; j++)
            AMat[i*AColBRow+j] = (j * 0.43);
    for (int i = 0; i < AColBRow; i++)
        for (int j = 0; j < BCol; j++)
            BMat[i*BCol+j] = (1 + j * 0.59);
}

void *multiply_matrices(void* arg)
{
    int core = (int)arg;
    int row, col, k;

    float Pvalue=0;

    for (row= core * ARow / CORE; row < (core + 1) * ARow / CORE; row++){

        for(col=0; col<BCol; col++) {

            Pvalue=0;

            for(k=0; k<BCol; k++){

                Pvalue+=AMat[row*AColBRow+k]*BMat[k*BCol+col];

            }

            CMat[row*BCol+col]=Pvalue;

        }
    }
      
}

int main()
{
    populate_matrices();
    struct timespec begin, end;
    double elapsed;
    clock_gettime(CLOCK_MONOTONIC, &begin);
    for (int i = 0; i < CORE; i++) {
        pthread_create(&thread[i], NULL, &multiply_matrices, (void*)i);
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