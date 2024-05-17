#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"

#define SOFTENING 1e-9f
#define CEIL_DIV(A, B) ((A+B-1) / B) 

typedef struct { float x, y, z, vx, vy, vz; } Body;

void randomizeBodies(float *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  }
}

__global__ void bodyForce(Body *p, float dt, int n) {
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;

  // #pragma omp parallel for schedule(dynamic)
  //for (int i = 0; i < n; i++) { 

    int i = tidx; 
    if(i < n)
    {
      float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

      for (int j = 0; j < n; j++) {
        float dx = p[j].x - p[i].x;
        float dy = p[j].y - p[i].y;
        float dz = p[j].z - p[i].z;
        float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
        float invDist = 1.0f / sqrtf(distSqr);
        float invDist3 = invDist * invDist * invDist;

        Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
      }

      p[i].vx += dt*Fx; p[i].vy += dt*Fy; p[i].vz += dt*Fz;
    }
  //}
}

__global__ void updatePosition(Body* p, float dt, int n)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) // integrate position
  {
      p[i].x += p[i].vx*dt;
      p[i].y += p[i].vy*dt;
      p[i].z += p[i].vz*dt;
  }
}

/**
 * Pass in the number of bodies you want to this command call. 
*/
int main(const int argc, const char** argv) {
  
  int nBodies = 30000;
  if (argc > 1) nBodies = atoi(argv[1]);

  const float dt = 0.01f; // time step
  const int nIters = 10;  // simulation iterations

  Body *p;

  cudaMallocManaged(&p, sizeof(Body) * nBodies, cudaMemAttachGlobal);

  randomizeBodies((float*)p, 6*nBodies); // Init pos / vel data

  double totalTime = 0.0;

  for (int iter = 1; iter <= nIters; iter++) {

    double tElapsed = 0;
    TIME((bodyForce<<<CEIL_DIV(nBodies, 256), 256 >>>(p, dt, nBodies)), "Body Force");
    tElapsed += elapsed;
    TIME((updatePosition<<< CEIL_DIV(nBodies, 256), 256>>>(p, dt, nBodies)), "Update Position");
    tElapsed += elapsed;

    if (iter > 1) { // First iter is warm up
      totalTime += tElapsed; 
    }

    #ifndef SHMOO
        printf("Iteration %d: %.3f ms\n", iter, tElapsed);
    #endif
  }
  double avgTime = totalTime / (double)(nIters-1); 

  #ifdef SHMOO
    printf("%d, %0.3f\n", nBodies, 1e-9 * nBodies * nBodies / avgTime);
  #else
    printf("Average rate for iterations 2 through %d: %.3f steps per second.\n",
          nIters,(float) totalTime / 10.0f);
    printf("%d Bodies: average %0.3f Billion Interactions / second\n", nBodies, 1e-9 * nBodies * nBodies / avgTime);
  #endif 

  cudaFree(p);
}