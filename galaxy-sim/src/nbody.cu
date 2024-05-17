#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"

#define SOFTENING 1e-9f
#define CEIL_DIV(A, B) ((A+B-1) / B) 
#define NITERS 10

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

  Body *p, *hp;

  hp = (Body * ) malloc(sizeof(Body) * nBodies);

  cudaMalloc(&p, sizeof(Body) * nBodies);

  randomizeBodies((float*)hp, 6*nBodies); // Init pos / vel data

  double totalTime = 0.0;

  cudaEventCreate(&start);                        
  cudaEventCreate(&stop);                         
  cudaEventRecord(start, 0);

  for (int iter = 0; iter <= NITERS; iter++) {

    // Time Calcs
    double tElapsed = 0;

    (cudaMemcpy(p, hp, sizeof(Body) * nBodies, cudaMemcpyHostToDevice));
    (bodyForce<<<CEIL_DIV(nBodies, 256), 256>>>(p, dt, nBodies));
    (updatePosition<<< CEIL_DIV(nBodies, 256), 256>>>(p, dt, nBodies));
    (cudaMemcpy(hp, p, sizeof(Body) * nBodies, cudaMemcpyDeviceToHost));
    
  }

  cudaEventRecord(stop, 0);                       
  cudaEventSynchronize (stop);                    
  cudaEventElapsedTime(&elapsed, start, stop);    
  printf("Time Without Streams: %f ms\n", elapsed); 

  // make streams
  cudaEventCreate(&start);                        
  cudaEventCreate(&stop);                         
  cudaEventRecord(start, 0);
  cudaStream_t stream[NITERS];
  for( int iter = 0; iter < NITERS; iter++) 
    cudaStreamCreate(&stream[iter]);

  for (int iter = 0; iter <= NITERS; iter++) {

    // Time Calcs
    double tElapsed = 0;

    (cudaMemcpy(p, hp, sizeof(Body) * nBodies, cudaMemcpyHostToDevice));
    (bodyForce<<<CEIL_DIV(nBodies, 256), 256, 0, stream[iter] >>>(p, dt, nBodies));
    (updatePosition<<< CEIL_DIV(nBodies, 256), 256, 0, stream[iter] >>>(p, dt, nBodies));
    (cudaMemcpy(hp, p, sizeof(Body) * nBodies, cudaMemcpyDeviceToHost));

    
  }

  for( int iter = 0; iter < NITERS; iter++) 
    cudaStreamDestroy(stream[iter]);

  cudaEventRecord(stop, 0);                       
  cudaEventSynchronize (stop);                    
  cudaEventElapsedTime(&elapsed, start, stop);    
  printf("Time With Streams: %f ms\n", elapsed);   

  

  cudaFree(p);
  free(hp);
}
