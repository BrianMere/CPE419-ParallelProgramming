#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"

#define SOFTENING 1e-9f
#define CEIL_DIV(A, B) ((A+B-1) / B) 
#define NITERS 10

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

typedef struct { float x, y, z, vx, vy, vz, Fx, Fy, Fz; } Body;

void randomizeBodies(float *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  }
}


__global__ void zeroForce(Body *p, float dt, int n) 
{
   int i = threadIdx.x + blockIdx.x * blockDim.x;
   if (i < n)
   {
    p[i].Fx = 0.0;
    p[i].Fy = 0.0;
    p[i].Fz = 0.0;
   }
}

__global__ void bodyForce(Body *p, float dt, int n) {
    
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;; 
    if(i < n && j < n)
    {
      // float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

      float dx = p[j].x - p[i].x;
      float dy = p[j].y - p[i].y;
      float dz = p[j].z - p[i].z;
      float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
      float invDist = 1.0f / sqrtf(distSqr);
      float invDist3 = invDist * invDist * invDist;


      // atomicAdd(&p[i].Fx, dx * invDist3);
      // atomicAdd(&p[i].Fy, dy * invDist3);
      // atomicAdd(&p[i].Fz, dz * invDist3);
      // p[i].Fx += dx * invDist3; 
      // p[i].Fy += dy * invDist3; 
      // p[i].Fz += dz * invDist3;  
    }
  //}
}

__global__ void updatePosition(Body* p, float dt, int n)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) // integrate position
  {
      p[i].vx += dt*p[i].Fx; 
      p[i].vy += dt*p[i].Fy; 
      p[i].vz += dt*p[i].Fz;
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
  // Body *pA[NITERS], *hpA[NITERS];
  // for (int i = 0; i < NITERS; i++)
  // {
  //   hp = (Body * ) malloc(sizeof(Body) * nBodies);
    
  //   cudaMalloc(&p, sizeof(Body) * nBodies);
  //   printf("HP IS %p, P is %p\n", hp, p);
  //   randomizeBodies((float*)hp, 9*nBodies); // Init pos / vel data
  //   pA[i] = p;
  //   hpA[i] = hp;
  // }
  hp = (Body * ) malloc(sizeof(Body) * nBodies);
  cudaMalloc(&p, sizeof(Body) * nBodies);
  randomizeBodies((float*)hp, 9*nBodies); // Init pos / vel data

  cudaEventCreate(&start);                        
  cudaEventCreate(&stop);                         
  cudaEventRecord(start, 0);

  for (int iter = 0; iter <= NITERS; iter++) {

    // p = pA[iter];
    // hp = hpA[iter];

    (cudaMemcpy(p, hp, sizeof(Body) * nBodies, cudaMemcpyHostToDevice));
    zeroForce<<< CEIL_DIV(nBodies, 256), 256 >>>(p, dt, nBodies);
    int nblocks = CEIL_DIV(nBodies, 32);
    dim3 block(nblocks,nblocks);
    dim3 threads(32,32);
    (bodyForce<<<block, threads>>>(p, dt, nBodies));
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

    // p = pA[iter];
    // hp = hpA[iter];

    (cudaMemcpy(p, hp, sizeof(Body) * nBodies, cudaMemcpyHostToDevice));
    zeroForce<<< CEIL_DIV(nBodies, 256), 256, 0, stream[iter] >>>(p, dt, nBodies);
    int nblocks = CEIL_DIV(nBodies, 32);
    dim3 block(nblocks,nblocks);
    dim3 threads(32,32);
    (bodyForce<<<block, threads, 0, stream[iter] >>>(p, dt, nBodies));
    (updatePosition<<< CEIL_DIV(nBodies, 256), 256, 0, stream[iter] >>>(p, dt, nBodies));
    (cudaMemcpy(hp, p, sizeof(Body) * nBodies, cudaMemcpyDeviceToHost));
    
  }

  for( int iter = 0; iter < NITERS; iter++) 
    cudaStreamDestroy(stream[iter]);

  cudaEventRecord(stop, 0);                       
  cudaEventSynchronize (stop);                    
  cudaEventElapsedTime(&elapsed, start, stop);    
  printf("Time With Streams: %f ms\n", elapsed);   

  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());

  cudaFree(p);
  free(hp);
}
