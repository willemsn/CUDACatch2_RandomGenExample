#include "CUDA_RandomKernel.h"

__global__ void kernelComputation(float *prngValues, float *result)
{
    int id =  (blockDim.x * blockIdx.x) + threadIdx.x;
    result[id] = (prngValues[id] * 2.0f - 1.0f) * 10.0f;
}

void runKernel(int n, float *prngValues, float *result)
{
    int BLOCK_SIZE = 1024;
    int GRID_SIZE = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernelComputation<<<GRID_SIZE, BLOCK_SIZE>>>(prngValues, result);

}


