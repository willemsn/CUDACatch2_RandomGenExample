#pragma once

#include <cuda.h>
#include <curand.h>

void runKernel(int n, float *prngValues, float *result);
