#include <catch2/catch_test_macros.hpp>
#include <iostream>

#include "CUDA_RandomKernel.h"

TEST_CASE("A test to show Catch2 is working for Unit Tests")
{
  int sum = 2 + 2;
  
  curandGenerator_t gen;
    
  /* Create pseudo-random number generator */
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);

  REQUIRE(sum == 4);
}

TEST_CASE("Another test to show Catch2 is working for Unit Tests")
{
  int sum = 2 + 2;
  int sub = 2 - 2;
  std::cout << "test" << std::endl;

  curandGenerator_t gen;
    
  /* Create pseudo-random number generator */
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  
  REQUIRE(sum == 4);
  REQUIRE(sub == 0);
}

TEST_CASE("CUDA Random Gen")
{
    size_t i;

    curandGenerator_t gen;
    float *devPRNVals, *hostData, *devResults, *hostResults;
    
    /* Create pseudo-random number generator */
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    
    /* Set seed */
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

    // start with 15k par
    int numParticles = 15000;
    
    for (int r=0; r<15; r++) {

        /* Allocate numParticle * 3 floats on host */
        int n = numParticles * 3;

        hostResults = (float *)calloc(n, sizeof(float));
    
        /* Allocate n floats on device */
	cudaMalloc((void **)&devPRNVals, n*sizeof(float));
	cudaMalloc((void **)&devResults, n*sizeof(float));

        /* Generate n random floats on device */
	curandGenerateUniform(gen, devPRNVals, n);

        // use random values in kernel
        runKernel(n, devPRNVals, devResults);
    
        /* Copy device memory to host */
        // CUDA_CALL(cudaMemcpy(hostData, devPRNVals, n * sizeof(float),
        // cudaMemcpyDeviceToHost));
        cudaMemcpy(hostResults, devResults, n * sizeof(float), cudaMemcpyDeviceToHost);
    
        /* Show result */
        double avgVal = 0.0;
        for(i = 0; i < n; i++) {
            // std::cout << hostResults[i] << std::endl;
            avgVal += hostResults[i];
        }
        avgVal /= double(n);

	double eps = 1.0e-1;
        // std::cout << "AVERAGE -> " << avgVal << std::endl;
        
	REQUIRE( avgVal < eps );
        REQUIRE( avgVal > -eps );

        numParticles *= 2;

        cudaFree(devPRNVals);
        cudaFree(devResults);
        
        free(hostResults);
    }

    /* Cleanup */
    curandDestroyGenerator(gen);
}

