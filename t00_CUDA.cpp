#include <catch2/catch_test_macros.hpp>
#include <iostream>

#include <cuda.h>
#include <curand.h>

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
