CMAKE_MINIMUM_REQUIRED (VERSION 3.18)
CMAKE_POLICY(SET CMP0074 NEW)

PROJECT(CUDACATCH2 VERSION 1.0.0)

# ----------------------------------------------------------
# Detect COMPILER
#  This may be useful for helping us determine the level of C++
#  support available.  For instance on GCC 4.4.7, -std=c++11 is not
#  available, but -std=c++0x is available.  Newer compilers
#  can handle the -std=c++11, -std=c++14, etc...
# ----------------------------------------------------------
include(CheckCXXCompilerFlag)
MESSAGE(STATUS "Compiling with ${CMAKE_CXX_COMPILER_ID}, Version: ${CMAKE_CXX_COMPILER_VERSION}")
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)

if(COMPILER_SUPPORTS_CXX11)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

  #
  # This code used C++ 11 and 14 features so enable these on CXX Flags
  #
  set(CMAKE_CXX_STANDARD 11)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  
elseif(COMPILER_SUPPORTS_CXX0X)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
  message(FATAL_ERROR "Compiler ${CMAKE_CXX_COMPILER} has no C++11 support.")
endif()

# ----------------------------------------------------------
# CUDA
# ----------------------------------------------------------
SET(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-arch=sm_52;-rdc=true;" )

FIND_PACKAGE(CUDA)
IF(CUDA_FOUND)
  MESSAGE(STATUS "CUDA Libraries: ${CUDA_LIBRARIES}")
  SET(HAS_CUDA_SUPPORT ON CACHE BOOL "Determines if CUDA/GPU functionality is compiled into the code base" )
  
  INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
  LINK_DIRECTORIES(${CUDA_SDK_ROOT_DIR}/lib64)
ELSE(CUDA_FOUND)
  #
  # CUDA was not found, so disable CUDA/GPU support
  #
  SET(HAS_CUDA_SUPPORT OFF CACHE BOOL "Determines if CUDA/GPU functionality is compiled into the code base" FORCE)
  
  MESSAGE(WARNING "CUDA is required for compiling GPU accelerated computations into this project.  Please install CUDA or re-run cmake with -i to specify the CUDA directories if you need these features.")
  MESSAGE(WARNING "GPU and related accelerations will be disable for this build.")
ENDIF(CUDA_FOUND)

# ----------------------------------------------------------
# CUDA and OPTIX
# if CUDA 10.2 or higher is found, then OptiX support can be enabled, so search for it.
# ----------------------------------------------------------
if(HAS_CUDA_SUPPORT)
  ADD_DEFINITIONS(-DHAS_CUDA)
  MESSAGE(STATUS "Using CUDA Version: ${CUDA_VERSION}")
else(HAS_CUDA_SUPPORT)
  MESSAGE(STATUS "Disabling CUDA Support.")
  SET(HAS_CUDA_SUPPORT OFF FORCE BOOL "Determines if CUDA/GPU functionality is compiled into the code base")
  REMOVE_DEFINITIONS(-DHAS_CUDA)
endif(HAS_CUDA_SUPPORT)

# ----------------------------------------------------------
#  Unit Testing:
#
# This "unit test" is only a simple test to show the system
# working. It simply returns a true value to indicate that a null
# program has ended OK. I pass on this test really just means the
# cmake system constructed the build environment OK.
# ----------------------------------------------------------
enable_testing() # once
include(CTest)

Include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.0.1 # or a later release
  )

FetchContent_MakeAvailable(Catch2)

add_executable(unit_test_example_t00 t00.cpp)
target_link_libraries(unit_test_example_t00 Catch2::Catch2WithMain)
add_test(NAME unit_test::unit_test_example_t00 COMMAND unit_test_example_t00)

cuda_add_executable(unit_test_example_t00_CUDA t00_CUDA.cpp)
target_link_libraries(unit_test_example_t00_CUDA ${CUDA_LIBRARIES})
target_link_libraries(unit_test_example_t00_CUDA ${CUDA_curand_LIBRARY})
target_link_libraries(unit_test_example_t00_CUDA Catch2::Catch2WithMain)
add_test(NAME unit_test::unit_test_example_t00_CUDA COMMAND unit_test_example_t00_CUDA)

cuda_add_executable(unit_test_example_t01_CUDA
  CUDA_RandomKernel.cu CUDA_RandomKernel.h
  t01_CUDA.cpp
  )
target_link_libraries(unit_test_example_t01_CUDA ${CUDA_LIBRARIES})
target_link_libraries(unit_test_example_t01_CUDA ${CUDA_curand_LIBRARY})
target_link_libraries(unit_test_example_t01_CUDA Catch2::Catch2WithMain)
add_test(NAME unit_test::unit_test_example_t01_CUDA COMMAND unit_test_example_t01_CUDA)

