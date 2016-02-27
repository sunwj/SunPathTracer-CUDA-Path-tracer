cmake_minimum_required(VERSION 3.3)
project(SunPathTracer)

# host compiler settings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# CUDA settings
find_package(CUDA QUIET REQUIRED)
list(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -use_fast_math -Xptxas -v -dlcm=cg)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
set(CUDA_INCLUDE_DIRS /usr/local/cuda/include/)
include_directories(${CUDA_INCLUDE_DIRS})

set(SOURCE_FILES main.cpp)
add_executable(SunPathTracer ${SOURCE_FILES})
target_link_libraries(SunPathTracer ${CUDA_LIBRARIES})