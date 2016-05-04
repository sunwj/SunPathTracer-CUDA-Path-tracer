cmake_minimum_required(VERSION 3.3)
project(SunPathTracer)

# host compiler settings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# CUDA settings
find_package(CUDA QUIET REQUIRED)
list(APPEND CUDA_NVCC_FLAGS --maxrregcount=32 --compiler-options -fno-strict-aliasing -use_fast_math -Xptxas -v)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
set(CUDA_INCLUDE_DIRS /usr/local/cuda/include/)
include_directories(${CUDA_INCLUDE_DIRS})

# OpenGL and GLFW
find_package(OpenGL REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_search_module(GLFW REQUIRED glfw3)
include_directories(${OPENGL_INCLUDE_DIR})
include_directories(${GLFW_INCLUDE_DIRS})

set(CUDA_SOURCES pathtracer.cu)
set(SOURCE_FILES main.cpp Scene.cpp ObjMesh.cpp Transformation.cpp BVH.cpp utils.cpp)

cuda_compile(CUDA_OBJS ${CUDA_SOURCES})

add_executable(SunPathTracer ${SOURCE_FILES} ${CUDA_OBJS})
target_link_libraries(SunPathTracer ${CUDA_LIBRARIES} ${OPENGL_LIBRARIES} ${GLFW_LIBRARIES})