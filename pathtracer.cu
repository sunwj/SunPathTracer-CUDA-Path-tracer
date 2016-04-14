#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "helper_math.h"
#include "cuda_shape.h"
#include "cuda_ray.h"
#include "cuda_camera.h"

#define IMG_BLACK make_uchar4(0, 0, 0, 0)

__host__ __device__ unsigned int wangHash(unsigned int a)
{
    //http://raytracey.blogspot.com/2015/12/gpu-path-tracing-tutorial-2-interactive.html
    a = (a ^ 61) ^ (a >> 16);
    a = a + (a << 3);
    a = a ^ (a >> 4);
    a = a * 0x27d4eb2d;
    a = a ^ (a >> 15);

    return a;
}