#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "helper_math.h"
#include "cuda_shape.h"
#include "cuda_ray.h"
#include "cuda_camera.h"
#include "cuda_scene.h"

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

__global__ void testSimpleScene(uchar4* img)
{
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int idy = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int offset = idy * 640 + idx;
    img[offset] = IMG_BLACK;

    //curandState rng;
    //curand_init(0, 0, 0, &rng);
    //cudaRay pRay;
    //scene.camera.GenerateRay(idx, idy, rng, &pRay);
//
    //float t = FLT_MAX;
    //for(auto i = 0; i < scene.num_aabb_boxes; ++i)
    //{
    //    const cudaAABB& box = scene.aabb_boxes[i];
    //    float ttmp;
    //    if(box.Intersect(pRay, &ttmp) && (ttmp < t))
    //    {
    //        t = ttmp;
    //        float3 n = box.GetNormal(pRay.PointOnRay(t));
    //        float costerm = fmaxf(0.f, dot(n, make_float3(0.f, 1.f, 0.f)));
    //        float3 c = scene.materials[box.material_id].reflectance * costerm;
    //        img[offset] = make_uchar4(c.x * 255, c.y * 255, c.z * 255, 0);
    //    }
    //}
}

extern "C" void test(uchar4* img)
{
    dim3 blockSize(16, 16);
    dim3 gridSize(640 / blockSize.x, 480 / blockSize.y);

    testSimpleScene<<<gridSize, blockSize>>>(img);
    checkCudaErrors(cudaDeviceSynchronize());
}