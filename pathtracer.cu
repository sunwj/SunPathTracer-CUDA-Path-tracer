#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "helper_math.h"
#include "cuda_shape.h"
#include "cuda_ray.h"
#include "cuda_camera.h"
#include "cuda_scene.h"
#include "tonemapping.h"
#include "sampling.h"
#include "render_parameters.h"

auto constexpr WIDTH = 640;
auto constexpr HEIGHT = 480;

#define IMG_BLACK make_uchar4(0, 0, 0, 0)

unsigned int wangHash(unsigned int a)
{
    //http://raytracey.blogspot.com/2015/12/gpu-path-tracing-tutorial-2-interactive.html
    a = (a ^ 61) ^ (a >> 16);
    a = a + (a << 3);
    a = a ^ (a >> 4);
    a = a * 0x27d4eb2d;
    a = a ^ (a >> 15);

    return a;
}

__inline__ __device__ void running_estimate(float3& acc_buffer, const float3& curr_est, unsigned int N)
{
    acc_buffer += (curr_est - acc_buffer) / (N + 1.f);
}

struct HitInfo
{
    bool intersected = false;
    float t = FLT_MAX;
    float3 normal;
    unsigned int matID;
};

__global__ void testSimpleScene(uchar4* img, cudaScene scene, RenderParameters params, unsigned int hashed_N)
{
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int idy = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int offset = idy * scene.camera.imageW + idx;
    img[offset] = IMG_BLACK;

    curandState rng;
    curand_init(hashed_N + offset, 0, 0, &rng);
    cudaRay pRay;
    scene.camera.GenerateRay(idx, idy, rng, &pRay);

    float3 L = make_float3(0.f, 0.f, 0.f);
    float3 T = make_float3(1.f, 1.f, 1.f);

    for(auto k = 0; k < 5; ++k)
    {
        HitInfo hit;
        //find nearest intersection
        for(auto i = 0; i < scene.num_aabb_boxes; ++i)
        {
            const cudaAABB& box = scene.aabb_boxes[i];
            float ttmp;
            if(box.Intersect(pRay, &ttmp) && (ttmp < hit.t))
            {
                hit.intersected = true;
                hit.t = ttmp;
                hit.normal = box.GetNormal(pRay.PointOnRay(hit.t));
                hit.matID = box.material_id;
            }
        }

        for(auto i = 0; i < scene.num_spheres; ++i)
        {
            const cudaSphere& sphere = scene.spheres[i];
            float ttmp;
            if(sphere.Intersect(pRay, &ttmp) && (ttmp < hit.t))
            {
                hit.intersected = true;
                hit.t = ttmp;
                hit.normal = sphere.GetNormal(pRay.PointOnRay(hit.t));
                hit.matID = sphere.material_id;
            }
        }

        for(auto i = 0; i < scene.num_planes; ++i)
        {
            const cudaPlane& plane = scene.planes[i];
            float ttmp;
            if(plane.Intersect(pRay, &ttmp) && (ttmp < hit.t))
            {
                hit.intersected = true;
                hit.t = ttmp;
                hit.normal = plane.GetNormal(pRay.PointOnRay(hit.t));
                hit.matID = plane.material_id;
            }
        }

        if(!hit.intersected)
        {
            L = make_float3(0.f);
            break;
        }

        pRay.orig = pRay.PointOnRay(hit.t);
        pRay.dir = cosine_weightd_sample_hemisphere(rng, hit.normal);

        L += scene.materials[hit.matID].emition * T;
        T *= scene.materials[hit.matID].albedo;
    }

    running_estimate(params.hdr_buffer[offset], L, params.iteration_count);
    L = reinhard_tone_mapping(params.hdr_buffer[offset], 0.6f);
    img[offset] = make_uchar4(fabsf(L.x) * 255, fabsf(L.y) * 255, fabsf(L.z) * 255, 0);
}

extern "C" void test(uchar4* img, cudaScene& scene, RenderParameters& params)
{
    dim3 blockSize(16, 16);
    dim3 gridSize(640 / blockSize.x, 480 / blockSize.y);

    if(params.iteration_count == 0)
    {
        checkCudaErrors(cudaMemset(params.hdr_buffer, 0, sizeof(float3) * WIDTH * HEIGHT));
    }

    testSimpleScene<<<gridSize, blockSize>>>(img, scene, params, wangHash(params.iteration_count));
    checkCudaErrors(cudaDeviceSynchronize());
}