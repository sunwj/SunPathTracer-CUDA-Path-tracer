#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda_shape.h"
#include "cuda_camera.h"
#include "cuda_scene.h"
#include "tonemapping.h"
#include "render_parameters.h"
#include "kernel_globals.h"
#include "shader.h"

auto constexpr WIDTH = 640;
auto constexpr HEIGHT = 480;

__global__ void testSimpleScene(glm::u8vec4* img, cudaScene scene, RenderParameters params, unsigned int hashed_N)
{
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int idy = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int offset = idy * scene.camera.imageW + idx;
    img[offset] = IMG_BLACK;

    curandState rng;
    curand_init(hashed_N + offset, 0, 0, &rng);
    cudaRay ray;
    scene.camera.GenerateRay(idx, idy, rng, &ray);

    glm::vec3 L = glm::vec3(0.f, 0.f, 0.f);
    glm::vec3 T = glm::vec3(1.f, 1.f, 1.f);

    SurfaceElement se;
    for(auto k = 0; k < params.rayDepth; ++k)
    {
        if(!scene_intersect(scene, ray, se))
        {
            L += T * scene.env_light.GetEnvRadiance(ray.dir, 0.6);
            break;
        }

        L += T * scene.materials[se.matID].emition;

        switch(scene.materials[se.matID].bsdf_type)
        {
            case BSDF_DIFFUSE:
                diffuse_shading(scene, se, rng, &ray, &T);
                break;
            case BSDF_GLASS:
                refractive_shading(scene, se, rng, &ray, &T);
                break;
            case BSDF_GLOSSY:
                glossy_shading(scene, se, rng, &ray, &T);
                break;
            case BSDF_PLASTIC:
                coat_shading(scene, se, rng, &ray, &T);
                break;
            default:
                break;
        }

        //russian roulette
        if(k >= 3)
        {
            float illum = illuminance(T);
            if(curand_uniform(&rng) > illum) break;
            T /= illum;
        }
    }

    running_estimate(params.hdr_buffer[offset], L, params.iteration_count);
    L = reinhard_tone_mapping(params.hdr_buffer[offset], params.exposure);
    img[offset] = glm::u8vec4(fabsf(L.x) * 255, fabsf(L.y) * 255, fabsf(L.z) * 255, 0);
}

extern "C" void test(glm::u8vec4* img, cudaScene& scene, RenderParameters& params)
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