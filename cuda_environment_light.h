//
// Created by 孙万捷 on 16/5/4.
//

#ifndef SUNPATHTRACER_CUDA_ENVIRONMENT_LIGHT_H
#define SUNPATHTRACER_CUDA_ENVIRONMENT_LIGHT_H

#include <string>

#define GLM_FORCE_INLINE
#include <glm/glm.hpp>

#include <cuda_runtime.h>

#ifdef __CUDACC__
#define TEX2D_FLOAT4(texobj, x, y) (tex2D<float4>(texobj, x, y))
#else
#define TEX2D_FLOAT4(texobj, x, y) (make_float4(0.f))
#endif

class cudaEnvironmentLight
{
public:
    __host__ __device__ cudaEnvironmentLight() {}
    __host__ __device__ cudaEnvironmentLight(cudaTextureObject_t _tex):
        tex(_tex)
    {}

    __device__ glm::vec3 GetEnvRadiance(const glm::vec2& texcoord)
    {
        auto val = TEX2D_FLOAT4(tex, texcoord.x, texcoord.y);
        return tex ? glm::vec3(val.x, val.y, val.z) : glm::vec3(0.f);
    }

    __device__ glm::vec3 GetEnvRadiance(const glm::vec3& dir, float u_offset = 0.f, float v_offset = 0.f)
    {
        float theta = acosf(dir.y);
        float phi = atan2f(dir.x, dir.z);
        phi = phi < 0.f ? phi + 2.f * M_PI : phi;
        float u = phi * 0.5f * M_1_PI;
        float v = theta * M_1_PI;

        auto val = TEX2D_FLOAT4(tex, u + u_offset, v + v_offset);
        return tex ? glm::vec3(val.x, val.y, val.z) : glm::vec3(0.f);
    }

public:
    cudaTextureObject_t tex;
};

#endif //SUNPATHTRACER_CUDA_ENVIRONMENT_LIGHT_H
