//
// Created by 孙万捷 on 16/4/22.
//

#ifndef SUNPATHTRACER_SHADER_H
#define SUNPATHTRACER_SHADER_H

#include <cuda_runtime.h>

#include "cuda_scene.h"
#include "sampling.h"
#include "kernel_globals.h"

#define ROUGH_THRESHOLD 9999.9f

__inline__ __device__ float fresnel_schlick(float ni, float no, float cosin)
{
    float R0 = (ni - no) * (ni - no) / ((ni + no) * (ni + no));
    float c = 1.f - cosin;
    return R0 + (1.f - R0) * c * c * c * c * c;
}

__device__ void diffuse_shading(const cudaScene& scene, SurfaceElement& se, curandState& rng, cudaRay* ray, float3* T)
{
    float3 nl = (dot(se.normal, ray->dir) < 0.f) ? se.normal : -se.normal;
    ray->orig = se.pt + nl * se.rayEpsilon;
    ray->dir = cosine_weightd_sample_hemisphere(rng, nl);

    *T *= scene.materials[se.matID].albedo;
}

__device__ void refractive_shading(const cudaScene& scene, SurfaceElement& se, curandState& rng, cudaRay* ray, float3* T)
{
    float eta = scene.materials[se.matID].ior;
    float3 nl = se.normal;
    if(dot(se.normal, ray->dir) > 0.f)
    {
        eta = 1.f / eta;
        nl = -se.normal;
    }
    eta = 1.f / eta;
    if(scene.materials[se.matID].roughness < ROUGH_THRESHOLD)
        nl = sample_phong(rng, scene.materials[se.matID].roughness, nl);
    float cosin = -dot(nl, ray->dir);
    float cost2 = 1.f - eta * eta * (1.f - cosin * cosin);

    if(cost2 < 0.f)
    {
        *T *= scene.materials[se.matID].albedo;
        ray->dir = reflect(ray->dir, nl);
        ray->orig = se.pt + nl * se.rayEpsilon;
    }
    else
    {
        float3 tdir = eta * ray->dir + nl * (eta * cosin - sqrtf(cost2));
        tdir = normalize(tdir);

        float n1 = (cosin < 0.f) ? 1.f : scene.materials[se.matID].ior;
        float n2 = (cosin < 0.f) ? scene.materials[se.matID].ior : 1.f;
        float Pr = fresnel_schlick(n1, n2, cosin);
        float Pt = 1.f - Pr;
        float P = 0.25f + 0.5f * Pr;

        if(curand_uniform(&rng) < P)
        {
            *T *= scene.materials[se.matID].albedo;
            *T *= (Pr / P);
            ray->dir = reflect(ray->dir, nl);
            ray->orig = se.pt + nl * se.rayEpsilon;
        }
        else
        {
            *T *= scene.materials[se.matID].albedo;
            *T *= (Pt / (1.f - P));
            ray->dir = tdir;
            ray->orig = se.pt - nl * se.rayEpsilon;
        }
    }
}

__device__ void glossy_shading(const cudaScene& scene, SurfaceElement& se, curandState& rng, cudaRay* ray, float3* T)
{
    float3 nl = (dot(se.normal, ray->dir) < 0.f) ? se.normal : -se.normal;
    ray->orig = se.pt + nl * se.rayEpsilon;
    if(scene.materials[se.matID].roughness < ROUGH_THRESHOLD)
        ray->dir = sample_phong(rng, scene.materials[se.matID].roughness, reflect(ray->dir, nl));
    else
        ray->dir = reflect(ray->dir, nl);

    *T *= scene.materials[se.matID].albedo;
}

__device__ void coat_shading(const cudaScene& scene, SurfaceElement& se, curandState& rng, cudaRay* ray, float3* T)
{
    float3 nl = (dot(ray->dir, se.normal) < 0.f) ? se.normal : -se.normal;
    float cosin = -dot(ray->dir, nl);
    float Pr = fresnel_schlick(1.f, scene.materials[se.matID].ior, cosin);
    float Pd = 1.f - Pr;
    float P = 0.25f + Pr * 0.5f;

    ray->orig = se.pt + nl * se.rayEpsilon;
    if(P < curand_uniform(&rng))
    {
        ray->dir = cosine_weightd_sample_hemisphere(rng, nl);
        *T *= scene.materials[se.matID].albedo;
        *T *= (Pd / (1.f - P));
    }
    else
    {
        ray->dir = reflect(ray->dir, nl);
        *T *= scene.materials[se.matID].specular;
        *T *= (Pr / P);
    }
}

#endif //SUNPATHTRACER_SHADER_H
