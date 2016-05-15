//
// Created by 孙万捷 on 16/4/22.
//

#ifndef SUNPATHTRACER_SHADER_H
#define SUNPATHTRACER_SHADER_H

#define GLM_FORCE_INLINE
#include <glm/glm.hpp>

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

__device__ void diffuse_shading(const cudaScene& scene, SurfaceElement& se, curandState& rng, cudaRay* ray, glm::vec3* T)
{
    glm::vec3 nl = (glm::dot(se.normal, ray->dir) < 0.f) ? se.normal : -se.normal;
    ray->dir = cosine_weightd_sample_hemisphere(rng, nl);
    ray->orig = se.pt + ray->dir * se.rayEpsilon;

    *T *= scene.materials[se.matID].albedo;
}

__device__ void refractive_shading(const cudaScene& scene, SurfaceElement& se, curandState& rng, cudaRay* ray, glm::vec3* T)
{
    //eta = ni / no
    float eta = scene.materials[se.matID].ior;
    glm::vec3 nl = se.normal;
    //out going
    if(glm::dot(se.normal, ray->dir) > 0.f)
    {
        eta = 1.f / eta;
        nl = -se.normal;
    }
    else
    {
        if(scene.materials[se.matID].roughness < ROUGH_THRESHOLD)
            nl = sample_phong(rng, scene.materials[se.matID].roughness, nl);
    }
    eta = 1.f / eta;
    float cosin = -glm::dot(nl, ray->dir);
    float cost2 = 1.f - eta * eta * (1.f - cosin * cosin);

    if(cost2 < 0.f)
    {
        //*T *= scene.materials[se.matID].albedo;
        ray->dir = glm::reflect(ray->dir, nl);
    }
    else
    {
        glm::vec3 tdir = eta * ray->dir + nl * (eta * cosin - sqrtf(cost2));
        tdir = glm::normalize(tdir);

        float ni, no;
        if(glm::dot(ray->dir, se.normal) < 0.f)
        {
            ni = 1.f;
            no = scene.materials[se.matID].ior;
        }
        else
        {
            ni = scene.materials[se.matID].ior;
            no = 1.f;
        }
        float Pr = fresnel_schlick(ni, no, cosin);
        float Pt = 1.f - Pr;
        float P = 0.25f + 0.5f * Pr;

        if(curand_uniform(&rng) < P)
        {
            *T *= scene.materials[se.matID].albedo;
            *T *= (Pr / P);
            ray->dir = glm::reflect(ray->dir, nl);
        }
        else
        {
            float invEta = no / ni;
            *T *= scene.materials[se.matID].albedo * invEta * invEta;
            *T *= (Pt / (1.f - P));
            ray->dir = tdir;
        }

        ray->orig = se.pt + ray->dir * se.rayEpsilon;
    }

    /*bool into = glm::dot(ray->dir, se.normal) < 0.f;
    glm::vec3 n = se.normal;
    glm::vec3 nl = into ? n : -n;
    float nc = 1.f;
    float nt = 1.5f;
    float nnt = into ? nc / nt : nt / nc;
    float ddn = glm::dot(ray->dir, nl);
    float cos2t = 1.f - nnt * nnt * (1.f - ddn * ddn);

    if(cos2t < 0.f)
    {
        ray->dir = glm::reflect(ray->dir, nl);
        ray->orig = se.pt + nl * se.rayEpsilon;
    }
    else
    {
        glm::vec3 tdir = ray->dir * nnt;
        tdir -= n * ((into ? 1.f : -1.f) * (ddn * nnt + sqrtf(cos2t)));
        tdir = glm::normalize(tdir);

        float R0 = (nt - nc)*(nt - nc) / (nt + nc)*(nt + nc);
        float c = 1.f - (into ? -ddn : glm::dot(tdir, n));
        float Re = R0 + (1.f - R0) * c * c * c * c * c;
        float Tr = 1 - Re; // Transmission
        float P = .25f + .5f * Re;
        float RP = Re / P;
        float TP = Tr / (1.f - P);

        if(curand_uniform(&rng) < 0.25)
        {
            *T *= RP;
            ray->dir = glm::reflect(ray->dir, nl);
            ray->orig = se.pt + nl * se.rayEpsilon;
        }
        else
        {
            *T *= TP;
            ray->dir = tdir;
            ray->orig = se.pt - nl * se.rayEpsilon;
        }
    }*/
}

__device__ void glossy_shading(const cudaScene& scene, SurfaceElement& se, curandState& rng, cudaRay* ray, glm::vec3* T)
{
    glm::vec3 nl = (glm::dot(se.normal, ray->dir) < 0.f) ? se.normal : -se.normal;
    if(scene.materials[se.matID].roughness < ROUGH_THRESHOLD)
        ray->dir = sample_phong(rng, scene.materials[se.matID].roughness, glm::reflect(ray->dir, nl));
    else
        ray->dir = glm::reflect(ray->dir, nl);
    ray->orig = se.pt + ray->dir * se.rayEpsilon;

    *T *= scene.materials[se.matID].albedo;
}

__device__ void coat_shading(const cudaScene& scene, SurfaceElement& se, curandState& rng, cudaRay* ray, glm::vec3* T)
{
    glm::vec3 nl = (glm::dot(ray->dir, se.normal) < 0.f) ? se.normal : -se.normal;
    float cosin = -glm::dot(ray->dir, nl);
    float Pr = fresnel_schlick(1.f, scene.materials[se.matID].ior, cosin);
    float Pd = 1.f - Pr;
    float P = 0.25f + Pr * 0.5f;

    if(P < curand_uniform(&rng))
    {
        ray->dir = cosine_weightd_sample_hemisphere(rng, nl);
        *T *= scene.materials[se.matID].albedo;
        *T *= (Pd / (1.f - P));
    }
    else
    {
        if(scene.materials[se.matID].roughness < ROUGH_THRESHOLD)
            ray->dir = sample_phong(rng, scene.materials[se.matID].roughness, glm::reflect(ray->dir, nl));
        else
            ray->dir = glm::reflect(ray->dir, nl);
        *T *= scene.materials[se.matID].specular;
        *T *= (Pr / P);
    }
    ray->orig = se.pt + ray->dir * se.rayEpsilon;
}

#endif //SUNPATHTRACER_SHADER_H
