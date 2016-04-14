//
// Created by 孙万捷 on 16/2/6.
//

#ifndef SUNPATHTRACER_SHAPE_H
#define SUNPATHTRACER_SHAPE_H

#include <float.h>

#include <cuda_runtime.h>

#include "cuda_ray.h"

class cudaSphere
{
public:
    __host__ __device__ cudaSphere(const float3& _center, float _radius)
    {
        center = _center;
        radius = _radius;
    }

    __device__ bool Intersect(const cudaRay& ray, float* t) const
    {
        // slove t ^ 2 * dir . dir + 2 * t * (o - c) . dir + (o - c) . (o - c) - radius ^ 2 = 0
        float3 L = ray.orig - center;
        //float a = dot(ray.dir, ray.dir);
        // ray.dir is normalized, so dir dot dir is cos(0) = 1
        float b = 2.f * dot(ray.dir, L);
        float c = dot(L, L) - radius * radius;

        float discr = b * b - 4.f * /*a*/ * c;
        if(discr > 0.f)
        {
            discr = sqrtf(discr);
            *t = (-b - discr) /*/ (2.f * a)*/ * 0.5f;

            // check valid interval
            constexpr float eps = 0.001f;
            if(*t < eps)
                *t = (-b + discr) /*/ (2.f * a)*/ * 0.5f;
            if(*t < eps || *t > (FLT_MAX - 1.f))
                return false;

            return true;
        }

        return false;
    }

    __device__ float3 GetNormal(const float3& pt) const
    {
        return normalize(pt - center);
    }

public:
    float3 center;
    float radius;
};

class cudaAABB
{
public:
    __host__ __device__ cudaAABB(const float3& _bMax, const float3& _bMin)
    {
        bMax = _bMax;
        bMin = _bMin;
    }

    __device__ bool Intersect(const cudaRay& ray, float* t) const
    {
        float3 tmin = (bMin - ray.orig) / ray.dir;
        float3 tmax = (bMax - ray.orig) / ray.dir;

        float3 real_min = fminf(tmin, tmax);
        fllat3 real_max = fmaxf(tmin, tmax);

        float minmax = fminf(fminf(real_max.x, real_max.y), real_max.z);
        float maxmin = fmaxf(fmaxf(real_min.x, real_min.y), real_min.z);

        if(minmax >= maxmin)
        {
            if(maxmin > 0.001f)
            {
                *t = maxmin;
                return true;
            }
        }

        return false;
    }

    __device__ float3 GetNormal(const float3& pt) const
    {
        float3 normal;
        if(fabsf(bMin.x - pt.x) < 0.001f) normal = make_float3(-1.f, 0.f, 0.f);
        else if(fabsf(bMax.x - pt.x) < 0.001f) normal = make_float3(1.f, 0.f, 0.f);
        else if(fabsf(bMin.y - pt.y) < 0.001f) normal = make_float3(0.f, -1.f, 0.f);
        else if(fabsf(bMax.y - pt.y) < 0.001f) normal = make_float3(0.f, 1.f, 0.f);
        else if(fabsf(bMin.z - pt.z) < 0.001f) normal = make_float3(0.f, 0.f, -1.f);
        else normal = make_float3(0.f, 0.f, 1.f);

        return normal;
    }

public:
    float3 bMax, bMin;
};

#endif //SUNPATHTRACER_SHAPE_H
