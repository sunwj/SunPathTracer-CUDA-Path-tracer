//
// Created by 孙万捷 on 16/2/6.
//

#ifndef SUNPATHTRACER_SHAPE_H
#define SUNPATHTRACER_SHAPE_H

#include <float.h>

#include <cuda_runtime.h>

#include "cuda_ray.h"
#include "cuda_material.h"

class cudaSphere
{
public:
    __host__ __device__ cudaSphere(const float3& _center, float _radius, unsigned int _material_id)
    {
        center = _center;
        radius = _radius;

        material_id = _material_id;
    }

    __device__ bool Intersect(const cudaRay& ray, float* t) const
    {
        // slove t ^ 2 * dir . dir + 2 * t * (o - c) . dir + (o - c) . (o - c) - radius ^ 2 = 0
        float3 L = ray.orig - center;
        //float a = dot(ray.dir, ray.dir);
        // ray.dir is normalized, so dir dot dir is cos(0) = 1
        float b = 2.f * dot(ray.dir, L);
        float c = dot(L, L) - radius * radius;

        float discr = b * b - 4.f * /*a **/ c;
        if(discr > 0.f)
        {
            discr = sqrtf(discr);

            constexpr float eps = 0.001f;
            if((*t = (-b - discr) /*/ (2.f * a)*/ * 0.5f) > eps)
                return true;
            else if((*t = (-b + discr) /*/ (2.f * a)*/ * 0.5f) > eps)
                return true;
            else
                return false;
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

    unsigned int material_id;
};

class cudaAABB
{
public:
    __host__ __device__ cudaAABB(const float3& _bMin, const float3& _bMax, unsigned int _material_id)
    {
        bMax = _bMax;
        bMin = _bMin;

        material_id = _material_id;
    }

    __device__ bool Intersect(const cudaRay& ray, float* t) const
    {
        float3 tmin = (bMin - ray.orig) / ray.dir;
        float3 tmax = (bMax - ray.orig) / ray.dir;

        float3 real_min = fminf(tmin, tmax);
        float3 real_max = fmaxf(tmin, tmax);

        float minmax = fminf(fminf(real_max.x, real_max.y), real_max.z);
        float maxmin = fmaxf(fmaxf(real_min.x, real_min.y), real_min.z);

        if(minmax >= maxmin)
        {
            constexpr float eps = 0.001f;
            if(maxmin > eps)
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
        constexpr float eps = 0.001f;

        if(fabsf(bMin.x - pt.x) < eps) normal = make_float3(-1.f, 0.f, 0.f);
        else if(fabsf(bMax.x - pt.x) < eps) normal = make_float3(1.f, 0.f, 0.f);
        else if(fabsf(bMin.y - pt.y) < eps) normal = make_float3(0.f, -1.f, 0.f);
        else if(fabsf(bMax.y - pt.y) < eps) normal = make_float3(0.f, 1.f, 0.f);
        else if(fabsf(bMin.z - pt.z) < eps) normal = make_float3(0.f, 0.f, -1.f);
        else normal = make_float3(0.f, 0.f, 1.f);

        return normal;
    }

public:
    float3 bMax, bMin;

    unsigned int material_id;
};

//todo: add triangle
class cudaTriangle
{
public:
public:
    
};

//todo: add plane
class cudaPlane
{
public:
    __host__ __device__ cudaPlane(const float3& _p, const float3& _normal, unsigned int _material_id)
    {
        p = _p;
        normal = _normal;

        material_id = _material_id;
    }

    __device__ bool Intersect(const cudaRay& ray, float* t) const
    {
        // t = ((p - ray.orig) . normal) / (ray.dir . normal)
        float denom = -dot(ray.dir, normal);
        if(denom > 1e-6)
        {
            constexpr float eps = 0.001f;
            *t = -dot(p - ray.orig, normal) / denom;
            return (*t > eps);
        }

        return false;
    }

    __device__ float3 GetNormal(const float3& pt) const
    {
        return normal;
    }
public:
    float3 p;
    float3 normal;
    unsigned int material_id;
};

#endif //SUNPATHTRACER_SHAPE_H
