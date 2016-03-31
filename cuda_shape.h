//
// Created by 孙万捷 on 16/2/6.
//

#ifndef SUNPATHTRACER_SHAPE_H
#define SUNPATHTRACER_SHAPE_H

#include <cuda_runtime.h>

#include "cuda_ray.h"

class cudaShape
{
public:
    __device__ virtual bool Intersect(const cudaRay& ray, float* t) const = 0;
};

class cudaSphere: public cudaShape
{
public:
    __host__ __device__ void cudaSphere(const float3& _center, float _radius)
    {
        center = _center;
        radius = _radius;
    }

    __device__ virtual bool Intersect(const cudaRay& ray, float* t) const
    {
        float3 L = ray.orig - center;
        float a = dot(ray.orig, ray.dir);
        float b = 2.f * dot(ray.dir, L);
        float c = dot(L, L) - radius * radius;

        float discr = b * b - 4.f * a * c;
        if(discr < 0)
            return false;
        else
        {
            float q = (b > 0.f) ? (-0.5f * (b + sqrtf(discr))) : (-0.5f * (b - sqrtf(discr)));
            float t1 = q / a;
            float t2 = c / q;
            *t = (t1 > t2) ? t1 : t2;

            return true;
        }
    }

private:
    float3 center;
    float radius;
};

#endif //SUNPATHTRACER_SHAPE_H
