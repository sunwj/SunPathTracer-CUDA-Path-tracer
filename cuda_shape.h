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
        //float a = dot(ray.dir, ray.dir);
        // ray.dir is normalized, so dir dot dir is cos(0) = 1
        float a = 1.f;
        float b = 2.f * dot(ray.dir, L);
        float c = dot(L, L) - radius * radius;

        float discr = b * b - 4.f * a * c;
        if(discr > 0.f)
        {
            discr = sqrtf(discr);
            *t = (-b - discr) / (2.f * a);

            // check valid interval
            if(*t < 0.f)
                *t = (-b + discr) / (2.f * a);
            if(*t < 0.f || *t > (FLT_MAX - 1.f))
                return false;

            return true;
        }

        return false;
    }

private:
    float3 center;
    float radius;
};

#endif //SUNPATHTRACER_SHAPE_H
