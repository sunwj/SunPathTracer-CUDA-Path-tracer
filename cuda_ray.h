//
// Created by 孙万捷 on 16/2/6.
//

#ifndef SUNPATHTRACER_RAY_H
#define SUNPATHTRACER_RAY_H

#include <cuda_runtime.h>

class cudaRay
{
public:
    __device__ cudaRay() {}
    __device__ cudaRay(const float3& orig, const float3& dir)
    {
        this->orig = orig;
        this->dir = dir;
    }

    __device__ float3 PointOnRay(float t) const
    {
        return orig + t * dir;
    }
public:
    float3 orig;
    float3 dir;
};

#endif //SUNPATHTRACER_RAY_H
