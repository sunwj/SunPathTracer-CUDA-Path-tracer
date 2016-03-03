//
// Created by 孙万捷 on 16/2/27.
//

#ifndef SUNPATHTRACER_ONB_H
#define SUNPATHTRACER_ONB_H

#include <cuda_runtime.h>
#include "helper_math.h"

class cudaONB
{
public:
    __device__ cudaONB(const float3& _w)
    {
        InitFromW(_w);
    }

    __device__ cudaONB(const float3& _v, const float3& _w)
    {
        InitFromVW(_v, _w);
    }

    __device__ void InitFromW(const float3& _w)
    {
        w = _w;
        if(fabsf(w.x) > fabsf(w.y))
        {
            float invLength = rsqrtf(w.x * w.x + w.z * w.z);
            v = make_float3(-w.z * invLength, 0.f, w.x * invLength);
        }
        else
        {
            float  invLength = rsqrtf(w.y * w.y + w.z * w.z);
            v = make_float3(0.f, w.z * invLength, -w.y * invLength);
        }
        u = cross(v, w);
    }

    __device__ void InitFromVW(const float3& _v, const float3& _w)
    {
        w = _w;
        u = cross(_v, w);
        v = cross(w, u);
    }
public:
    float3 u, v, w;
};

#endif //SUNPATHTRACER_ONB_H
