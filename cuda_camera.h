//
// Created by 孙万捷 on 16/2/6.
//

#ifndef SUNPATHTRACER_CAMERA_H
#define SUNPATHTRACER_CAMERA_H

#include <cuda_runtime.h>

#include "cuda_ray.h"

class cudaCamera
{
public:
    __host__ __device__ cudaCamera(const float3& _pos, const float3& _u, const float3& _v, const float3& _w, unsigned int _filmW,
                                   unsigned int _filmH)
    {
        pos = pos;
        u = _u;
        v = _v;
        w = _w;
        filmW = _filmW;
        filmH = _filmH;
    }

    __device__ void GenerateRay(unsigned int x, unsigned int y, cudaRay* ray)
    {

    }

public:
    unsigned int filmW, filmH;
    float3 pos;
    float3 u, v, w;
};

#endif //SUNPATHTRACER_CAMERA_H
