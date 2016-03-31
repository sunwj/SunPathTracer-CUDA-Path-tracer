//
// Created by 孙万捷 on 16/2/6.
//

#ifndef SUNPATHTRACER_CAMERA_H
#define SUNPATHTRACER_CAMERA_H

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "helper_math.h"
#include "cuda_ray.h"

class cudaCamera
{
public:
    __host__ __device__ cudaCamera(const float3& _pos, const float3& _u, const float3& _v, const float3& _w, float fov, unsigned int _imageW,
                                   unsigned int _imageH)
    {
        pos = pos;
        u = _u;
        v = _v;
        w = _w;
        imageW = _imageW;
        imageH = _imageH;
        aspectRatio = (float)imageW / (float)imageH;
        tanFovOverTwo = tanf(fov * 0.5f * M_PI / 180.f);
    }

    __host__ __device__ cudaCamera(const float3& pos, const float3& target, const float3& up, float fov, unsigned int _imageW, unsigned int _imageH)
    {
        pos = pos;
        w = normalize(pos - target);
        u = cross(v, w);
        v = cross(w, u);
        imageW = _imageW;
        imageH = _imageH;
        aspectRatio = (float)imageW / (float)imageH;
        tanFovOverTwo = tanf(fov * 0.5f * M_PI / 180.f);
    }

    // TODO: depth of field
    // TODO: jittered sampling
    __device__ void GenerateRay(unsigned int x, unsigned int y, curandState& rng, cudaRay* ray) const
    {
        float nx = 2.f * ((x + curand_uniform(rng)) / (imageW - 1.f)) - 1.f;
        float ny = 2.f * ((y + curand_uniform(rng)) / (imageH - 1.f)) - 1.f;

        nx = nx * aspectRatio * tanFovOverTwo;
        ny = ny * tanFovOverTwo;

        ray->orig = pos;
        ray->dir = normalize(nx * u + ny * v - w);
    }

public:
    unsigned int imageW, imageH;
    float aspectRatio;
    float tanFovOverTwo;
    float3 pos;
    float3 u, v, w;
};

#endif //SUNPATHTRACER_CAMERA_H
