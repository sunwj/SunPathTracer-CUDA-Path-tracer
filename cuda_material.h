//
// Created by 孙万捷 on 16/4/12.
//

#ifndef SUNPATHTRACER_CUDA_MATERIAL_H
#define SUNPATHTRACER_CUDA_MATERIAL_H

#include <cuda_runtime.h>
#include "helper_math.h"

class cudaMaterial
{
public:
    cudaMaterial()
    {
        reflectance = make_float3(0.8f);
        emittance = make_float3(0.f);
        ior = 1.f;
    }

public:
    float3 reflectance;
    float3 emittance;
    float ior;
};

#endif //SUNPATHTRACER_CUDA_MATERIAL_H