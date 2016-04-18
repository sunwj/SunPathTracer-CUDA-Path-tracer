//
// Created by 孙万捷 on 16/4/12.
//

#ifndef SUNPATHTRACER_CUDA_MATERIAL_H
#define SUNPATHTRACER_CUDA_MATERIAL_H

#include <cuda_runtime.h>
#include "helper_math.h"

enum MATERIAL_TYPE{DIFFUSE, PERFECT_SPECULAR, REFRACTIVE};

class cudaMaterial
{
public:
    cudaMaterial()
    {
        MATERIAL_TYPE mat_type = DIFFUSE;
        albedo = make_float3(0.8f);
        emition = make_float3(0.f);
        ior = 1.f;
    }

public:
    union {
        float3 albedo;
        float3 specular_color;
    };
    float3 emition;
    float ior;
};

#endif //SUNPATHTRACER_CUDA_MATERIAL_H
