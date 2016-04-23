//
// Created by 孙万捷 on 16/4/12.
//

#ifndef SUNPATHTRACER_CUDA_MATERIAL_H
#define SUNPATHTRACER_CUDA_MATERIAL_H

#include <cuda_runtime.h>
#include "helper_math.h"

enum BSDFType{BSDF_DIFFUSE, BSDF_GLOSSY, BSDF_GLASS, BSDF_PLASTIC};

class cudaMaterial
{
public:
    cudaMaterial()
    {
        bsdf_type = BSDF_DIFFUSE;
        emition = make_float3(0.f);
        albedo = make_float3(0.8f);
        specular = make_float3(1.f);
        roughness = 0.f;
        ior = 1.f;
    }

public:
    float3 emition;
    float3 albedo;
    float3 specular;
    float ior;
    float roughness;
    BSDFType bsdf_type;
};

#endif //SUNPATHTRACER_CUDA_MATERIAL_H
