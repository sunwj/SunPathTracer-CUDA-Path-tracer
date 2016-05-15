//
// Created by 孙万捷 on 16/4/12.
//

#ifndef SUNPATHTRACER_CUDA_MATERIAL_H
#define SUNPATHTRACER_CUDA_MATERIAL_H

#include <cuda_runtime.h>

#define GLM_FORCE_INLINE
#include <glm/glm.hpp>

enum BSDFType{BSDF_DIFFUSE, BSDF_GLOSSY, BSDF_GLASS, BSDF_PLASTIC};

class cudaMaterial
{
public:
    cudaMaterial()
    {
        bsdf_type = BSDF_DIFFUSE;
        emition = glm::vec3(0.f);
        albedo = glm::vec3(0.8f);
        specular = glm::vec3(1.f);
        roughness = 0.f;
        ior = 1.f;
    }

public:
    glm::vec3 emition;
    glm::vec3 albedo;
    glm::vec3 specular;
    float ior;
    float roughness;
    BSDFType bsdf_type;
};

#endif //SUNPATHTRACER_CUDA_MATERIAL_H
