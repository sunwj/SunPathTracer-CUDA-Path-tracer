//
// Created by 孙万捷 on 16/2/6.
//

#ifndef SUNPATHTRACER_RAY_H
#define SUNPATHTRACER_RAY_H

#define GLM_FORCE_INLINE
#include <glm/glm.hpp>

#include <cuda_runtime.h>

class cudaRay
{
public:
    __device__ cudaRay() {}
    __device__ cudaRay(const glm::vec3& orig, const glm::vec3& dir)
    {
        this->orig = orig;
        this->dir = dir;
    }

    __device__ glm::vec3 PointOnRay(float t) const
    {
        return orig + t * dir;
    }
public:
    glm::vec3 orig;
    glm::vec3 dir;
};

#endif //SUNPATHTRACER_RAY_H
