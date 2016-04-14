//
// Created by 孙万捷 on 16/4/14.
//

#ifndef SUNPATHTRACER_CUDA_SCENE_H
#define SUNPATHTRACER_CUDA_SCENE_H

#include <cuda_runtime.h>

#include "helper_cuda.h"
#include "cuda_shape.h"
#include "cuda_material.h"

class cudaScene
{
public:
    Scene()
    {
        spheres = NULL;
        aabb_boxes = NULL;

        materials = NULL;
    }

    ~Scene()
    {
        if(spheres != NULL)
            checkCudaErrors(cudaFree(spheres));

        if(aabb_boxes != NULL)
            checkCudaErrors(cudaFree(aabb_boxes));

        if(materials != NULL)
            checkCudaErrors(cudaFree(materials));
    }

public:
    unsigned int num_spheres;
    cudaSphere* spheres;

    unsigned int num_aabb_boxes;
    cudaAABB* aabb_boxes;

    unsigned int num_materials;
    cudaMaterial* materials;
};

#endif //SUNPATHTRACER_CUDA_SCENE_H
