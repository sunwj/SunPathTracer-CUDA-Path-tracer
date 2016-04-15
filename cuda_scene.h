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
    cudaScene()
    {
        materials = NULL;

        spheres = NULL;
        aabb_boxes = NULL;
    }

    ~cudaScene()
    {
        //todo: fix it
        ////release materials
        //if(materials != NULL)
        //    checkCudaErrors(cudaFree(materials));
//
        ////release geometries
        //if(spheres != NULL)
        //    checkCudaErrors(cudaFree(spheres));
//
        //if(aabb_boxes != NULL)
        //    checkCudaErrors(cudaFree(aabb_boxes));
    }

public:
    //camera
    cudaCamera camera;

    //materials
    unsigned int num_materials;
    cudaMaterial* materials;

    //geometries
    unsigned int num_spheres;
    cudaSphere* spheres;

    unsigned int num_aabb_boxes;
    cudaAABB* aabb_boxes;
};

#endif //SUNPATHTRACER_CUDA_SCENE_H
