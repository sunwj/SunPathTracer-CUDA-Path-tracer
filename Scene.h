//
// Created by 孙万捷 on 16/4/14.
//

#ifndef SUNPATHTRACER_SCENE_H
#define SUNPATHTRACER_SCENE_H

#include <vector>
#include <cuda_runtime.h>

#include "cuda_material.h"
#include "cuda_camera.h"
#include "cuda_shape.h"
#include "cuda_scene.h"

class Scene
{
public:
    Scene();
    ~Scene();

    void AddCamera(const cudaCamera& camera)
    {
        this->camera = camera;
    }

    void AddMaterial(const cudaMaterial& material)
    {
        this->materials.push_back(material);
    }

    void AddSphere(const cudaSphere& sphere)
    {
        this->spheres.push_back(sphere);
    }

    void AddAABB(const cudaAABB& aabb)
    {
        this->aabb_boxes.push_back(aabb);
    }

    unsigned int GetLastMaterialID(void) {return this->materials.size() - 1;}

    //todo: add build scene
    void BuildSceneForGPU(cudaScene& scene);

public:
    //camera
    cudaCamera camera;

    //material
    std::vector<cudaMaterial> materials;

    //geometries
    std::vector<cudaSphere> spheres;
    std::vector<cudaAABB> aabb_boxes;
};


#endif //SUNPATHTRACER_SCENE_H
