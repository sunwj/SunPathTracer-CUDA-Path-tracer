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
#include "cuda_environment_light.h"

class Scene
{
public:
    Scene();
    ~Scene();

    void AddCamera(const cudaCamera& camera)
    {
        this->camera = camera;
    }

    void AddEnvLight(const cudaEnvironmentLight& env_light)
    {
        this->env_light = env_light;
    }

    void AddMaterial(const cudaMaterial& material)
    {
        this->materials.push_back(material);
    }

    void AddSphere(const cudaSphere& sphere)
    {
        this->spheres.push_back(sphere);
    }

    void AddAAB(const cudaAAB& aab)
    {
        this->aab.push_back(aab);
    }

    void AddPlane(const cudaPlane& plane)
    {
        this->planes.push_back(plane);
    }

    void AddMesh(const cudaMesh& mesh)
    {
        this->meshes.push_back(mesh);
    }

    unsigned int GetLastMaterialID(void) {return this->materials.size() - 1;}

    void BuildSceneForGPU(cudaScene& scene);

public:
    //camera
    cudaCamera camera;

    //environment light
    cudaEnvironmentLight env_light;

    //material
    std::vector<cudaMaterial> materials;

    //geometries
    std::vector<cudaSphere> spheres;
    std::vector<cudaAAB> aab;
    std::vector<cudaPlane> planes;
    std::vector<cudaMesh> meshes;
};


#endif //SUNPATHTRACER_SCENE_H
