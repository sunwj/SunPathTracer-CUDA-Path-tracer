//
// Created by 孙万捷 on 16/4/14.
//

#include "Scene.h"
#include "helper_cuda.h"

Scene::Scene()
{

}

Scene::~Scene()
{

}

void Scene::BuildSceneForGPU(cudaScene &scene)
{
    //copy camera
    scene.camera = this->camera;

    //copy environment light
    scene.env_light = this->env_light;

    //copy materials
    scene.num_materials = this->materials.size();
    if(scene.num_materials != 0)
    {
        checkCudaErrors(cudaMalloc((void**)&(scene.materials), sizeof(cudaMaterial) * this->materials.size()));
        checkCudaErrors(cudaMemcpy(scene.materials, this->materials.data(), sizeof(cudaMaterial) * this->materials.size(), cudaMemcpyHostToDevice));
    }

    //copy geometries
    scene.num_spheres = this->spheres.size();
    if(scene.num_spheres != 0)
    {
        checkCudaErrors(cudaMalloc((void**)&(scene.spheres), sizeof(cudaSphere) * this->spheres.size()));
        checkCudaErrors(cudaMemcpy(scene.spheres, this->spheres.data(), sizeof(cudaSphere) * this->spheres.size(), cudaMemcpyHostToDevice));
    }

    scene.num_aab = this->aab.size();
    if(scene.num_aab != 0)
    {
        checkCudaErrors(cudaMalloc((void**)&(scene.aab), sizeof(cudaAAB) * this->aab.size()));
        checkCudaErrors(cudaMemcpy(scene.aab, this->aab.data(), sizeof(cudaAAB) * this->aab.size(), cudaMemcpyHostToDevice));
    }

    scene.num_planes = this->planes.size();
    if(scene.num_planes != 0)
    {
        checkCudaErrors(cudaMalloc((void**)&(scene.planes), sizeof(cudaPlane) * this->planes.size()));
        checkCudaErrors(cudaMemcpy(scene.planes, this->planes.data(), sizeof(cudaPlane) * this->planes.size(), cudaMemcpyHostToDevice));
    }

    scene.num_meshes = this->meshes.size();
    if(scene.num_meshes != 0)
    {
        checkCudaErrors(cudaMalloc((void**)&(scene.meshes), sizeof(cudaMesh) * this->meshes.size()));
        checkCudaErrors(cudaMemcpy(scene.meshes, this->meshes.data(), sizeof(cudaMesh) * this->meshes.size(), cudaMemcpyHostToDevice));
    }
}