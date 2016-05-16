//
// Created by 孙万捷 on 16/4/21.
//

#ifndef SUNPATHTRACER_KERNEL_GLOBALS_H
#define SUNPATHTRACER_KERNEL_GLOBALS_H

#define GLM_FORCE_INLINE
#include <glm/glm.hpp>

#include <cuda_runtime.h>

//marcos
#define IMG_BLACK glm::u8vec4(0)

//data types
struct SurfaceElement
{
    float rayEpsilon;
    glm::vec3 pt;
    glm::vec3 normal;
    unsigned int matID;
};

//functions
__inline__ __device__ void running_estimate(glm::vec3& acc_buffer, const glm::vec3& curr_est, unsigned int N)
{
    acc_buffer += (curr_est - acc_buffer) / (N + 1.f);
}

__host__ __device__ unsigned int wangHash(unsigned int a)
{
    //http://raytracey.blogspot.com/2015/12/gpu-path-tracing-tutorial-2-interactive.html
    a = (a ^ 61) ^ (a >> 16);
    a = a + (a << 3);
    a = a ^ (a >> 4);
    a = a * 0x27d4eb2d;
    a = a ^ (a >> 15);

    return a;
}

__device__ bool scene_intersect(const cudaScene& scene, cudaRay& ray, SurfaceElement& se)
{
    bool intersected = false;

    float t = ray.tMax;
    for(auto i = 0; i < scene.num_spheres; ++i)
    {
        const cudaSphere& sphere = scene.spheres[i];
        if(sphere.Intersect(ray, &t))
        {
            ray.tMax = t;
            intersected = true;

            se.rayEpsilon = 0.0005f * ray.tMax;
            se.pt = ray.PointOnRay(ray.tMax);
            se.normal = sphere.GetNormal(se.pt);
            se.matID = sphere.material_id;
        }
    }

    for(auto i = 0; i < scene.num_aab; ++i)
    {
        const cudaAAB& aab = scene.aab[i];
        if(aab.Intersect(ray, &t))
        {
            ray.tMax = t;
            intersected = true;

            se.rayEpsilon = 0.0005f * ray.tMax;
            se.pt = ray.PointOnRay(ray.tMax);
            se.normal = aab.GetNormal(se.pt);
            se.matID = aab.material_id;
        }
    }

    for(auto i = 0; i < scene.num_planes; ++i)
    {
        const cudaPlane& plane = scene.planes[i];
        if(plane.Intersect(ray, &t))
        {
            ray.tMax = t;
            intersected = true;

            se.rayEpsilon = 0.001f * ray.tMax;
            se.pt = ray.PointOnRay(ray.tMax);
            se.normal = plane.GetNormal(se.pt);
            se.matID = plane.material_id;
        }
    }

    for(auto i = 0; i < scene.num_meshes; ++i)
    {
        const cudaMesh& mesh = scene.meshes[i];
        int32_t id = -1;
        if(mesh.Intersect(ray, &t, &id))
        {
            ray.tMax = t;
            intersected = true;

            se.rayEpsilon = 0.0005f * ray.tMax;
            se.pt = ray.PointOnRay(ray.tMax);
            se.normal = mesh.GetNormal(id);
            se.matID = mesh.material_id;
        }
    }

    return intersected;
}

#endif