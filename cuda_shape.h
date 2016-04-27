//
// Created by 孙万捷 on 16/2/6.
//

#ifndef SUNPATHTRACER_SHAPE_H
#define SUNPATHTRACER_SHAPE_H

#include <float.h>

#include <cuda_runtime.h>

#include "helper_cuda.h"
#include "cuda_ray.h"
#include "cuda_material.h"
#include "BVH.h"

/***************************************************************************
 * cudaSphere
 ***************************************************************************/
class cudaSphere
{
public:
    __host__ __device__ cudaSphere(const float3& _center, float _radius, unsigned int _material_id)
    {
        center = _center;
        radius = _radius;

        material_id = _material_id;
    }

    __device__ bool Intersect(const cudaRay& ray, float* t) const
    {
        // slove t ^ 2 * dir . dir + 2 * t * (o - c) . dir + (o - c) . (o - c) - radius ^ 2 = 0
        float3 L = ray.orig - center;
        //float a = dot(ray.dir, ray.dir);
        // ray.dir is normalized, so dir dot dir is cos(0) = 1
        float b = 2.f * dot(ray.dir, L);
        float c = dot(L, L) - radius * radius;

        float discr = b * b - 4.f * /*a **/ c;
        if(discr > 0.f)
        {
            discr = sqrtf(discr);

            constexpr float eps = 0.0001f;
            if((*t = (-b - discr) /*/ (2.f * a)*/ * 0.5f) > eps)
                return true;
            else if((*t = (-b + discr) /*/ (2.f * a)*/ * 0.5f) > eps)
                return true;
            else
                return false;
        }
        return false;
    }

    __device__ float3 GetNormal(const float3& pt) const
    {
        return normalize(pt - center);
    }

public:
    float3 center;
    float radius;

    unsigned int material_id;
};

/***************************************************************************
 * cudaAAB
 ***************************************************************************/
class cudaAAB
{
public:
    __host__ __device__ cudaAAB(const float3& _bMin, const float3& _bMax, unsigned int _material_id)
    {
        bMax = _bMax;
        bMin = _bMin;

        material_id = _material_id;
    }

    __device__ bool Intersect(const cudaRay& ray, float* t) const
    {
        float3 invDir = inverse(ray.dir);

        float3 tmin = (bMin - ray.orig) * invDir;
        float3 tmax = (bMax - ray.orig) * invDir;

        float3 real_min = fminf(tmin, tmax);
        float3 real_max = fmaxf(tmin, tmax);

        float minmax = fminf(fminf(real_max.x, real_max.y), real_max.z);
        float maxmin = fmaxf(fmaxf(real_min.x, real_min.y), real_min.z);

        if(minmax >= maxmin)
        {
            constexpr float eps = 0.0001f;
            if(maxmin > eps)
            {
                *t = maxmin;
                return true;
            }
        }

        return false;
    }

    static __device__ bool Intersect(const cudaRay& ray, const float3& bmin, const float3& bmax, float* t)
    {
        float3 invDir = inverse(ray.dir);

        float3 tmin = (bmin - ray.orig) * invDir;
        float3 tmax = (bmax - ray.orig) * invDir;

        float3 real_min = fminf(tmin, tmax);
        float3 real_max = fmaxf(tmin, tmax);

        float minmax = fminf(fminf(real_max.x, real_max.y), real_max.z);
        float maxmin = fmaxf(fmaxf(real_min.x, real_min.y), real_min.z);

        if(minmax >= maxmin)
        {
            *t = maxmin;
            return true;
        }
        return false;
    }

    __device__ float3 GetNormal(const float3& pt) const
    {
        float3 normal;
        constexpr float eps = 0.0001f;

        if(fabsf(bMin.x - pt.x) < eps) normal = make_float3(-1.f, 0.f, 0.f);
        else if(fabsf(bMax.x - pt.x) < eps) normal = make_float3(1.f, 0.f, 0.f);
        else if(fabsf(bMin.y - pt.y) < eps) normal = make_float3(0.f, -1.f, 0.f);
        else if(fabsf(bMax.y - pt.y) < eps) normal = make_float3(0.f, 1.f, 0.f);
        else if(fabsf(bMin.z - pt.z) < eps) normal = make_float3(0.f, 0.f, -1.f);
        else normal = make_float3(0.f, 0.f, 1.f);

        return normal;
    }

public:
    float3 bMax, bMin;

    unsigned int material_id;
};

/***************************************************************************
 * cudaTriangle
 ***************************************************************************/
class cudaTriangle
{
public:
    __host__ __device__ cudaTriangle(const float3& _v1, const float3& _v2, const float3& _v3):
        v1(_v1), v2(_v2), v3(_v3)
    {
        normal = normalize(cross(v2 - v1, v3 - v1));
    }

    __device__ bool Intersect(const cudaRay& ray, float* t) const
    {
        float3 edge1 = v2 - v1;
        float3 edge2 = v3 - v1;
        float3 pvec = cross(ray.dir, edge2);
        float det = dot(pvec, edge1);

        constexpr float eps = 0.0001f;
        if(fabsf(det) < eps) return false;

        float invDet = 1.f / det;

        float3 tvec = ray.orig - v1;
        float u = dot(tvec, pvec) * invDet;
        if(u < 0 || u > 1) return false;

        float3 qvec = cross(tvec, edge1);
        float v = dot(ray.dir, qvec) * invDet;
        if(v < 0 || (u + v) > 1) return false;

        *t = dot(edge2, qvec) * invDet;

        return *t > eps;
    }

    static __device__ bool Intersect(const cudaRay& ray, const float3& v1, const float3& edge1, const float3& edge2, float* t)
    {
        float3 pvec = cross(ray.dir, edge2);
        float det = dot(edge1, pvec);

        constexpr float eps = 0.0001f;
        if(fabsf(det) < eps) return false;

        float invDet = 1.f / det;

        float3 tvec = ray.orig - v1;
        float u = dot(tvec, pvec) * invDet;
        if(u < 0 || u > 1) return false;

        float3 qvec = cross(tvec, edge1);
        float v = dot(ray.dir, qvec) * invDet;
        if(v < 0 || (u + v) > 1) return false;

        *t = dot(edge2, qvec) * invDet;

        return *t > eps;
    }

    __device__ float3 GetNormal(const float3& pt) const
    {
        return normal;
    }

public:
    float3 v1, v2, v3;
    float3 normal;
};

/***************************************************************************
 * cudaPlane
 ***************************************************************************/
class cudaPlane
{
public:
    __host__ __device__ cudaPlane(const float3& _p, const float3& _normal, unsigned int _material_id)
    {
        p = _p;
        normal = _normal;

        material_id = _material_id;
    }

    __device__ bool Intersect(const cudaRay& ray, float* t) const
    {
        // t = ((p - ray.orig) . normal) / (ray.dir . normal)
        float denom = -dot(ray.dir, normal);
        if(denom > 1e-6)
        {
            constexpr float eps = 0.0001f;
            *t = -dot(p - ray.orig, normal) / denom;
            return (*t > eps);
        }

        return false;
    }

    __device__ float3 GetNormal(const float3& pt) const
    {
        return normal;
    }
public:
    float3 p;
    float3 normal;
    unsigned int material_id;
};

/***************************************************************************
 * cudaMesh
 ***************************************************************************/
#define BVH_STACK_SIZE 32
class cudaMesh
{
public:
    __host__ __device__ cudaMesh(LBVHNode* _bvh, float3* _triangles, unsigned int _material_id):
        bvh(_bvh), triangles(_triangles), material_id(_material_id)
    {}

    __device__ bool Intersect(const cudaRay& ray, float* t, uint32_t* id) const
    {
        int stack_top = 0;
        uint32_t stack[BVH_STACK_SIZE] = {0};

        float tmin = FLT_MAX;

        return false;
    }

    __device__ float3 GetNormal(uint32_t id) const
    {
        return normalize(cross(triangles[id * 3 + 1], triangles[id * 3 + 2]));
    }

public:
    LBVHNode* bvh;
    float3* triangles;
    unsigned int material_id;
};

inline cudaMesh create_cudaMesh(const BVH& bvh, unsigned int material_id)
{
    std::vector<float3> triangles;
    triangles.reserve(bvh.mesh.faces.size() * 3);
    for(auto& item : bvh.mesh.faces)
    {
        float3 v1 = bvh.mesh.vertices[item.x];
        float3 v2 = bvh.mesh.vertices[item.y];
        float3 v3 = bvh.mesh.vertices[item.z];

        triangles.push_back(v1);
        triangles.push_back(v2 - v1);
        triangles.push_back(v3 - v1);
    }

    float3* d_triangles;
    checkCudaErrors(cudaMalloc((void**)&(d_triangles), sizeof(float3) * triangles.size()));
    checkCudaErrors(cudaMemcpy(d_triangles, triangles.data(), sizeof(float3) * triangles.size(), cudaMemcpyHostToDevice));

    LBVHNode* d_bvh;
    checkCudaErrors(cudaMalloc((void**)&(d_bvh), sizeof(LBVHNode) * bvh.lbvh.size()));
    checkCudaErrors(cudaMemcpy(d_bvh, bvh.lbvh.data(), sizeof(LBVHNode) * bvh.lbvh.size(), cudaMemcpyHostToDevice));

    return cudaMesh(d_bvh, d_triangles, material_id);
}

#endif //SUNPATHTRACER_SHAPE_H
