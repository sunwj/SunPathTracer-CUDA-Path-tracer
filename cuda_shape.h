//
// Created by 孙万捷 on 16/2/6.
//

#ifndef SUNPATHTRACER_SHAPE_H
#define SUNPATHTRACER_SHAPE_H

#include <float.h>

#define GLM_FORCE_INLINE
#include <glm/glm.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

#include "helper_math.h"
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
    __host__ __device__ cudaSphere(const glm::vec3& _center, float _radius, unsigned int _material_id)
    {
        center = _center;
        radius = _radius;

        material_id = _material_id;
    }

    __device__ bool Intersect(const cudaRay& ray, float* t) const
    {
        // slove t ^ 2 * dir . dir + 2 * t * (o - c) . dir + (o - c) . (o - c) - radius ^ 2 = 0
        glm::vec3 L = ray.orig - center;
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

    __device__ glm::vec3 GetNormal(const glm::vec3& pt) const
    {
        return normalize(pt - center);
    }

public:
    glm::vec3 center;
    float radius;

    unsigned int material_id;
};

/***************************************************************************
 * cudaAAB
 ***************************************************************************/
class cudaAAB
{
public:
    __host__ __device__ cudaAAB(const glm::vec3& _bMin, const glm::vec3& _bMax, unsigned int _material_id)
    {
        bMax = _bMax;
        bMin = _bMin;

        material_id = _material_id;
    }

    __device__ bool Intersect(const cudaRay& ray, float* t) const
    {
        //glm::vec3 invDir = inverse(ray.dir);
        glm::vec3 invDir = 1.f / ray.dir;

        glm::vec3 tmin = (bMin - ray.orig) * invDir;
        glm::vec3 tmax = (bMax - ray.orig) * invDir;

        glm::vec3 real_min = glm::min(tmin, tmax);
        glm::vec3 real_max = glm::max(tmin, tmax);

        float minmax = fminf(fminf(real_max.x, real_max.y), real_max.z);
        float maxmin = fmaxf(fmaxf(real_min.x, real_min.y), real_min.z);

        constexpr float eps = 0.0001f;
        if((minmax >= maxmin) && (minmax > eps))
        {
            *t = maxmin;
            return true;
        }
        return false;
    }

    //todo: change if to index form
    static __device__ bool Intersect(const cudaRay& ray, const glm::vec3& bmin, const glm::vec3& bmax, const glm::vec3& invRayDir, float* t)
    {
        float boundmin, boundmax;
        if(invRayDir.x < 0.f)
        {
            boundmin = bmax.x;
            boundmax = bmin.x;
        }
        else
        {
            boundmin = bmin.x;
            boundmax = bmax.x;
        }
        float tmin = (boundmin - ray.orig.x) * invRayDir.x;
        float tmax = (boundmax - ray.orig.x) * invRayDir.x;

        if(invRayDir.y < 0.f)
        {
            boundmin = bmax.y;
            boundmax = bmin.y;
        }
        else
        {
            boundmin = bmin.y;
            boundmax = bmax.y;
        }
        float tymin = (boundmin - ray.orig.y) * invRayDir.y;
        float tymax = (boundmax - ray.orig.y) * invRayDir.y;

        if((tmin > tymax) || (tymin > tmax)) return false;
        tmin = fmaxf(tymin, tmin);
        tmax = fminf(tymax, tmax);

        if(invRayDir.z < 0.f)
        {
            boundmin = bmax.z;
            boundmax = bmin.z;
        }
        else
        {
            boundmin = bmin.z;
            boundmax = bmax.z;
        }
        float tzmin = (boundmin - ray.orig.z) * invRayDir.z;
        float tzmax = (boundmax - ray.orig.z) * invRayDir.z;

        if((tmin > tzmax) || (tzmin > tmax)) return false;
        tmin = fmaxf(tzmin, tmin);
        tmax = fminf(tzmax, tmax);

        *t = tmin;
        return tmax > 0.f;
    }

    __device__ glm::vec3 GetNormal(const glm::vec3& pt) const
    {
        glm::vec3 normal;
        constexpr float eps = 0.0001f;

        if(fabsf(bMin.x - pt.x) < eps) normal = glm::vec3(-1.f, 0.f, 0.f);
        else if(fabsf(bMax.x - pt.x) < eps) normal = glm::vec3(1.f, 0.f, 0.f);
        else if(fabsf(bMin.y - pt.y) < eps) normal = glm::vec3(0.f, -1.f, 0.f);
        else if(fabsf(bMax.y - pt.y) < eps) normal = glm::vec3(0.f, 1.f, 0.f);
        else if(fabsf(bMin.z - pt.z) < eps) normal = glm::vec3(0.f, 0.f, -1.f);
        else normal = glm::vec3(0.f, 0.f, 1.f);

        return normal;
    }

public:
    glm::vec3 bMax, bMin;

    unsigned int material_id;
};

/***************************************************************************
 * cudaTriangle
 ***************************************************************************/
class cudaTriangle
{
public:
    __host__ __device__ cudaTriangle(const glm::vec3& _v1, const glm::vec3& _v2, const glm::vec3& _v3):
        v1(_v1), v2(_v2), v3(_v3)
    {
        normal = normalize(cross(v2 - v1, v3 - v1));
    }

    __device__ bool Intersect(const cudaRay& ray, float* t) const
    {
        glm::vec3 edge1 = v2 - v1;
        glm::vec3 edge2 = v3 - v1;
        glm::vec3 pvec = cross(ray.dir, edge2);
        float det = dot(pvec, edge1);

        constexpr float eps = 1e-7;
        if(fabsf(det) < eps) return false;

        float invDet = 1.f / det;

        glm::vec3 tvec = ray.orig - v1;
        float u = dot(tvec, pvec) * invDet;
        if(u + eps < 0.f || u - eps > 1.f) return false;

        glm::vec3 qvec = cross(tvec, edge1);
        float v = dot(ray.dir, qvec) * invDet;
        if(v + eps < 0.f || (u + v) - eps > 1.f) return false;

        *t = dot(edge2, qvec) * invDet;

        return *t > eps;
    }

    static __device__ bool Intersect(const cudaRay& ray, const glm::vec3& v1, const glm::vec3& edge1, const glm::vec3& edge2, float* t)
    {
        glm::vec3 pvec = cross(ray.dir, edge2);
        float det = dot(pvec, edge1);

        constexpr float eps = 1e-7;
        if(fabsf(det) < eps) return false;

        float invDet = 1.f / det;

        glm::vec3 tvec = ray.orig - v1;
        float u = dot(tvec, pvec) * invDet;
        if(u < 0.f || u > 1.f) return false;

        glm::vec3 qvec = cross(tvec, edge1);
        float v = dot(ray.dir, qvec) * invDet;
        if(v < 0.f || (u + v) > 1.f) return false;

        *t = dot(edge2, qvec) * invDet;

        return *t > eps;
    }

    __device__ glm::vec3 GetNormal(const glm::vec3& pt) const
    {
        return normal;
    }

public:
    glm::vec3 v1, v2, v3;
    glm::vec3 normal;
};

/***************************************************************************
 * cudaPlane
 ***************************************************************************/
class cudaPlane
{
public:
    __host__ __device__ cudaPlane(const glm::vec3& _p, const glm::vec3& _normal, unsigned int _material_id)
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

    __device__ glm::vec3 GetNormal(const glm::vec3& pt) const
    {
        return normal;
    }
public:
    glm::vec3 p;
    glm::vec3 normal;
    unsigned int material_id;
};

/***************************************************************************
 * cudaMesh
 ***************************************************************************/
#ifdef __CUDACC__
#define TEX_FLOAT4(texobj, texcoord) (tex1D<float4>(texobj, texcoord))
#define TEX_FETCH_FLOAT4(texobj, texcoord) (tex1Dfetch<float4>(texobj, texcoord))
#else
#define TEX_FLOAT4(texobj, texcoord) (make_float4(0.f))
#define TEX_FETCH_FLOAT4(texobj, texcoord) (make_float4(0.f))
#endif

#define BVH_STACK_SIZE 32
class cudaMesh
{
public:
    __host__ cudaMesh(const BVH& bvh, unsigned int _material_id)
    {
        material_id = _material_id;
        CreateMesh(bvh);
    }

    __device__ bool Intersect(const cudaRay& ray, float* t, int32_t* id) const
    {
        glm::vec3 invRayDir = 1.f / ray.dir;
        //glm::vec3 invRayDir = inverse(ray.dir);
        float tmin = FLT_MAX;
        int stackTop = 0;
        uint32_t stack[BVH_STACK_SIZE] = {0};
        stack[stackTop++] = 0;

        while(stackTop)
        {
            uint32_t node_id = stack[--stackTop];

            LBVHNode node = bvhNodes[node_id];
            if(cudaAAB::Intersect(ray, node.bMin, node.bMax, invRayDir, t))
            {
                if(*t > tmin) continue;
                //inner node
                if(node.nPrimitives == 0)
                {
                    stack[stackTop++] = node.rightChildOffset;
                    stack[stackTop++] = node_id + 1;
                    if(stackTop >= BVH_STACK_SIZE) return false;
                }
                else //leaf node
                {
                    for(auto i = node.primitiveOffset; i < (node.primitiveOffset + node.nPrimitives); ++i)
                    {
                        if(*id == i) continue;
                        auto val = TEX_FETCH_FLOAT4(triangleTex, i * 4);
                        glm::vec3 v1 = glm::vec3(val.x, val.y, val.z);
                        val = TEX_FETCH_FLOAT4(triangleTex, i * 4 + 1);
                        glm::vec3 e1 = glm::vec3(val.x, val.y, val.z);
                        val = TEX_FETCH_FLOAT4(triangleTex, i * 4 + 2);
                        glm::vec3 e2 = glm::vec3(val.x, val.y, val.z);
                        if(cudaTriangle::Intersect(ray, v1, e1, e2, t) && *t < tmin)
                        {
                            tmin = *t;
                            *id = i;
                        }
                    }
                }
            }
        }

        *t = tmin;
        return *id != -1;
    }

    __device__ glm::vec3 GetNormal(uint32_t id) const
    {
        auto val = TEX_FETCH_FLOAT4(triangleTex, id * 4 + 3);
        return glm::vec3(val.x, val.y, val.z);
    }

    __host__ void CreateMesh(const BVH& bvh)
    {
        std::vector<glm::vec4> tri_list;
        tri_list.reserve(bvh.mesh.faces.size() * 4);
        for(const auto& face : bvh.mesh.faces)
        {
            auto v1 = bvh.mesh.vertices[face.x];
            auto v2 = bvh.mesh.vertices[face.y];
            auto v3 = bvh.mesh.vertices[face.z];

            //select best normal
            auto e1 = v2 - v1;
            auto e2 = v3 - v2;
            auto e3 = v1 - v3;

            auto n1 = glm::cross(e1, e2);
            auto n2 = glm::cross(e2, e3);
            auto n3 = glm::cross(e3, e1);

            auto l1 = glm::length(n1);
            auto l2 = glm::length(n2);
            auto l3 = glm::length(n3);

            glm::vec3 n = glm::vec3(glm::uninitialize);
            if ((l1 > l2) && (l1 > l3))
                n = n1 / l1;
            else if (l2 > l3)
                n = n2 / l2;
            else
                n = n3 / l3;

            tri_list.push_back(glm::vec4(v1, 0.f));
            tri_list.push_back(glm::vec4(v2 - v1, 0.f));
            tri_list.push_back(glm::vec4(v3 - v1, 0.f));
            tri_list.push_back(glm::vec4(n, 0.f));
        }

        std::cout<<tri_list.size()<<std::endl;
        //allocate buffer
        checkCudaErrors(cudaMalloc((void**)&triangleBuffer, sizeof(glm::vec4) * tri_list.size()));
        checkCudaErrors(cudaMemcpy(triangleBuffer, tri_list.data(), sizeof(glm::vec4) * tri_list.size(), cudaMemcpyHostToDevice));
        //specify texture
        cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(cudaResourceDesc));
        resDesc.resType = cudaResourceTypeLinear;
        resDesc.res.linear.devPtr = triangleBuffer;
        resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
        resDesc.res.linear.desc.x = 32;
        resDesc.res.linear.desc.y = 32;
        resDesc.res.linear.desc.z = 32;
        resDesc.res.linear.desc.w = 32;
        resDesc.res.linear.sizeInBytes = sizeof(glm::vec4) * tri_list.size();
        //specify texture object parameter
        cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(cudaTextureDesc));
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.filterMode = cudaFilterModePoint;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = false;
        //create texture object
        checkCudaErrors(cudaCreateTextureObject(&triangleTex, &resDesc, &texDesc, NULL));

        //copy bvh data
        checkCudaErrors(cudaMalloc((void**)&(bvhNodes), sizeof(LBVHNode) * bvh.lbvh.size()));
        checkCudaErrors(cudaMemcpy(bvhNodes, bvh.lbvh.data(), sizeof(LBVHNode) * bvh.lbvh.size(), cudaMemcpyHostToDevice));
    }

public:
    cudaTextureObject_t triangleTex;
    LBVHNode* bvhNodes;
    unsigned int material_id;

private:
    float* triangleBuffer;
};

#endif //SUNPATHTRACER_SHAPE_H
