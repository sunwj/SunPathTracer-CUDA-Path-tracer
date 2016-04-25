//
// Created by 孙万捷 on 16/4/25.
//

#ifndef SUNPATHTRACER_BVH_H
#define SUNPATHTRACER_BVH_H

#include "ObjMesh.h"

class BBox
{
public:
    BBox()
    {
        bmin = make_float3(FLT_MAX);
        bmax = make_float3(-FLT_MAX);
        bcenter = make_float3(0.f);
    }

    BBox(const float3& _min, const float3& _max)
    {
        bmin = _min;
        bmax = _max;
        bcenter = (bmin + bmax) * 0.5f;
    }

    BBox(const float3& v1, const float3& v2, const float3& v3)
    {
        bmin = make_float3(FLT_MAX);
        bmax = make_float3(-FLT_MAX);
        bcenter = make_float3(0.f);

        bmin = fminf(bmin, v1);
        bmin = fminf(bmin, v2);
        bmin = fminf(bmin, v3);

        bmax = fmaxf(bmax, v1);
        bmax = fmaxf(bmax, v2);
        bmax = fmaxf(bmax, v3);

        bcenter = (bmin + bmax) * 0.5f;
    }

    int MaxExtent()
    {
        float3 diag = bmax - bmin;
        if((diag.x > diag.y) && (diag.x > diag.z))
            return 0;
        else if(diag.y > diag.z)
            return 1;
        else
            return 2;
    }

    float SurfaceArea()
    {
        float3 extent = bmax - bmin;
        return (extent.x * extent.y + extent.x * extent.z + extent.y * extent.z) * 2.f;
    }

public:
    float3 bmax;
    float3 bmin;
    float3 bcenter;
};

inline BBox Union(const BBox& b, const float3& v)
{
    return BBox(fminf(b.bmin, v), fmaxf(b.bmax, v));
}

inline BBox Union(const BBox& b1, const BBox& b2)
{
    return BBox(fminf(b1.bmin, b2.bmin), fmaxf(b1.bmax, b2.bmax));
}

class BVHPrimitiveInfo
{
public:
    size_t pIdx;
    BBox bounds;
};

class BucketInfo
{
public:
    BucketInfo() {count = 0;}

public:
    size_t count;
    BBox bounds;
};

class BVHNode
{
public:
    BVHNode()
    {
        left = NULL;
        right = NULL;
        firstPrimOffset = 0;
        nPrims = 0;
    }
    void InitLeaf(size_t first, size_t n, const BBox& b)
    {
        firstPrimOffset = first;
        nPrims = n;
        bounds = b;
    }
    void InitInner(BVHNode* l, BVHNode* r)
    {
        left = l;
        right = r;
        bounds = Union(l->bounds, r->bounds);
    }

public:
    BBox bounds;
    BVHNode* left;
    BVHNode* right;
    size_t firstPrimOffset;
    size_t nPrims;
};

class LBVHNode
{
public:
    LBVHNode()
    {
        nPrimitives = 0;
        primitiveOffset = 0;
    }
public:
    BBox bounds;
    union {
        uint32_t primitiveOffset;
        uint32_t rightChildOffset;
    };
    uint8_t nPrimitives;
};

class BVH
{
public:
    BVH(ObjMesh& _mesh);
    ~BVH();
    BVHNode* RecursiveBuild(uint32_t start, uint32_t end);
    uint32_t Flatten(BVHNode* node, float* offset);

private:
    void Delete(BVHNode* node);

public:
    ObjMesh mesh;
    std::vector<BVHPrimitiveInfo> workList;
    std::vector<uint3> orderedPrims;
    size_t totalNodes = 0;
    BVHNode* root;
    std::vector<LBVHNode> lbvh;
};

#endif //SUNPATHTRACER_BVH_H
