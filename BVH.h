//
// Created by 孙万捷 on 16/4/25.
//

#ifndef SUNPATHTRACER_BVH_H
#define SUNPATHTRACER_BVH_H

#define GLM_FORCE_INLINE
#include <glm/glm.hpp>

#include "termcolor.hpp"
#include "ObjMesh.h"

class BBox
{
public:
    BBox()
    {
        bmin = glm::vec3(FLT_MAX);
        bmax = glm::vec3(-FLT_MAX);
        bcenter = glm::vec3(0.f);
    }

    BBox(const glm::vec3& _min, const glm::vec3& _max)
    {
        bmin = _min;
        bmax = _max;
        bcenter = (bmin + bmax) * 0.5f;
    }

    BBox(const glm::vec3& v1, const glm::vec3& v2, const glm::vec3& v3)
    {
        bmin = glm::vec3(FLT_MAX);
        bmax = glm::vec3(-FLT_MAX);
        bcenter = glm::vec3(0.f);

        bmin = glm::min(bmin, v1);
        bmin = glm::min(bmin, v2);
        bmin = glm::min(bmin, v3);

        bmax = glm::max(bmax, v1);
        bmax = glm::max(bmax, v2);
        bmax = glm::max(bmax, v3);

        bcenter = (bmin + bmax) * 0.5f;
    }

    BBox(const BBox& box)
    {
        this->bmin = box.bmin;
        this->bmax = box.bmax;
        this->bcenter = box.bcenter;
    }

    int MaxExtent()
    {
        glm::vec3 diag = bmax - bmin;
        if((diag.x > diag.y) && (diag.x > diag.z))
            return 0;
        else if(diag.y > diag.z)
            return 1;
        else
            return 2;
    }

    float SurfaceArea()
    {
        glm::vec3 extent = bmax - bmin;
        return (extent.x * extent.y + extent.x * extent.z + extent.y * extent.z) * 2.f;
    }

public:
    glm::vec3 bmax;
    glm::vec3 bmin;
    glm::vec3 bcenter;
};

inline BBox Union(const BBox& b, const glm::vec3& v)
{
    return BBox(glm::min(b.bmin, v), glm::max(b.bmax, v));
}

inline BBox Union(const BBox& b1, const BBox& b2)
{
    return BBox(glm::min(b1.bmin, b2.bmin), glm::max(b1.bmax, b2.bmax));
}

class BVHPrimitiveInfo
{
public:
    BVHPrimitiveInfo(uint32_t idx, const BBox& box)
    {
        pIdx = idx;
        bounds = box;
    }

public:
    uint32_t pIdx;
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

struct LBVHNode
{
    LBVHNode()
    {
        nPrimitives = 0;
        primitiveOffset = 0;
    }

    glm::vec3 bMax;
    glm::vec3 bMin;
    union {
        uint32_t primitiveOffset;
        uint32_t rightChildOffset;
    };
    uint32_t nPrimitives;
};

class BVH
{
public:
    BVH(ObjMesh& _mesh);
    ~BVH();
    BVHNode* RecursiveBuild(uint32_t start, uint32_t end, uint32_t depth = 0);
    uint32_t Flatten(BVHNode* node, uint32_t* offset);

private:
    void Delete(BVHNode* node);

public:
    ObjMesh mesh;
    std::vector<BVHPrimitiveInfo> workList;
    std::vector<glm::uvec3> orderedPrims;
    uint32_t totalNodes = 0;
    uint32_t maxDepth = 0;
    BVHNode* root;
    std::vector<LBVHNode> lbvh;
};

void export_linear_bvh(const BVH& bvh, std::string filename);

#endif //SUNPATHTRACER_BVH_H
