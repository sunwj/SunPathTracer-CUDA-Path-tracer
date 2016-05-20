//
// Created by 孙万捷 on 16/4/25.
//

#ifndef SUNPATHTRACER_BVH_H
#define SUNPATHTRACER_BVH_H

#define GLM_FORCE_INLINE
#include <glm/glm.hpp>

#include "termcolor.hpp"
#include "BBox.h"
#include "ObjMesh.h"

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
