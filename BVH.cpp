//
// Created by 孙万捷 on 16/4/25.
//

#include "BVH.h"

#define MIN_LEAF_PRIM_NUM 16
#define MAX_LEAF_PRIM_NUM 255

static constexpr int nBuckets = 16;

BVH::BVH(ObjMesh& _mesh)
{
    mesh = _mesh;

    //build work list
    workList.reserve(mesh.faces.size());
    for(auto i = 0; i < mesh.faces.size(); ++i)
    {
        BVHPrimitiveInfo pInfo;
        pInfo.pIdx = i;
        pInfo.bounds = BBox(mesh.vertices[mesh.faces[i].x - 1], mesh.vertices[mesh.faces[i].y - 1], mesh.vertices[mesh.faces[i].z - 1]);
        workList.push_back(pInfo);
    }

    //recursive build
    orderedPrims.reserve(mesh.faces.size());
    root = RecursiveBuild(0, workList.size());

    //replace mesh faces order with ordered one
    mesh.faces.swap(orderedPrims);

    //build linear bvh
    lbvh.reserve(totalNodes);
    for(auto i = 0; i < totalNodes; ++i)
        lbvh.push_back(LBVHNode());
    float offset = 0;
    Flatten(root, &offset);
}

BVH::~BVH()
{
    Delete(root);
}

void BVH::Delete(BVHNode* node)
{
    if(node->left != NULL)
    {
        Delete(node->left);
    }
    if(node->right != NULL)
    {
        Delete(node->right);
    }
    delete node;
}

uint32_t BVH::Flatten(BVHNode *node, float* offset)
{
    LBVHNode* linearNode = &lbvh[*offset];
    linearNode->bounds = node->bounds;
    uint32_t myOffset = (*offset)++;
    if(node->nPrims > 0)
    {
        linearNode->primitiveOffset = node->firstPrimOffset;
        linearNode->nPrimitives = node->nPrims;
    }
    else
    {
        linearNode->nPrimitives = 0;
        Flatten(node->left, offset);
        linearNode->rightChildOffset = Flatten(node->right, offset);
    }

    return myOffset;
}

BVHNode* BVH::RecursiveBuild(uint32_t start, uint32_t end)
{
    totalNodes++;
    BVHNode* node = new BVHNode;

    //compute bounds of all primitives in BVH node
    BBox bbox;
    for(auto i = start; i < end; ++i)
        bbox = Union(bbox, workList[i].bounds);

    uint32_t nPrims = end - start;
    //if number of primitives are less than threshold, create leaf node
    if(nPrims < MIN_LEAF_PRIM_NUM)
    {
        uint32_t firstPrimOffset = orderedPrims.size();
        for(auto i = start; i < end; ++i)
        {
            auto pIdx = workList[i].pIdx;
            orderedPrims.push_back(mesh.faces[pIdx]);
        }
        node->InitLeaf(firstPrimOffset, nPrims, bbox);
    }
    else
    {
        //compute bound of primitive centroids, choose split dimension
        BBox centroidBounds;
        for(auto i = start; i < end; ++i)
            centroidBounds = Union(centroidBounds, workList[i].bounds.bcenter);

        int dim = centroidBounds.MaxExtent();

        //partition primitives into two sets and build children
        float mid = (end + start) / 2;
        if((get_by_idx(centroidBounds.bmax, dim) - get_by_idx(centroidBounds.bmin, dim)) < 1e-8)
        {
            uint32_t firstPrimOffset = orderedPrims.size();
            for(auto i = start; i < end; ++i)
            {
                auto pIdx = workList[i].pIdx;
                orderedPrims.push_back(mesh.faces[pIdx]);
            }
            node->InitLeaf(firstPrimOffset, nPrims, bbox);

            return node;
        }

        //partition primitives based on SAH
        BucketInfo buckets[nBuckets];
        float extent = get_by_idx(centroidBounds.bmax, dim) - get_by_idx(centroidBounds.bmin, dim);
        for(auto i = start; i < end; ++i)
        {
            int b = nBuckets * ((get_by_idx(workList[i].bounds.bcenter, dim) - get_by_idx(centroidBounds.bmin, dim)) / extent);
            if(b == nBuckets) b -= 1;
            buckets[b].count++;
            buckets[b].bounds = Union(buckets[b].bounds, workList[i].bounds);
        }

        //compute costs for splitting after each bucket
        float cost[nBuckets - 1];
        for(auto i = 0; i < nBuckets - 1; ++i)
        {
            BBox b0, b1;
            int count0 = 0, count1 = 0;

            for(auto j = 0; j <= i; ++j)
            {
                b0 = Union(b0, buckets[j].bounds);
                count0 += buckets[j].count;
            }
            for(auto j = i + 1; j < nBuckets; ++j)
            {
                b1 = Union(b1, buckets[j].bounds);
                count1 += buckets[j].count;
            }

            cost[i] = (count0 * b0.SurfaceArea() + count1 * b1.SurfaceArea()) / bbox.SurfaceArea();
        }

        //find best split
        float minCost = cost[0];
        uint32_t bestSplit = 0;
        for(auto i = 1; i < nBuckets - 1; ++i)
        {
            if(cost[i] < minCost)
            {
                minCost = cost[i];
                bestSplit = i;
            }
        }

        //either create leaf or split at selected SAH bucket
        if(nPrims > MAX_LEAF_PRIM_NUM || minCost < nPrims)
        {
            int b = -1;
            auto i = start;
            while((i < end) && (b <= bestSplit))
            {
                b = nBuckets * ((get_by_idx(workList[i].bounds.bcenter, dim) - get_by_idx(centroidBounds.bmin, dim)) / extent);
                if(b == nBuckets) b -= 1;
                mid = b;
                i++;
            }
        }
        else
        {
            uint32_t firstPrimOffset = orderedPrims.size();
            for(auto i = start; i < end; ++i)
            {
                auto pIdx = workList[i].pIdx;
                orderedPrims.push_back(mesh.faces[pIdx]);
            }
            node->InitLeaf(firstPrimOffset, nPrims, bbox);

            return node;
        }

        node->InitInner(RecursiveBuild(start, mid), RecursiveBuild(mid, end));
    }

    return node;
}