//
// Created by 孙万捷 on 16/5/16.
//

#ifndef SUNPATHTRACER_BBOX_H
#define SUNPATHTRACER_BBOX_H

#include <glm/glm.hpp>

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

    // initialize BBox with triangle vertices (bounding box of a triangle)
    BBox(const glm::vec3& v1, const glm::vec3& v2, const glm::vec3& v3)
    {
        bmin = glm::vec3(FLT_MAX);
        bmax = glm::vec3(-FLT_MAX);
        bcenter = glm::vec3(0.f);

        bmin = glm::min(bmin, v1);
        bmin = glm::min(bmin, v2);
        bmin = glm::min(bmin, v3);
        bmin -= 1e-6f;

        bmax = glm::max(bmax, v1);
        bmax = glm::max(bmax, v2);
        bmax = glm::max(bmax, v3);
        bmax += 1e-6f;

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

#endif //SUNPATHTRACER_BBOX_H
