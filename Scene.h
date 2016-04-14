//
// Created by 孙万捷 on 16/4/14.
//

#ifndef SUNPATHTRACER_SCENE_H
#define SUNPATHTRACER_SCENE_H

#include <vector>
#include <cuda_runtime.h>

#include "cuda_shape.h"

class Scene
{
public:
    Scene();
    ~Scene();

public:
    std::vector<cudaSphere> spheres;
    std::vector<cudaAABB> aabb_boxes;
};


#endif //SUNPATHTRACER_SCENE_H
