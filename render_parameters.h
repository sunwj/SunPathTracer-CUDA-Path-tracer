//
// Created by 孙万捷 on 16/4/18.
//

#ifndef SUNPATHTRACER_RENDER_PARAMETERS_H
#define SUNPATHTRACER_RENDER_PARAMETERS_H

#define GLM_FORCE_INLINE
#include <glm/fwd.hpp>
#include <glm/gtc/type_precision.hpp>

#include <cuda_runtime.h>

struct RenderParameters
{
    uint32_t iteration_count = 0;
    uint32_t rayDepth = 10;
    float exposure = 1.f;
    glm::vec3* hdr_buffer;
};

#endif //SUNPATHTRACER_RENDER_PARAMETERS_H
