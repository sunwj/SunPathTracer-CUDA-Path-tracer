//
// Created by 孙万捷 on 16/4/7.
//

#ifndef SUNPATHTRACER_PATHTRACER_H
#define SUNPATHTRACER_PATHTRACER_H

#include "render_parameters.h"

extern "C" void test(glm::u8vec4* img, cudaScene& scene, RenderParameters& params);

#endif //SUNPATHTRACER_PATHTRACER_H
