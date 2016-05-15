//
// Created by 孙万捷 on 16/5/4.
//

#ifndef SUNPATHTRACER_UTILS_H
#define SUNPATHTRACER_UTILS_H

#include <vector>

#include <glm/fwd.hpp>

#include <cuda_runtime.h>

#include "helper_cuda.h"

cudaTextureObject_t create_environment_light_texture(const std::string& filename);

#endif //SUNPATHTRACER_UTILS_H
