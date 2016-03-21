//
// Created by 孙万捷 on 16/3/21.
//

#ifndef SUNPATHTRACER_SAMPLING_H
#define SUNPATHTRACER_SAMPLING_H

#include <curand.h>
#include <curand_kernel.h>

#include "helper_math.h"

// return r and theta in polar coordinate
__inline__ float2 uniform_sample_disk(curandState* rng)
{
    float r = sqrtf(curand_uniform(rng));
    float theta = curand_uniform(rng) * 2.f * M_PI;

    return make_float2(r, theta);
}

// return x and y in cartesian coordinate
__inline__ float2 uniform_sample_disk(curandState* rng, float r)
{
    float2 p = uniform_sample_disk(rng);

    return make_float2(cosf(p.x), sinf(p.y)) * r;
}

#endif //SUNPATHTRACER_SAMPLING_H
