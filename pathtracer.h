//
// Created by 孙万捷 on 16/4/7.
//

#ifndef SUNPATHTRACER_PATHTRACER_H
#define SUNPATHTRACER_PATHTRACER_H

extern "C" void test(uchar4* img, cudaScene& scene, float3* mc_buffer, unsigned int N);

#endif //SUNPATHTRACER_PATHTRACER_H
