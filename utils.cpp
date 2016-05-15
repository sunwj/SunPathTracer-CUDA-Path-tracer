//
// Created by 孙万捷 on 16/5/4.
//

#include <iostream>

#include <glm/glm.hpp>

#include "utils.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

cudaTextureObject_t create_environment_light_texture(const std::string& filename)
{
    int w = 0, h = 0, n = 0;
    float* data = stbi_loadf(filename.c_str(), &w, &h, &n, 0);
    if(!data)
    {
        std::cerr<<"Unable to load environment map: "<<filename<<std::endl;
        exit(0);
    }

    //create channel desc
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
    //create cudaArray
    cudaArray* array;
    checkCudaErrors(cudaMallocArray(&array, &channelDesc, w, h));
    if(n == 3)
    {
        uint32_t count = w * h;
        std::vector<float4> ext_data;
        ext_data.reserve(count);
        for(auto i = 0; i < count; ++i)
            ext_data.push_back(make_float4(data[i * 3], data[i * 3 + 1], data[i * 3 + 2], 0.f));

        checkCudaErrors(cudaMemcpyToArray(array, 0, 0, ext_data.data(), sizeof(float4) * w * h, cudaMemcpyHostToDevice));
    }
    else
        checkCudaErrors(cudaMemcpyToArray(array, 0, 0, data, sizeof(float4) * w * h, cudaMemcpyHostToDevice));
    //create resource desc
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = array;
    //create texture desc
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = true;
    //create cudaTextureObject
    cudaTextureObject_t tex;
    checkCudaErrors(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));

    return tex;
}