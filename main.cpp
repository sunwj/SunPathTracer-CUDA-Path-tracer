#include <iostream>
#include <vector>

#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "Scene.h"
#include "pathtracer.h"
#include "BVH.h"

auto constexpr WIDTH = 640;
auto constexpr HEIGHT = 480;

GLuint pbo;

cudaGraphicsResource_t resource;

RenderParameters renderParams;

cudaScene device_scene;

void init()
{
    glClearColor(0.f, 0.f, 0.f, 0.f);

    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, sizeof(uchar4) * WIDTH * HEIGHT, NULL, GL_DYNAMIC_COPY);

    checkCudaErrors(cudaSetDevice(0));
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&resource, pbo, cudaGraphicsRegisterFlagsNone));

    checkCudaErrors(cudaMalloc((void**)&(renderParams.hdr_buffer), sizeof(float3) * WIDTH * HEIGHT));
    checkCudaErrors(cudaMemset((void*)renderParams.hdr_buffer, 0, sizeof(float3) * WIDTH * HEIGHT));

    renderParams.exposure = 0.5f;
    Scene host_scene;
    //build scene
    /*
    //camera
    host_scene.AddCamera(cudaCamera(make_float3(3.f, 0.0f, 3.0f), make_float3(0.f, 0.0f, 0), make_float3(0.f, 1.f, 0.f), 15.f));

    cudaMaterial mat1;
    mat1.albedo = make_float3(0.7f, 0.5f, 0.3f);
    host_scene.AddMaterial(mat1);
    //table top
    host_scene.AddAABB(cudaAAB(make_float3(-0.5, -0.35, -0.5), make_float3(0.3, -0.3, 0.5), host_scene.GetLastMaterialID()));
    //table legs
    host_scene.AddAABB(cudaAAB(make_float3(-0.45, -1, -0.45), make_float3(-0.4, -0.35, -0.4), host_scene.GetLastMaterialID()));
    host_scene.AddAABB(cudaAAB(make_float3(0.2, -1, -0.45), make_float3(0.25, -0.35, -0.4), host_scene.GetLastMaterialID()));
    host_scene.AddAABB(cudaAAB(make_float3(-0.45, -1, 0.4), make_float3(-0.4, -0.35, 0.45), host_scene.GetLastMaterialID()));
    host_scene.AddAABB(cudaAAB(make_float3(0.2, -1, 0.4), make_float3(0.25, -0.35, 0.45), host_scene.GetLastMaterialID()));
    //chair set
    host_scene.AddAABB(cudaAAB(make_float3(0.3, -0.6, -0.2), make_float3(0.7, -0.55, 0.2), host_scene.GetLastMaterialID()));
    //chair legs
    host_scene.AddAABB(cudaAAB(make_float3(0.3, -1, -0.2), make_float3(0.35, -0.6, -0.15), host_scene.GetLastMaterialID()));
    host_scene.AddAABB(cudaAAB(make_float3(0.3, -1, 0.15), make_float3(0.35, -0.6, 0.2), host_scene.GetLastMaterialID()));
    host_scene.AddAABB(cudaAAB(make_float3(0.65, -1, -0.2), make_float3(0.7, 0.1, -0.15), host_scene.GetLastMaterialID()));
    host_scene.AddAABB(cudaAAB(make_float3(0.65, -1, 0.15), make_float3(0.7, 0.1, 0.2), host_scene.GetLastMaterialID()));
    //chair back
    host_scene.AddAABB(cudaAAB(make_float3(0.65, 0.05, -0.15), make_float3(0.7, 0.1, 0.15), host_scene.GetLastMaterialID()));
    host_scene.AddAABB(cudaAAB(make_float3(0.65, -0.55, -0.09), make_float3(0.7, 0.1, -0.03), host_scene.GetLastMaterialID()));
    host_scene.AddAABB(cudaAAB(make_float3(0.65, -0.55, 0.03), make_float3(0.7, 0.1, 0.09), host_scene.GetLastMaterialID()));

    //sphere on the table
    cudaMaterial mat2;
    //mat2.albedo = make_float3(0.448f, 0.8f, 0.666f);
    mat2.albedo = make_float3(1.f);
    mat2.bsdf_type = BSDF_GLASS;
    mat2.ior = 1.5f;
    mat2.roughness = 9999.f;
    host_scene.AddMaterial(mat2);
    host_scene.AddSphere(cudaSphere(make_float3(-0.1, 0.12, 0.1), 0.20, host_scene.GetLastMaterialID()));
    //host_scene.AddAABB(cudaAAB(make_float3(-0.3, -0.13, -0.1), make_float3(0.1, 0.27, 0.3), host_scene.GetLastMaterialID()));

    //cudaMaterial mat7;
    //mat7.albedo = make_float3(0.4f, 0.8f, 0.7f);
    //host_scene.AddMaterial(mat7);
    //host_scene.AddSphere(cudaSphere(make_float3(0.2, -0.099, -0.1), 0.20, host_scene.GetLastMaterialID()));

    ////walls
    cudaMaterial mat3;
    mat3.albedo = make_float3(1.f, 1.f, 1.f);
    host_scene.AddMaterial(mat3);
    //back
    host_scene.AddPlane(cudaPlane(make_float3(0.f, 0.f, -0.8f), normalize(make_float3(0.f, 0.f, 1.f)), host_scene.GetLastMaterialID()));
    //front
    host_scene.AddPlane(cudaPlane(make_float3(0.f, 0.f, 1.2f), normalize(make_float3(0.f, 0.f, -1.f)), host_scene.GetLastMaterialID()));
    //bottom
    host_scene.AddPlane(cudaPlane(make_float3(0.f, -1.f, 0.f), normalize(make_float3(0.f, 1.f, 0.f)), host_scene.GetLastMaterialID()));
    //top
    host_scene.AddPlane(cudaPlane(make_float3(0.f, 0.8f, 0.f), normalize(make_float3(0.f, -1.f, 0.f)), host_scene.GetLastMaterialID()));
    //left
    cudaMaterial mat4;
    mat4.albedo = make_float3(0.1f, 0.5f, 1.f);
    host_scene.AddMaterial(mat4);
    host_scene.AddPlane(cudaPlane(make_float3(-0.8f, 0.f, 0.f), normalize(make_float3(1.f, 0.f, 0.f)), host_scene.GetLastMaterialID()));
    //right
    cudaMaterial mat5;
    mat5.albedo = make_float3(1.0f, 0.9f, 0.1f);
    host_scene.AddMaterial(mat5);
    host_scene.AddPlane(cudaPlane(make_float3(0.8f, 0.f, 0.f), normalize(make_float3(-1.f, 0.f, 0.f)), host_scene.GetLastMaterialID()));

    //light
    cudaMaterial mat6;
    mat6.albedo = make_float3(1.f, 1.f, 1.f);
    mat6.emition = make_float3(2.f, 2.f, 2.f);
    host_scene.AddMaterial(mat6);
    host_scene.AddAABB(cudaAAB(make_float3(-0.38, 0.78, -0.25), make_float3(0.32, 0.8, 0.25), host_scene.GetLastMaterialID()));
    */

    /*
    //cornell box
    //camera
    host_scene.AddCamera(cudaCamera(make_float3(50.f, 52.f, 335.6f), make_float3(50.f, 52.f - 0.042612f, 335.6f - 1.f), make_float3(0.f, 1.f, 0.f), 25.f));

    //left
    cudaMaterial mat1;
    mat1.albedo = make_float3(0.75f, 0.0f, 0.0f);
    host_scene.AddMaterial(mat1);
    host_scene.AddSphere(cudaSphere(make_float3(1e5f + 1.0f, 40.8f, 81.6f), 1e5f, host_scene.GetLastMaterialID()));

    //right
    cudaMaterial mat2;
    mat2.albedo = make_float3(.0f, .75f, .0f);
    host_scene.AddMaterial(mat2);
    host_scene.AddSphere(cudaSphere(make_float3(-1e5f + 99.0f, 40.8f, 81.6f), 1e5f, host_scene.GetLastMaterialID()));

    //back
    cudaMaterial mat3;
    mat3.albedo = make_float3(.75f, .75f, .75f);
    host_scene.AddMaterial(mat3);
    host_scene.AddSphere(cudaSphere(make_float3(50.0f, 40.8f, 1e5f), 1e5f, host_scene.GetLastMaterialID()));

    //front
    cudaMaterial mat4;
    mat4.albedo = make_float3(1.00f, 1.00f, 1.00f);
    host_scene.AddMaterial(mat4);
    host_scene.AddSphere(cudaSphere(make_float3(50.0f, 40.8f, -1e5f + 600.0f), 1e5f, host_scene.GetLastMaterialID()));

    //top
    cudaMaterial mat6;
    mat6.albedo = make_float3(.75f, .75f, .75f);
    host_scene.AddMaterial(mat6);
    host_scene.AddSphere(cudaSphere(make_float3(50.0f, -1e5f + 81.6f, 81.6f), 1e5f, host_scene.GetLastMaterialID()));

    //bottom
    cudaMaterial mat5;
    mat5.albedo = make_float3(.75f, .75f, .75f);
    host_scene.AddMaterial(mat5);
    host_scene.AddSphere(cudaSphere(make_float3(50.0f, 1e5f, 81.6f), 1e5f, host_scene.GetLastMaterialID()));

    //light
    cudaMaterial mat7;
    mat7.albedo = make_float3(1.00f, 1.00f, 1.00f);
    mat7.emition = make_float3(2.f);
    host_scene.AddMaterial(mat7);
    host_scene.AddSphere(cudaSphere(make_float3(50.0f, 681.6f - .77f, 81.6f), 600.0f, host_scene.GetLastMaterialID()));

    //glass sphere
    cudaMaterial mat8;
    mat8.albedo = make_float3(1.00f, 1.00f, 1.00f);
    mat8.ior = 1.5;
    mat8.bsdf_type = BSDF_GLASS;
    mat8.roughness = 10000;
    host_scene.AddMaterial(mat8);
    host_scene.AddSphere(cudaSphere(make_float3(73.0f, 26.5f, 88.0f), 16.5f, host_scene.GetLastMaterialID()));
    */

    //triangle mesh
    host_scene.AddCamera(cudaCamera(make_float3(0.f, 35.0f, 140.0f), make_float3(0.f, 25.f, 0.f), make_float3(0.f, 1.f, 0.f), 25.f));

    //ground
    cudaMaterial mat1;
    mat1.albedo = make_float3(0.3f, 0.6f, 0.2f);
    mat1.bsdf_type = BSDF_PLASTIC;
    host_scene.AddMaterial(mat1);

    host_scene.AddPlane(cudaPlane(make_float3(0.f, 0.f, 0.f), make_float3(0.f, 1.f, 0.f), host_scene.GetLastMaterialID()));

    //back
    cudaMaterial mat4;
    //mat4.albedo = make_float3(0.9f, 0.3f, 0.3f);
    mat4.albedo = make_float3(0.9f);
    host_scene.AddMaterial(mat4);

    host_scene.AddPlane(cudaPlane(make_float3(0.f, 0.f, -50.f), make_float3(0.f, 0.f, 1.f), host_scene.GetLastMaterialID()));

    //front
    cudaMaterial mat8;
    mat8.albedo = make_float3(1.f);
    host_scene.AddMaterial(mat8);
    host_scene.AddPlane(cudaPlane(make_float3(0.f, 0.f, 160.f), make_float3(0.f, 0.f, -1.f), host_scene.GetLastMaterialID()));

    //left
    cudaMaterial mat5;
    mat5.albedo = make_float3(0.1f, 0.5f, 1.f);
    host_scene.AddMaterial(mat5);
    host_scene.AddPlane(cudaPlane(make_float3(-50.f, 0.f, 0.f), make_float3(1.f, 0.f, 0.f), host_scene.GetLastMaterialID()));

    //right
    cudaMaterial mat6;
    mat6.albedo = make_float3(1.0f, .9f, .1f);
    host_scene.AddMaterial(mat6);
    host_scene.AddPlane(cudaPlane(make_float3(50.f, 0.f, 0.f), make_float3(-1.f, 0.f, 0.f), host_scene.GetLastMaterialID()));

    //top
    host_scene.AddPlane(cudaPlane(make_float3(0.f, 100.f, 0.f), make_float3(0.f, -1.f, 0.f), host_scene.GetLastMaterialID()));

    //ball1
    cudaMaterial mat7;
    mat7.bsdf_type = BSDF_GLOSSY;
    //mat7.emition = make_float3(1.f);
    mat7.ior = 1.5f;
    //mat7.albedo = make_float3(0.9f, 0.4f, 0.7f);
    mat7.albedo = make_float3(1.f);
    mat7.roughness = 99999.f;
    host_scene.AddMaterial(mat7);

    host_scene.AddSphere(cudaSphere(make_float3(-25.f, 10.f, -6.f), 10.f, host_scene.GetLastMaterialID()));

    //light
    cudaMaterial mat2;
    mat2.albedo = make_float3(1.f);
    mat2.emition = make_float3(2.f);
    host_scene.AddMaterial(mat2);

    host_scene.AddSphere(cudaSphere(make_float3(0.f, 70.f, 30.f), 20.f, host_scene.GetLastMaterialID()));

    //objmesh
    ObjMesh mesh("monkey.obj");
    Transformation t;
    t.Scale(50.f / make_float3(length(mesh.vmax - mesh.vmin)));
    t.Translate(make_float3(0.f, 25.f, 10.f));
    mesh.ApplyTransform(t);
    BVH bvh(mesh);
    export_linear_bvh(bvh, "bvh.bvh");

    cudaMaterial mat3;
    mat3.albedo = make_float3(0.8f, 0.331f, 0.065f);
    //mat3.albedo = make_float3(0.f, 0.8f, 0.661f);
    mat3.albedo = make_float3(1.f);
    mat3.bsdf_type = BSDF_GLASS;
    mat3.ior = 1.5f;
    mat3.roughness = 99999.f;
    host_scene.AddMaterial(mat3);
    host_scene.AddMesh(create_cudaMesh(bvh, host_scene.GetLastMaterialID()));

    //copy host scene to device scene
    host_scene.BuildSceneForGPU(device_scene);
}

void render()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    uchar4* img;
    size_t size;
    checkCudaErrors(cudaGraphicsMapResources(1, &resource));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&img, &size, resource));

    //todo: add rendering function call
    for(auto i = 0; i < 5; ++i)
    {
        test(img, device_scene, renderParams);
        renderParams.iteration_count++;
    }

    checkCudaErrors(cudaGraphicsUnmapResources(1, &resource));

    glDrawPixels(WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if(key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GL_TRUE);
}

#include "ObjMesh.h"

int main(int argc, char** argv)
{
    GLFWwindow* window;
    if(!glfwInit())
        return -1;

    window = glfwCreateWindow(WIDTH, HEIGHT, "SunPathTracer", NULL, NULL);
    if(!window)
    {
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, key_callback);
    init();

    while(!glfwWindowShouldClose(window))
    {
        render();
        glfwSwapBuffers(window);

        glfwPollEvents();

        if(renderParams.iteration_count % 100 == 0)
            std::cout<<"iterations: "<<renderParams.iteration_count<<std::endl;
    }
    glfwDestroyWindow(window);
    glfwTerminate();
    cudaDeviceReset();

    return 0;

    //Transformation t;
    //ObjMesh mesh("buddha.obj");
    //BVH bvh(mesh);
    //export_linear_bvh(bvh, "bvh.bvh");
    //return 0;
}

