#include <iostream>
#include <vector>

#include <OpenGL/OpenGL.h>
#include <GLUT/glut.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "Scene.h"
#include "helper_cuda.h"
#include "cuda_camera.h"
#include "cuda_shape.h"
#include "cuda_scene.h"
#include "cuda_material.h"
#include "pathtracer.h"

auto constexpr WIDTH = 640;
auto constexpr HEIGHT = 480;

GLuint pbo;

cudaGraphicsResource_t resource;
float3* mc_buffer;

cudaScene device_scene;

void init()
{
    glClearColor(0.f, 0.f, 0.f, 0.f);

    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, sizeof(uchar4) * WIDTH * HEIGHT, NULL, GL_DYNAMIC_COPY);

    checkCudaErrors(cudaSetDevice(0));
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&resource, pbo, cudaGraphicsRegisterFlagsNone));

    checkCudaErrors(cudaMalloc((void**)&mc_buffer, sizeof(float3) * WIDTH * HEIGHT));
    checkCudaErrors(cudaMemset((void*)mc_buffer, 0, sizeof(float3) * WIDTH * HEIGHT));

    Scene host_scene;
    //build scene
    host_scene.AddMaterial(cudaMaterial());
    //table top
    host_scene.AddAABB(cudaAABB(make_float3(-0.5, -0.35, -0.5), make_float3(0.3, -0.3, 0.5), host_scene.GetLastMaterialID()));
    //table legs
    host_scene.AddAABB(cudaAABB(make_float3(-0.45, -1, -0.45), make_float3(-0.4, -0.35, -0.4), host_scene.GetLastMaterialID()));
    host_scene.AddAABB(cudaAABB(make_float3(0.2, -1, -0.45), make_float3(0.25, -0.35, -0.4), host_scene.GetLastMaterialID()));
    host_scene.AddAABB(cudaAABB(make_float3(-0.45, -1, 0.4), make_float3(-0.4, -0.35, 0.45), host_scene.GetLastMaterialID()));
    host_scene.AddAABB(cudaAABB(make_float3(0.2, -1, 0.4), make_float3(0.25, -0.35, 0.45), host_scene.GetLastMaterialID()));
    //chair set
    host_scene.AddAABB(cudaAABB(make_float3(0.3, -0.6, -0.2), make_float3(0.7, -0.55, 0.2), host_scene.GetLastMaterialID()));
    //chair legs
    host_scene.AddAABB(cudaAABB(make_float3(0.3, -1, -0.2), make_float3(0.35, -0.6, -0.15), host_scene.GetLastMaterialID()));
    host_scene.AddAABB(cudaAABB(make_float3(0.3, -1, 0.15), make_float3(0.35, -0.6, 0.2), host_scene.GetLastMaterialID()));
    host_scene.AddAABB(cudaAABB(make_float3(0.65, -1, -0.2), make_float3(0.7, 0.1, -0.15), host_scene.GetLastMaterialID()));
    host_scene.AddAABB(cudaAABB(make_float3(0.65, -1, 0.15), make_float3(0.7, 0.1, 0.2), host_scene.GetLastMaterialID()));
    //chair back
    host_scene.AddAABB(cudaAABB(make_float3(0.65, 0.05, -0.15), make_float3(0.7, 0.1, 0.15), host_scene.GetLastMaterialID()));
    host_scene.AddAABB(cudaAABB(make_float3(0.65, -0.55, -0.09), make_float3(0.7, 0.1, -0.03), host_scene.GetLastMaterialID()));
    host_scene.AddAABB(cudaAABB(make_float3(0.65, -0.55, 0.03), make_float3(0.7, 0.1, 0.09), host_scene.GetLastMaterialID()));
    //sphere on the table
    host_scene.AddSphere(cudaSphere(make_float3(-0.1, -0.05, 0), 0.25, host_scene.GetLastMaterialID()));

    //copy host scene to device scene
    host_scene.BuildSceneForGPU(device_scene);
}

void display()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    uchar4* img;
    size_t size;
    checkCudaErrors(cudaGraphicsMapResources(1, &resource));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&img, &size, resource));

    //todo: add rendering function call
    test(img);

    checkCudaErrors(cudaGraphicsUnmapResources(1, &resource));

    glDrawPixels(WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    
    glutSwapBuffers();
}

void keyboard(unsigned char key, int x, int y)
{
    switch(key)
    {
        case 27:
            exit(0);
            break;
        default:
            break;
    }
}

int main(int argc, char** argv)
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(WIDTH, HEIGHT);
    glutCreateWindow("SunPathTracer");
    init();

    glutKeyboardFunc(keyboard);
    glutDisplayFunc(display);

    glutMainLoop();

    return 0;
}