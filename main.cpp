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
unsigned int N = 0;

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
    //camera
    host_scene.AddCamera(cudaCamera(make_float3(0.f, 0.0f, 3.0f), make_float3(0.f, 0.0f, 0), make_float3(0.f, 1.f, 0.f), 35.f));
    //material
    cudaMaterial mat;
    mat.albedo = make_float3(0.7f, 0.7f, 0.7f);
    host_scene.AddMaterial(mat);
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
    mat.albedo = make_float3(0.8f, 0.8f, 0.8f);
    host_scene.AddMaterial(mat);
    host_scene.AddSphere(cudaSphere(make_float3(-0.1, -0.099, 0), 0.2, host_scene.GetLastMaterialID()));
    //walls
    mat.albedo = make_float3(0.8f, 0.8f, 0.8f);
    host_scene.AddMaterial(mat);
    //back
    host_scene.AddPlane(cudaPlane(make_float3(0.f, 0.f, -0.8f), normalize(make_float3(0.f, 0.f, 1.f)), host_scene.GetLastMaterialID()));
    //front
    host_scene.AddPlane(cudaPlane(make_float3(0.f, 0.f, 1.2f), normalize(make_float3(0.f, 0.f, -1.f)), host_scene.GetLastMaterialID()));
    //bottom
    host_scene.AddPlane(cudaPlane(make_float3(0.f, -1.f, 0.f), normalize(make_float3(0.f, 1.f, 0.f)), host_scene.GetLastMaterialID()));
    //top
    host_scene.AddPlane(cudaPlane(make_float3(0.f, 0.8f, 0.f), normalize(make_float3(0.f, -1.f, 0.f)), host_scene.GetLastMaterialID()));
    //left
    mat.albedo = make_float3(0.1f, 0.5f, 1.f);
    host_scene.AddMaterial(mat);
    host_scene.AddPlane(cudaPlane(make_float3(-0.8f, 0.f, 0.f), normalize(make_float3(1.f, 0.f, 0.f)), host_scene.GetLastMaterialID()));
    //right
    mat.albedo = make_float3(1.0f, 0.9f, 0.1f);
    host_scene.AddMaterial(mat);
    host_scene.AddPlane(cudaPlane(make_float3(0.8f, 0.f, 0.f), normalize(make_float3(-1.f, 0.f, 0.f)), host_scene.GetLastMaterialID()));
    //light
    mat.albedo = make_float3(1.f, 1.f, 1.f);
    mat.emition = make_float3(2.f, 2.f, 2.f);
    host_scene.AddMaterial(mat);
    host_scene.AddAABB(cudaAABB(make_float3(-0.4, 0.78, -0.25), make_float3(0.3, 0.8, 0.25), host_scene.GetLastMaterialID()));

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
    test(img, device_scene, mc_buffer, N);

    checkCudaErrors(cudaGraphicsUnmapResources(1, &resource));

    glDrawPixels(WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    
    glutSwapBuffers();
    N++;
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

void idle()
{
    glutPostRedisplay();
    if(N % 100 == 0)
        std::cout<<"iterations: "<<N<<std::endl;
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
    glutIdleFunc(idle);

    glutMainLoop();

    return 0;
}