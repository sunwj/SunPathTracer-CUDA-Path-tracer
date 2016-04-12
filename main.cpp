#include <iostream>
#include <vector>

#include <OpenGL/OpenGL.h>
#include <GLUT/glut.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "cuda_camera.h"
#include "cuda_shape.h"
#include "pathtracer.h"

auto constexpr WIDTH = 640;
auto constexpr HEIGHT = 480;

GLuint pbo;

cudaGraphicsResource_t resource;

void init()
{
    glClearColor(0.f, 0.f, 0.f, 0.f);

    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, sizeof(uchar4) * WIDTH * HEIGHT, NULL, GL_DYNAMIC_COPY);

    cudaSetDevice(0);
    cudaGraphicsGLRegisterBuffer(&resource, pbo, cudaGraphicsRegisterFlagsNone);
}

void display()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    uchar4* img;
    size_t size;
    cudaGraphicsMapResources(1, &resource);
    cudaGraphicsResourceGetMappedPointer((void**)&img, &size, resource);

    cudaGraphicsUnmapResources(1, &resource);

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