//
// Created by 孙万捷 on 16/4/24.
//

#ifndef SUNPATHTRACER_OBJMESH_H
#define SUNPATHTRACER_OBJMESH_H

#include <iostream>
#include <fstream>
#include <vector>
#include <cfloat>

#include "helper_math.h"
#include "Transformation.h"

class ObjMesh
{
public:
    ObjMesh() {}
    ObjMesh(std::string filename);
    void Load(std::string filename);
    void ApplyTransform(Transformation& t);

private:
    void FixNormal();

public:
    std::vector<float3> vertices;
    std::vector<uint3> faces;
    std::vector<float3> vertex_normals;
    std::vector<float3> face_normals;
    float3 vmax;
    float3 vmin;
};

#endif //SUNPATHTRACER_OBJMESH_H
