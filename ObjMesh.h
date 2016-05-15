//
// Created by 孙万捷 on 16/4/24.
//

#ifndef SUNPATHTRACER_OBJMESH_H
#define SUNPATHTRACER_OBJMESH_H

#include <iostream>
#include <fstream>
#include <vector>
#include <cfloat>

#include <glm/fwd.hpp>
#include <glm/vec3.hpp>

class ObjMesh
{
public:
    ObjMesh() {}
    ObjMesh(std::string filename);
    void Load(std::string filename);
    void ApplyTransform(const glm::mat4& t);

private:
    void FixNormal();

public:
    std::vector<glm::vec3> vertices;
    std::vector<glm::uvec3> faces;
    std::vector<glm::vec3> vertex_normals;
    std::vector<glm::vec3> face_normals;
    glm::vec3 vmax;
    glm::vec3 vmin;
};

#endif //SUNPATHTRACER_OBJMESH_H
