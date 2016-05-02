//
// Created by 孙万捷 on 16/4/24.
//

#include "ObjMesh.h"

ObjMesh::ObjMesh(std::string filename)
{
    vmax = make_float3(-FLT_MAX);
    vmin = make_float3(FLT_MAX);

    Load(filename);
}

void ObjMesh::Load(std::string filename)
{
    std::ifstream input(filename);
    if(!input)
    {
        std::cerr<<"Unable to open file: "<<filename<<std::endl;
        exit(-1);
    }

    vertices.clear();
    faces.clear();
    vertex_normals.clear();
    face_normals.clear();

    char buffer[1024] = {0};
    while(input.getline(buffer, 1024))
    {
        if((buffer[0] == 'v') && (buffer[1] == ' '))
        {
            float3 v;
            sscanf(buffer, "v %f %f %f\n", &v.x, &v.y, &v.z);
            vmax = fmaxf(vmax, v);
            vmin = fminf(vmin, v);
            vertices.push_back(v);
        }
        else if((buffer[0] == 'f') && (buffer[1] == ' '))
        {
            uint3 f;
            sscanf(buffer, "f %u %u %u\n", &f.x, &f.y, &f.z);
            f = f - 1;
            faces.push_back(f);
        }
        else
            continue;
    }

    //translate mesh center into local coordinate's origin
    float3 center = (vmin + vmax) * 0.5f;
    vmin -= center;
    vmax -= center;
    for(auto& item : vertices)
        item -= center;

    //fix normal
    FixNormal();

#define __PRINT_INFO__
#ifdef __PRINT_INFO__
    std::cout<<filename<<" load successfully"<<std::endl;
    std::cout<<"Number of vertices: "<<vertices.size()<<std::endl;
    std::cout<<"Number of faces: "<<faces.size()<<std::endl;
    std::cout<<"Mesh extent:"<<std::endl
        <<"max: ("<<vmax.x<<", "<<vmax.y<<", "<<vmax.z<<")"<<std::endl
        <<"min: ("<<vmin.x<<", "<<vmin.y<<", "<<vmin.z<<")"<<std::endl;
    std::cout<<"Mesh diagnal length: "<<length(vmax - vmin)<<std::endl;
#endif
}

void ObjMesh::FixNormal()
{
    vertex_normals.reserve(vertices.size());
    face_normals.reserve(faces.size());

    for(auto& normal : vertex_normals)
        normal = make_float3(0.f);

    for(const auto& face : faces) {
        auto v1 = vertices[face.x];
        auto v2 = vertices[face.y];
        auto v3 = vertices[face.z];

        //select best normal
        auto e1 = v2 - v1;
        auto e2 = v3 - v2;
        auto e3 = v1 - v3;

        auto n1 = cross(e1, e2);
        auto n2 = cross(e2, e3);
        auto n3 = cross(e3, e1);

        auto l1 = length(n1);
        auto l2 = length(n2);
        auto l3 = length(n3);

        float3 n = make_float3(0.f);
        if ((l1 > l2) && (l1 > l3))
            n = n1 / l1;
        else if (l2 > l3)
            n = n2 / l2;
        else
            n = n3 / l3;

        face_normals.push_back(n);

        //add best normal to the face corresponding vertices
        vertex_normals[face.x] += n;
        vertex_normals[face.y] += n;
        vertex_normals[face.z] += n;

        //average vertice normal
        for (auto &normal : vertex_normals)
            normal = normalize(normal);
    }
}

void ObjMesh::ApplyTransform(Transformation& t)
{
    vmax = make_float3(-FLT_MAX);
    vmin = make_float3(FLT_MAX);

    for(auto& item : vertices)
    {
        item = t * item;
        vmin = fminf(vmin, item);
        vmax = fmaxf(vmax, item);
    }

#ifdef __PRINT_INFO__
    std::cout<<"After transformation:"<<std::endl;
    std::cout<<"Mesh extent:"<<std::endl
        <<"max: ("<<vmax.x<<", "<<vmax.y<<", "<<vmax.z<<")"<<std::endl
        <<"min: ("<<vmin.x<<", "<<vmin.y<<", "<<vmin.z<<")"<<std::endl;
    std::cout<<"Mesh diagnal length: "<<length(vmax - vmin)<<std::endl;
#endif
}