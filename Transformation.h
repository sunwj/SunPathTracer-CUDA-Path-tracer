//
// Created by 孙万捷 on 16/4/25.
//

#ifndef SUNPATHTRACER_TRANSFORMATION_H
#define SUNPATHTRACER_TRANSFORMATION_H

#include "helper_math.h"

class Matrix4
{
public:
    Matrix4();
    Matrix4 operator * (Matrix4& m);
    float& operator () (int i, int j);

public:
    float mat[4][4];
};

/***************************************************************************
 * Transformation
 ***************************************************************************/
class Transformation
{
    enum AXIS{AXIS_NONE, AXIS_X, AXIS_Y, AXIS_Z};
public:
    Transformation();
    void Scale(const float3& s);
    void Translate(const float3& t);
    void Rotate(float r, AXIS axis);
    float3 operator * (const float3& v);

public:
    Matrix4 mat;
};


#endif //SUNPATHTRACER_TRANSFORMATION_H
