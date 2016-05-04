//
// Created by 孙万捷 on 16/4/25.
//

#include <iostream>
#include "Transformation.h"

Matrix4::Matrix4()
{
    for(int i = 0; i < 4; ++i)
    {
        for(int j = 0; j < 4; ++j)
        {
            if(i == j)
                mat[i][j] = 1.f;
            else
                mat[i][j] = 0.f;
        }
    }
}

Matrix4 Matrix4::operator * (Matrix4 &m)
{
    Matrix4 tmp;
    for(int i = 0; i < 4; ++i)
    {
        for(int j = 0; j < 4; ++j)
        {
            float sum = 0;
            for(int k = 0; k < 4; ++k)
                sum += this->mat[i][k] * m.mat[k][j];
            tmp.mat[i][j] = sum;
        }
    }

    return tmp;
}

float& Matrix4::operator () (int i, int j)
{
    i = clamp(i, 0, 3);
    j = clamp(j, 0, 3);

    return mat[i][j];
}

/***************************************************************************
 * Transformation
 ***************************************************************************/
Transformation::Transformation()
{

}

void Transformation::LoadIdentity()
{
    for(int i = 0; i < 4; ++i)
    {
        for(int j = 0; j < 4; ++j)
        {
            if(i == j)
                mat(i, j) = 1.f;
            else
                mat(i, j) = 0.f;
        }
    }
}

void Transformation::Scale(const float3& s)
{
    mat(0, 0) *= s.x;
    mat(1, 1) *= s.y;
    mat(2, 2) *= s.z;
}

void Transformation::Translate(const float3 &t)
{
    mat(0, 3) += t.x;
    mat(1, 3) += t.y;
    mat(2, 3) += t.z;
}

void Transformation::Rotate(float r, AXIS axis)
{
    Matrix4 m;
    r = r * M_PI / 180.f;
    if(axis == AXIS_X)
    {
        m(1, 1) = cosf(r);
        m(1, 2) = -sinf(r);
        m(2, 1) = sinf(r);
        m(2, 2) = cosf(r);
    }

    if(axis == AXIS_Y)
    {
        m(0, 0) = cosf(r);
        m(0, 2) = sinf(r);
        m(2, 0) = -sinf(r);
        m(2, 2) = cos(r);
    }

    if(axis == AXIS_Z)
    {
        m(0, 0) = cosf(r);
        m(0, 1) = -sinf(r);
        m(1, 0) = sinf(r);
        m(1, 1) = cosf(r);
    }

    mat = m * mat;
}

float3 Transformation::operator * (const float3& v)
{
    float3 tmp;
    tmp.x = mat(0, 0) * v.x + mat(0, 1) * v.y + mat(0, 2) * v.z + mat(0, 3);
    tmp.y = mat(1, 0) * v.x + mat(1, 1) * v.y + mat(1, 2) * v.z + mat(1, 3);
    tmp.z = mat(2, 0) * v.x + mat(2, 1) * v.y + mat(2, 2) * v.z + mat(2, 3);

    return tmp;
}