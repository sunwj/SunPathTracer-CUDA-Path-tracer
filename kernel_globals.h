//
// Created by 孙万捷 on 16/4/21.
//

#ifndef SUNPATHTRACER_KERNEL_GLOBALS_H
#define SUNPATHTRACER_KERNEL_GLOBALS_H

#include <cuda_runtime.h>

//marcos
#define IMG_BLACK make_uchar4(0, 0, 0, 0)

//data types
struct SurfaceElement
{
    float rayEpsilon;
    float3 pt;
    float3 normal;
    unsigned int matID;
};

//functions
__inline__ __device__ void running_estimate(float3& acc_buffer, const float3& curr_est, unsigned int N)
{
    acc_buffer += (curr_est - acc_buffer) / (N + 1.f);
}

__host__ __device__ unsigned int wangHash(unsigned int a)
{
    //http://raytracey.blogspot.com/2015/12/gpu-path-tracing-tutorial-2-interactive.html
    a = (a ^ 61) ^ (a >> 16);
    a = a + (a << 3);
    a = a ^ (a >> 4);
    a = a * 0x27d4eb2d;
    a = a ^ (a >> 15);

    return a;
}

__device__ bool scene_intersect(const cudaScene& scene, const cudaRay& ray, SurfaceElement& se)
{
    bool intersected = false;
    float tmin = FLT_MAX;

    float t;
    for(int i = 0; i < scene.num_spheres; ++i)
    {
        const cudaSphere& sphere = scene.spheres[i];
        if(sphere.Intersect(ray, &t) && (t < tmin))
        {
            tmin = t;
            intersected = true;

            se.rayEpsilon = 0.0005f * tmin;
            se.pt = ray.PointOnRay(tmin);
            se.normal = sphere.GetNormal(se.pt);
            se.matID = sphere.material_id;
        }
    }

    for(int i = 0; i < scene.num_aabb_boxes; ++i)
    {
        const cudaAABB& aabb = scene.aabb_boxes[i];
        if(aabb.Intersect(ray, &t) && (t < tmin))
        {
            tmin = t;
            intersected = true;

            se.rayEpsilon = 0.0005f * tmin;
            se.pt = ray.PointOnRay(tmin);
            se.normal = aabb.GetNormal(se.pt);
            se.matID = aabb.material_id;
        }
    }

    for(int i = 0; i < scene.num_planes; ++i)
    {
        const cudaPlane& plane = scene.planes[i];
        if(plane.Intersect(ray, &t) && (t < tmin))
        {
            tmin = t;
            intersected = true;

            se.rayEpsilon = 0.0005f * tmin;
            se.pt = ray.PointOnRay(tmin);
            se.normal = plane.GetNormal(se.pt);
            se.matID = plane.material_id;
        }
    }

    return intersected;
}

/* Ray offset to avoid self intersection */

__device__ float3 ray_offset(float3 P, float3 Ng)
{
#ifdef __INTERSECTION_REFINE__
    const float epsilon_f = 1e-5f;
	/* ideally this should match epsilon_f, but instancing/mblur
	 * precision makes it problematic */
	constexpr float epsilon_test = 1.0f;
	constexpr int epsilon_i = 32;

	float3 res;

	/* x component */
	if(fabsf(P.x) < epsilon_test) {
		res.x = P.x + Ng.x*epsilon_f;
	}
	else {
		uint ix = __float_as_uint(P.x);
		ix += ((ix ^ __float_as_uint(Ng.x)) >> 31)? -epsilon_i : epsilon_i;
		res.x = __uint_as_float(ix);
	}

	/* y component */
	if(fabsf(P.y) < epsilon_test) {
		res.y = P.y + Ng.y*epsilon_f;
	}
	else {
		uint iy = __float_as_uint(P.y);
		iy += ((iy ^ __float_as_uint(Ng.y)) >> 31)? -epsilon_i : epsilon_i;
		res.y = __uint_as_float(iy);
	}

	/* z component */
	if(fabsf(P.z) < epsilon_test) {
		res.z = P.z + Ng.z*epsilon_f;
	}
	else {
		uint iz = __float_as_uint(P.z);
		iz += ((iz ^ __float_as_uint(Ng.z)) >> 31)? -epsilon_i : epsilon_i;
		res.z = __uint_as_float(iz);
	}

	return res;
#else
    constexpr float epsilon_f = 1e-4f;
    return P + epsilon_f * Ng;
#endif
}

#endif //SUNPATHTRACER_KERNEL_GLOBALS_H
