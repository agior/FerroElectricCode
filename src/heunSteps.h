#ifndef heunSteps_H
#define heunSteps_H
#include "Vec3.h"
#include <thrust/execution_policy.h>

// Functor to update vector m based on slope k_m and time step dt
template<typename T>
struct updated_y {
    T dt;
    updated_y(T dt_);  // Constructor to initialize dt
    __host__ __device__ Vec3<T> operator()(const Vec3<T>& m, const Vec3<T>& k_m) const; // Update function
};

// Functor to calculate the average of two slopes, k1_m and k2_m
template<typename T>
struct k_avg {
    __host__ __device__ Vec3<T> operator()(const Vec3<T>& k1_m, const Vec3<T>& k2_m) const;
};

#include "heunSteps.cuh"

#endif // heunSteps_H
