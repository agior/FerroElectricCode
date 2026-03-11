#ifndef heunSteps_CUH
#define heunSteps_CUH

#include "heunSteps.h"

// Constructor for updated_y functor to initialize dt
template<typename T>
__host__ __device__ updated_y<T>::updated_y(T dt_) : dt(dt_) {}

// Applies Heun’s method to update vector m using slope k_m and time step dt
template<typename T>
__host__ __device__ Vec3<T> updated_y<T>::operator()(const Vec3<T>& m, const Vec3<T>& k_m) const {
    Vec3<T> temp = m + k_m * dt;
    // temp.Normalize();
    return temp;
}

// Calculates the average of slopes k1_m and k2_m
template<typename T>
__host__ __device__ Vec3<T> k_avg<T>::operator()(const Vec3<T>& k1_m, const Vec3<T>& k2_m) const {
    return (k1_m + k2_m) * 0.5;
}

#endif // heunSteps_CUH
