#pragma once 
#ifndef RK5STEPS_H 
#define RK5STEPS_H
#include "Vec3.h" 
#include <thrust/functional.h>
#include <cmath>
#include <thrust/execution_policy.h>

// Define constants for RK45 method
__constant__ Type_var bb21 = 1.0 / 5.0;
__constant__ Type_var bb31 = 3.0 / 40.0;
__constant__ Type_var bb32 = 9.0 / 40.0;
__constant__ Type_var bb41 = 3.0 / 10.0;
__constant__ Type_var bb42 = -9.0 / 10.0;
__constant__ Type_var bb43 = 6.0 / 5.0;
__constant__ Type_var bb51 = -11.0 / 54.0;
__constant__ Type_var bb52 = 5.0 / 2.0;
__constant__ Type_var bb53 = -70.0 / 27.0;
__constant__ Type_var bb54 = 35.0 / 27.0;
__constant__ Type_var bb61 = 1631.0 / 55296.0;
__constant__ Type_var bb62 = 175.0 / 512.0;
__constant__ Type_var bb63 = 575.0 / 13824.0;
__constant__ Type_var bb64 = 44275.0 / 110592.0;
__constant__ Type_var bb65 = 253.0 / 4096.0;

__constant__ Type_var cc1 = 37.0 / 378.0;
__constant__ Type_var cc2 = 0.0;
__constant__ Type_var cc3 = 250.0 / 621.0;
__constant__ Type_var cc4 = 125.0 / 594.0;
__constant__ Type_var cc5 = 0.0;
__constant__ Type_var cc6 = 512.0 / 1771.0;

__constant__ Type_var cc_1 = 2825.0 / 27648.0;
__constant__ Type_var cc_2 = 0.0;
__constant__ Type_var cc_3 = 18575.0 / 48384.0;
__constant__ Type_var cc_4 = 13525.0 / 55296.0;
__constant__ Type_var cc_5 = 277.0 / 14336.0;
__constant__ Type_var cc_6 = 1.0 / 4.0;


// Template structure for k_1
template<typename T>
struct y_2 {
    Vec3<T>* d_m; // Pointer to polarization vector
    Vec3<T>* k_1; // Pointer to effective field vector
    T dt; // Time step for integration

    // Constructor
    __host__ __device__ y_2(Vec3<T>* d_m_, Vec3<T>* k_1_, T dt_);

    // Operator for computation
    __host__ __device__ Vec3<T> operator()(int idx) const;
};

// Template structure for k_2
template<typename T>
struct y_3 {
    Vec3<T>* d_m; // Pointer to polarization vector
    Vec3<T>* k_1; // Pointer to k_1 field vector
    Vec3<T>* k_2; // Pointer to k_2 field vector
    T dt; // Time step for integration

    // Constructor
    __host__ __device__ y_3(Vec3<T>* d_m_, Vec3<T>* k_1_, Vec3<T>* k_2_, T dt_);

    // Operator for computation
    __host__ __device__ Vec3<T> operator()(int idx) const;
};

// Template structure for k_3
template<typename T>
struct y_4 {
    Vec3<T>* d_m; // Pointer to polarization vector
    Vec3<T>* k_1; // Pointer to k_1 field vector
    Vec3<T>* k_2; // Pointer to k_2 field vector
    Vec3<T>* k_3; // Pointer to k_3 field vector
    T dt; // Time step for integration

    // Constructor
    __host__ __device__ y_4(Vec3<T>* d_m_, Vec3<T>* k_1_, Vec3<T>* k_2_, Vec3<T>* k_3_, T dt_);

    // Operator for computation
    __host__ __device__ Vec3<T> operator()(int idx) const;
};

// Template structure for k_4
template<typename T>
struct y_5 {
    Vec3<T>* d_m; // Pointer to polarization vector
    Vec3<T>* k_1; // Pointer to k_1 field vector
    Vec3<T>* k_2; // Pointer to k_2 field vector
    Vec3<T>* k_3; // Pointer to k_3 field vector
    Vec3<T>* k_4; // Pointer to k_4 field vector
    T dt; // Time step for integration

    // Constructor
    __host__ __device__ y_5(Vec3<T>* d_m_, Vec3<T>* k_1_, Vec3<T>* k_2_, Vec3<T>* k_3_, Vec3<T>* k_4_, T dt_);

    // Operator for computation
    __host__ __device__ Vec3<T> operator()(int idx) const;
};

// Template structure for k_5
template<typename T>
struct y_6 {
    Vec3<T>* d_m; // Pointer to polarization vector
    Vec3<T>* k_1; // Pointer to k_1 field vector
    Vec3<T>* k_2; // Pointer to k_2 field vector
    Vec3<T>* k_3; // Pointer to k_3 field vector
    Vec3<T>* k_4; // Pointer to k_4 field vector
    Vec3<T>* k_5; // Pointer to k_5 field vector
    T dt; // Time step for integration

    // Constructor
    __host__ __device__ y_6(Vec3<T>* d_m_, Vec3<T>* k_1_, Vec3<T>* k_2_, Vec3<T>* k_3_, Vec3<T>* k_4_, Vec3<T>* k_5_, T dt_);

    // Operator for computation
    __host__ __device__ Vec3<T> operator()(int idx) const;
};

// Template structure for final_y
template<typename T>
struct final_y {
    Vec3<T>* d_m; // Pointer to polarization vector
    Vec3<T>* k_1; // Pointer to k_1 field vector
    Vec3<T>* k_2; // Pointer to k_2 field vector
    Vec3<T>* k_3; // Pointer to k_3 field vector
    Vec3<T>* k_4; // Pointer to k_4 field vector
    Vec3<T>* k_5; // Pointer to k_5 field vector
    Vec3<T>* k_6; // Pointer to k_6 field vector
    T dt; // Time step for integration

    // Constructor
    __host__ __device__ final_y(Vec3<T>* d_m_, Vec3<T>* k_1_, Vec3<T>* k_2_, Vec3<T>* k_3_, Vec3<T>* k_4_,
        Vec3<T>* k_5_, Vec3<T>* k_6_, T dt_);

    // Operator for computation
    __host__ __device__ Vec3<T> operator()(int idx) const;
};

// Template structure for prime_y
template<typename T>
struct prime_y {
    Vec3<T>* d_m; // Pointer to polarization vector
    Vec3<T>* k_1; // Pointer to k_1 field vector
    Vec3<T>* k_2; // Pointer to k_2 field vector
    Vec3<T>* k_3; // Pointer to k_3 field vector
    Vec3<T>* k_4; // Pointer to k_4 field vector
    Vec3<T>* k_5; // Pointer to k_5 field vector
    Vec3<T>* k_6; // Pointer to k_6 field vector
    T dt; // Time step for integration

    // Constructor
    __host__ __device__ prime_y(Vec3<T>* d_m_, Vec3<T>* k_1_, Vec3<T>* k_2_, Vec3<T>* k_3_, Vec3<T>* k_4_,
        Vec3<T>* k_5_, Vec3<T>* k_6_, T dt_);
    // Operator for computation
    __host__ __device__ Vec3<T> operator()(int idx) const;
};

#include "rk5Steps.cuh" // Include implementation file

#endif // End include guard
