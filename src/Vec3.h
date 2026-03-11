#pragma once
#ifndef VEC3_H
#define VEC3_H

#include "cuda_runtime.h"
#include <ostream>
#include <thrust/execution_policy.h>

// ============================================================================
// Vec3<T>
// ---------------------------------------------------------------------------
// A simple templated 3D vector structure supporting basic arithmetic,
// normalization, dot/cross products, and CUDA host/device compatibility.
// ============================================================================
template<typename T>
struct Vec3
{
    T x, y, z; // Vector components

    // ------------------------------------------------------------------------
    // Constructors
    // ------------------------------------------------------------------------
    __host__ __device__ Vec3();                                     // Default constructor
    __host__ __device__ Vec3(T x_, T y_, T z_);                     // Parameterized constructor
    __host__ __device__ Vec3(const Vec3<T>& other);                 // Copy constructor

    // ------------------------------------------------------------------------
    // Assignment
    // ------------------------------------------------------------------------
    __host__ __device__ Vec3<T>& operator=(const Vec3<T>& other);   // Assignment operator

    // ------------------------------------------------------------------------
    // Unary operators
    // ------------------------------------------------------------------------
    __host__ __device__ Vec3<T> operator+() const;                  // Unary plus
    __host__ __device__ Vec3<T> operator-() const;                  // Unary minus

    // ------------------------------------------------------------------------
    // Binary arithmetic operators
    // ------------------------------------------------------------------------
    __host__ __device__ Vec3<T> operator+(const Vec3<T>& other) const;  // Vector addition
    __host__ __device__ void operator+=(const Vec3<T>& other);          // Addition assignment

    __host__ __device__ Vec3<T> operator-(const Vec3<T>& other) const;  // Vector subtraction
    __host__ __device__ void operator-=(const Vec3<T>& other);          // Subtraction assignment

    __host__ __device__ Vec3<T> operator*(const T scalar) const;        // Scalar multiplication
    __host__ __device__ void operator*=(const T scalar);                // Scalar multiplication assignment

    __host__ __device__ Vec3<T> operator/(const T scalar) const;        // Scalar division
    __host__ __device__ void operator/=(const T scalar);                // Scalar division assignment

    // ------------------------------------------------------------------------
    // Vector products
    // ------------------------------------------------------------------------
    __host__ __device__ T operator*(const Vec3<T>& other) const;        // Dot product
    __host__ __device__ Vec3<T> Cross(const Vec3<T>& other);            // Cross product

    // ------------------------------------------------------------------------
    // Vector properties
    // ------------------------------------------------------------------------
    __host__ __device__ T maxComponent() const;                         // Return maximum component
    __host__ __device__ T SquareModulus();                              // Squared magnitude
    __host__ __device__ T Modulus();                                    // Magnitude (length)
    __host__ __device__ void Normalize();                               // Normalize in-place
};

// ============================================================================
// Non-member operators
// ============================================================================
template<typename T>
__host__ __device__ inline Vec3<T> operator*(const T scalar, const Vec3<T>& vector); // Scalar * Vec3

template<typename T>
std::ostream& operator<<(std::ostream& out, const Vec3<T>& vec); // Stream output

// Include the corresponding implementation
#include "Vec3.cuh"

#endif // VEC3_H
