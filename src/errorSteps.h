/**
 * @file errorSteps.h
 * @brief Declares CUDA-compatible functors for vector difference and maximum component operations.
 *
 * This header defines the template functors used in TDGL simulations to compute
 * local error estimates between polarization vectors and to extract the maximum
 * vector component for normalization or convergence checks.
 *
 * Functors declared:
 *  - delta_m<T>             : Computes element-wise absolute difference between two Vec3<T> objects.
 *  - MaxComponentFunctor<T> : Extracts the maximum component value from a Vec3<T>.
 */

#pragma once
#ifndef ERRORSTEPS_H
#define ERRORSTEPS_H

#include "Vec3.h"
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

 //------------------------------------------------------------------------------
 // Struct: delta_m
 //------------------------------------------------------------------------------
 //! @brief Functor to compute absolute per-component difference between two vectors.
 //!
 //! @tparam T Numeric type (e.g., float, double).
 //!
 //! This structure defines a device/host functor that calculates the absolute
 //! difference between corresponding components of two Vec3<T> vectors.
template <typename T>
struct delta_m {
    //! @brief Compute |final - prime| component-wise.
    //! @param final Final vector (Vec3<T>).
    //! @param prime Predicted or intermediate vector (Vec3<T>).
    //! @return Vec3<T> containing absolute differences per component.
    __host__ __device__
        Vec3<T> operator()(const Vec3<T> & final, const Vec3<T>& prime) const;
};

//------------------------------------------------------------------------------
// Struct: MaxComponentFunctor
//------------------------------------------------------------------------------
//! @brief Functor to extract the maximum component value of a Vec3<T>.
//!
//! @tparam T Numeric type (e.g., float, double).
//!
//! This structure defines a device/host functor that returns the largest
//! component among x, y, and z of a given Vec3<T>.
template <typename T>
struct MaxComponentFunctor {
    //! @brief Return the maximum component of a given Vec3<T>.
    //! @param vec Input vector.
    //! @return Maximum scalar component (x, y, or z).
    __host__ __device__
        T operator()(const Vec3<T>& vec) const;
};

//------------------------------------------------------------------------------
// Implementation include
//------------------------------------------------------------------------------
//! Includes CUDA-specific definitions for the declared functors.
#include "errorSteps.cuh"

#endif // ERRORSTEPS_H
