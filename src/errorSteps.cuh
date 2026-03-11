/**
 * @file errorSteps.cuh
 * @brief Defines CUDA-compatible functors for vector difference and maximum component operations.
 *
 * This header provides small device/host functors used for error estimation and comparison
 * between intermediate and final polarization vectors in TDGL-type solvers.
 *
 * Functors:
 *  - delta_m<T>: Computes absolute per-component difference between two Vec3<T> objects.
 *  - MaxComponentFunctor<T>: Returns the maximum component of a Vec3<T>.
 */

#pragma once
#ifndef ERRORSTEPS_CUH
#define ERRORSTEPS_CUH

 //------------------------------------------------------------------------------
 // Functor: delta_m
 //------------------------------------------------------------------------------
 //! @brief Computes element-wise absolute difference between final and prime vectors.
 //!
 //! @tparam T Numeric type (e.g., float, double).
 //! @param final Final vector (Vec3<T>).
 //! @param prime Intermediate or predicted vector (Vec3<T>).
 //! @return A Vec3<T> containing absolute differences of each component.
template <typename T>
__host__ __device__
Vec3<T> delta_m<T>::operator()(const Vec3<T> & final, const Vec3<T>& prime) const
{
    // Compute the absolute difference between each component of the two vectors
    return Vec3<T>(
        abs(final.x - prime.x),
        abs(final.y - prime.y),
        abs(final.z - prime.z)
    );
}

//------------------------------------------------------------------------------
// Functor: MaxComponentFunctor
//------------------------------------------------------------------------------
//! @brief Returns the maximum component value of a Vec3<T> instance.
//!
//! @tparam T Numeric type (e.g., float, double).
//! @param vec Input vector.
//! @return The largest scalar value among x, y, and z components.
template <typename T>
__host__ __device__
T MaxComponentFunctor<T>::operator()(const Vec3<T>& vec) const
{
    // Utilize Vec3<T>::maxComponent() to obtain the maximum element
    return vec.maxComponent();
}

#endif // ERRORSTEPS_CUH
